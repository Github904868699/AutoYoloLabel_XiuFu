import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sampro.device import resolve_device
from sampro.sam2.build_sam import build_sam2_video_predictor
from util.xmlfile import xml_message

SAMPRO_ROOT = Path(__file__).resolve().parent

class AnythingVideo_TW():
    def __init__(self):
        # SAM2 模型配置
        checkpoint_path = os.getenv(
            "SAM2_CHECKPOINT",
            SAMPRO_ROOT / "checkpoints" / "sam2.1_hiera_large.pt",
        )
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"未找到 SAM2 权重文件: {checkpoint_path}. 请将模型文件放在 sampro/checkpoints/ 目录"
            )

        self.sam2_checkpoint = str(checkpoint_path)
        self.model_cfg = os.getenv("SAM2_MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
        self.device = resolve_device()
        self.video_path = ""
        self.output_path = ""
        self.predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint, self.device)

        # 全局变量
        self.frame = None

        self.option = False
        self.clicked_x = None
        self.clicked_y = None
        self.method = None

        self.inference_state = None
        self.out_obj_ids = None
        self.out_mask_logits = None
        self.video_segments = {}
        self.target_points = {}
        self.target_methods = {}
        self.target_labels = {}
        self.object_boxes = {}

        # 矩形框
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

    def set_video(self, video_dir):
        self.video_path = video_dir
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))  # 根据文件名排序

        # 加载第一帧
        frame_idx = 5
        frame_name = frame_names[frame_idx]
        frame_path = os.path.join(video_dir, frame_name)
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame = frame
        return frame

    def inference(self, video_dir):
        self.inference_state = self.predictor.init_state(video_path=video_dir)
        self.predictor.reset_state(self.inference_state)

    def extract_frames_from_video(self, video_path, output_dir, fps=24):
        """
        从视频中提取帧并保存为图片
        Args:
            video_path: 输入视频的路径
            output_dir: 输出图片的文件夹路径
            fps: 每秒提取的帧数，默认为2
        Returns:
            output_dir: 保存帧的文件夹路径
        Raises:
            ValueError: 当fps超过24或视频文件无法打开时
        """
        # 检查fps是否超过限制
        if fps > 24:
            raise ValueError(f"fps不能超过24帧，当前设置为{fps}帧")
        
        # 确保输出目录存在
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频的基本信息
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)  # 计算需要跳过的帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 打印视频信息
        # print(f"视频信息:")
        # print(f"- 原始帧率: {video_fps:.2f} fps")
        # print(f"- 目标帧率: {fps} fps")
        # print(f"- 总帧数: {total_frames}")
        # print(f"- 预计提取帧数: {total_frames // frame_interval}")
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按指定间隔保存帧
            if frame_count % frame_interval == 0:
                # 转换为PIL Image以便调整大小
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # 获取并调整尺寸
                width, height = frame_pil.size
                ratio = 1300 / width
                width = 1300
                height = int(height * ratio)
                reduced_image = frame_pil.resize((width, height))
                
                # 如果高度超过850，进一步调整
                if height > 850:
                    ratio = 850 / height
                    height = 850
                    width = int(width * ratio)
                    reduced_image = reduced_image.resize((width, height))
                
                # 保存调整后的帧
                frame_path = output_dir / f"{saved_count}.jpg"
                reduced_image.save(str(frame_path))
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        # content = f"已从视频中提取 {saved_count} 帧，保存至 {output_dir}"
        return str(output_dir),saved_count
    
    # 设置点击位置
    def Set_Clicked(self, clicked, method):
        self.clicked_x, self.clicked_y = clicked
        self.method = method

    # 显示点击点
    def Draw_Point(self, image, label):
        if label == 1:
            cv2.circle(image, (self.clicked_x, self.clicked_y), 5, (255, 0, 0), -1)  # 蓝色点
        elif label == 0:
            cv2.circle(image, (self.clicked_x, self.clicked_y), 5, (0, 0, 255), -1)  # 红色点

    def add_new_points_or_box(self, obj_id, points, labels, class_name):
        if not points or not labels:
            return

        ann_frame_idx = 0  # 当前帧的索引

        points_array = np.array(points, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int32)

        _, self.out_obj_ids, self.out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj_id,
            points=points_array,
            labels=labels_array,
        )
        self.target_points[obj_id] = points_array
        self.target_methods[obj_id] = labels_array
        self.target_labels[obj_id] = class_name
        self.option = True
        

    # def Draw_Mask(self, mask, frame,obj_id=None):
    #     # 转换 mask 为 NumPy 数组
    #     mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    #     h, w = mask.shape[-2:]
    #     mask = mask.reshape(h, w, 1)
    #     white = np.zeros([h, w, 1], dtype="uint8")
    #     white[:, :, 0] = 255
    #     x = mask * white
    #     x = np.uint8(x)

    #     # 使用 Canny 算法提取轮廓
    #     canny = cv2.Canny(x, 50, 100)
        
    #     # 绘制点击点
    #     self.Draw_Point(frame, self.method)

    #     img = frame.copy()
    #     contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     # 获取最大轮廓
    #     max_area = 0
    #     max_contour = None
    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         if area > max_area:
    #             max_area = area
    #             max_contour = contour

    #     if max_contour is not None:
    #         # 绘制最大轮廓的边界框
    #         self.x, self.y, self.w, self.h = cv2.boundingRect(max_contour)
    #         cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
    #         cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    #     self.image_mask = img
    #     cv2.imshow("img", img)
    #     return img
    
    def Draw_Mask(self, mask, frame, obj_id=None):
        # 转换 mask 为 NumPy 数组
        mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        h, w = mask.shape[-2:]
        mask = mask.reshape(h, w, 1)
        
        # 创建一个彩色的 mask
        color_mask = np.zeros_like(frame, dtype=np.uint8)
        color_mask[:, :, 0] = 0    # 蓝色分量
        color_mask[:, :, 1] = 255  # 绿色分量
        color_mask[:, :, 2] = 0    # 红色分量

        # 将 mask 应用到彩色 mask 上
        mask = (mask > 0).astype(np.uint8)  # 二值化
        color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask.squeeze())

        # 叠加到原始帧上 (半透明)
        alpha = 0.5  # 透明度
        frame_with_mask = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)

        # 显示轮廓
        contours, _ = cv2.findContours(mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_with_mask, contours, -1, (0, 255, 0), 2)  # 绿色轮廓

        # 初始化最大面积和对应的最大轮廓

        img = frame_with_mask.copy()

        max_area = 0
        max_contour = None

        # 遍历每个轮廓
        for contour in contours:
        # 计算轮廓的面积
            area = cv2.contourArea(contour)
            
            # 如果当前面积大于最大面积，则更新最大面积和对应的最大轮廓
            if area > max_area:
                max_area = area
                max_contour = contour
                
        if max_contour is None:
            return img, None

        # 使用矩形框绘制最大轮廓
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 在原图上绘制边缘线
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        self.image_mask = img
        self.x, self.y, self.w, self.h = x, y, w, h
        if obj_id is not None:
            self.object_boxes[obj_id] = (x, y, w, h)
        print(self.x, self.y, self.w, self.h)
        return img, (x, y, w, h)

    def Draw_Mask_Video(self, output_video_path="segmented_output.mp4"):
        # 收集所有帧的分割结果
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # 获取所有帧的名称
        frame_names = [
            p for p in os.listdir(self.video_path)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # 初始化视频写入器
        first_frame_path = os.path.join(self.video_path, frame_names[0])
        first_frame = cv2.imread(first_frame_path)
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

        # 遍历所有帧，处理分割结果并保存
        for out_frame_idx, frame_name in enumerate(frame_names):
            # 读取当前帧
            frame_path = os.path.join(self.video_path, frame_name)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            for out_obj_id, out_mask in self.video_segments[out_frame_idx].items():
                frame = self.Draw_Mask(out_mask, frame, out_obj_id)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    

                video_writer.write(frame)

        # 释放视频写入器
        video_writer.release()
        print(f"分割结果视频已保存到: {output_video_path}")


    def Draw_Mask_picture(self,frame_stride):
        # 1. 收集所有帧的分割结果
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # 2. 获取所有帧的名称
        frame_names = [
            p for p in os.listdir(self.video_path)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # 3. 每几帧显示一次结果
        vis_frame_stride = frame_stride
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            # 读取当前帧
            frame_path = os.path.join(self.video_path, frame_names[out_frame_idx])
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 如果该帧有分割结果，则绘制
            if out_frame_idx in self.video_segments:
                for out_obj_id, out_mask in self.video_segments[out_frame_idx].items():
                    frame = self.Draw_Mask(out_mask, frame, out_obj_id)
            
            # 显示结果
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换回BGR用于显示
            # cv2.imshow(f"Frame {out_frame_idx}", frame)
            # cv2.waitKey(1)  # 添加短暂延迟



    def Draw_Mask_at_frame(self, start_frame=0, return_frames=False, save_image_path=None ,save_path=None):
        """
        遍历所有帧并绘制轮廓
        Args:
            start_frame (int): 起始帧序号
            return_frames (bool): 是否返回处理后的帧列表
            save_path (str): 保存路径
        Returns:
            tuple: (processed_frames, xml_messages)
        """
        # 1. 收集所有帧的分割结果
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # 2. 获取所有帧的名称
        frame_names = [
            p for p in os.listdir(self.video_path)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        processed_frames = [] if return_frames else None

        xml_messages = []
        for frame_idx in range(start_frame, len(frame_names)):
            frame_name = frame_names[frame_idx]
            frame_stem = Path(frame_name).stem
            frame_path = os.path.join(self.video_path, frame_name)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Warning: Invalid frame at index {frame_idx}")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame = frame_rgb.copy()

            frame_results = []
            frame_file_path = None
            frame_size = None

            if frame_idx in self.video_segments:
                for out_obj_id, out_mask in self.video_segments[frame_idx].items():
                    annotated_frame, bbox = self.Draw_Mask(out_mask, annotated_frame, out_obj_id)
                    if not bbox:
                        continue

                    x, y, w, h = bbox
                    class_name = self.target_labels.get(out_obj_id, f"obj_{out_obj_id}")

                    if save_path:
                        result, file_path, size = xml_message(
                            save_path,
                            frame_stem,
                            int(annotated_frame.shape[1]),
                            int(annotated_frame.shape[0]),
                            class_name,
                            x,
                            y,
                            x + w,
                            y + h,
                        )
                    else:
                        size = [int(annotated_frame.shape[1]), int(annotated_frame.shape[0]), 3]
                        file_path = None
                        result = {
                            'name': class_name,
                            'pose': 'Unspecified',
                            'truncated': 0,
                            'difficult': 0,
                            'bndbox': [x, y, x + w, y + h],
                        }

                    frame_results.append(result)
                    frame_file_path = file_path
                    frame_size = size

            annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            if return_frames and processed_frames is not None:
                processed_frames.append(annotated_bgr.copy())

            if save_image_path:
                frame_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
                width, height = frame_pil.size
                ratio = 1300 / width
                width = 1300
                height *= ratio
                reduced_image = frame_pil.resize((int(width), int(height)))

                if height > 850:
                    ratio = 850 / height
                    height = 850
                    width *= ratio
                    reduced_image = reduced_image.resize((int(width), int(height)))

                save_path_frame = f"{save_image_path}/{frame_idx}.jpg"
                reduced_image.save(save_path_frame)

            if frame_results:
                xml_messages.append({
                    "frame_name": frame_stem,
                    "file_path": frame_file_path,
                    "size": frame_size,
                    "results": frame_results,
                })

        return processed_frames, xml_messages



if __name__ == '__main__':
    
    AD = AnythingVideo_TW()
    # 提取视频帧
    video_path = r"sampro\notebooks\videos\bedroom.mp4"
    output_dir = r"sampro/notebooks/videos/extracted_frames"
    video_dir = AD.extract_frames_from_video(video_path, output_dir, fps=2)
    frame = AD.set_video(video_dir)
    AD.inference(video_dir)
    AD.Set_Clicked([300, 483], 1)
    AD.add_new_points_or_box(obj_id=1, points=[[300, 483]], labels=[1], class_name="object")
    # AD.Draw_Mask_picture(frame_stride=1)
    # AD.Draw_Mask((AD.out_mask_logits[0] > 0.0).cpu().numpy(),frame) #暂时不用
    # AD.Draw_Mask_Video(output_video_path="segmented_output.mp4")
    AD.Draw_Mask_at_frame() # 在指定帧上绘制轮廓（例如第10帧）
    cv2.waitKey(0)