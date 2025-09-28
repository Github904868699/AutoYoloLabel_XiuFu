import sys, os
from pathlib import Path
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QCoreApplication, QRect, pyqtSignal,QTimer
from PyQt5.QtCore import Qt, QLineF,QUrl
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import cv2
import numpy as np
from util.QtFunc import *
from util.xmlfile import *

from GUI.UI_Main import Ui_MainWindow
from GUI.message import LabelInputDialog

sys.path.append("smapro")
from sampro.LabelQuick_TW import Anything_TW
from sampro.LabelVideo_TW import AnythingVideo_TW

from PyQt5.QtCore import QThread, pyqtSignal, QTimer

class VideoProcessingThread(QThread):
    finished = pyqtSignal()  # 完成信号
    frame_ready = pyqtSignal(object)  # 添加新信号用于传递处理后的帧

    def __init__(self, avt, video_path, output_dir,clicked_x, clicked_y, method,text,save_path):
        super().__init__()
        self.AVT = AnythingVideo_TW()
        self.video_path = video_path
        self.output_dir = output_dir
        self.clicked_x = clicked_x
        self.clicked_y = clicked_y
        self.method = method
        self.text = text
        self.save_path = save_path
        self.xml_messages = []
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        print(self.clicked_x, self.clicked_y, self.method)
        try:
            # 创建输出目录和mask子目录
            os.makedirs(self.output_dir, exist_ok=True)
            mask_dir = os.path.join(self.output_dir, "mask")
            os.makedirs(mask_dir, exist_ok=True)
            
            # 提取视频帧
            self.AVT.extract_frames_from_video(self.video_path, self.output_dir,fps=2)
            self.AVT.set_video(self.output_dir)
            self.AVT.inference(self.output_dir)
            self.AVT.Set_Clicked([self.clicked_x, self.clicked_y], self.method)
            self.AVT.add_new_points_or_box()
            
            # 获取处理后的帧并发送信号
            processed_frame, xml_messages = self.AVT.Draw_Mask_at_frame(save_image_path=mask_dir,save_path=self.save_path,text=self.text)  # 使用新的mask_dir路径
            self.xml_messages = xml_messages
            self.frame_ready.emit(processed_frame)  # 发送处理后的帧
            
        except Exception as e:
            print(f"处理出错: {str(e)}")
            import traceback
            traceback.print_exc()
        self.finished.emit()


class MainFunc(QMainWindow):
    my_signal = pyqtSignal()

    def __init__(self):
        super(MainFunc, self).__init__()
        # 连接应用程序的 aboutToQuit 信号到自定义的槽函数
        QCoreApplication.instance().aboutToQuit.connect(self.clean_up)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.sld_video_pressed=False


        self.image_files = None
        self.img_path = None
        self.save_path = None
        self.clicked_event = False
        self.paint_event = False
        self.labels = []
        self.clicked_save = []
        self.paint_save = []
        self.label_boxes_by_row = []
        self.list_labels = []
        self.flag = False
        self.save = True
        self.cap = None
        self.video_path = None

        self.AT = Anything_TW()
        self.AVT = AnythingVideo_TW()

        self.timer_camera = QTimer()

        self.annotation_format_actions = {
            "YOLO": self.ui.actionSaveTypeYOLO,
            "XML": self.ui.actionSaveTypeXML,
        }
        self.save_type_action_group = QtWidgets.QActionGroup(self)
        self.save_type_action_group.setExclusive(True)
        for fmt, action in self.annotation_format_actions.items():
            action.setCheckable(True)
            action.setData(fmt)
            self.save_type_action_group.addAction(action)
            action.triggered.connect(lambda checked, fmt=fmt: self.on_save_type_triggered(fmt, checked))

        self.annotation_format = None
        self.on_annotation_format_changed("YOLO")
        self.ui.currentImageLabel.setText("Path")

        self.ui.actionOpen_Dir.triggered.connect(self.get_dir)
        self.ui.actionNext_Image.triggered.connect(self.next_img)
        self.ui.actionPrev_Image.triggered.connect(self.prev_img)
        self.ui.actionChange_Save_Dir.triggered.connect(self.set_save_path)
        self.ui.actionCreate_RectBox.triggered.connect(self.mousePaint)
        self.ui.actionOpen_Video.triggered.connect(self.get_video)
        self.ui.actionVideo_marking.triggered.connect(self.video_marking)


        self.ui.pushButton.clicked.connect(self.Btn_Start)
        self.ui.pushButton_2.clicked.connect(self.Btn_Stop)
        self.ui.pushButton_3.clicked.connect(self.Btn_Save)
        self.ui.pushButton_4.clicked.connect(self.Btn_Replay)
        self.ui.pushButton_5.clicked.connect(self.Btn_Auto)
        self.ui.pushButton_start_marking.clicked.connect(self.Btn_Start_Marking)

        self.ui.horizontalSlider.sliderReleased.connect(self.releaseSlider)
        self.ui.horizontalSlider.sliderPressed.connect(self.pressSlider)
        self.ui.horizontalSlider.sliderMoved.connect(self.moveSlider)

        # 获取视频总帧数和当前帧位置
        self.total_frames = 0
        self.current_frame = 0

    def Change_Enable(self,method="",state=False):
        if method=="ShowVideo":
            self.ui.pushButton.setEnabled(state)
            self.ui.pushButton_2.setEnabled(state)
            self.ui.pushButton_3.setEnabled(state)
            self.ui.pushButton_4.setEnabled(state)  # 初始时禁用重播按钮
            self.ui.pushButton_5.setEnabled(state)
            self.ui.horizontalSlider.setEnabled(state)
        if method=="MakeTag":
            self.ui.actionPrev_Image.setEnabled(state)
            self.ui.actionNext_Image.setEnabled(state)
            self.ui.actionCreate_RectBox.setEnabled(state)

    def clear_label_list(self):
        self.ui.listWidget.clear()
        self.label_boxes_by_row = []

    @staticmethod
    def _normalized_box(x1, y1, x2, y2):
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def _remove_box_from_collections(self, box):
        for collection in (self.clicked_save, self.paint_save):
            for idx, existing in enumerate(collection):
                if existing == box:
                    collection.pop(idx)
                    return

    def _persist_labels_after_edit(self):
        if not self.save_path or not self.image_name:
            return

        base_path = Path(self.save_path) / self.image_name
        if not self.labels:
            xml_path = base_path.with_suffix(".xml")
            txt_path = base_path.with_suffix(".txt")
            if xml_path.exists():
                xml_path.unlink()
            if txt_path.exists():
                txt_path.unlink()
            return

        if self.img_width and self.img_height:
            size = [self.img_width, self.img_height, 3]
            self.save_annotation_files(self.image_path, self.image_name, size, self.labels)

    def remove_selected_labels(self):
        selected_indexes = self.ui.listWidget.selectedIndexes()
        if not selected_indexes or not self.labels:
            return False

        rows = sorted({index.row() for index in selected_indexes}, reverse=True)
        removed = False
        for row in rows:
            if row >= len(self.labels):
                continue

            removed = True
            self.ui.listWidget.takeItem(row)
            del self.labels[row]
            if self.list_labels and row < len(self.list_labels):
                del self.list_labels[row]

            if row < len(self.label_boxes_by_row):
                box = self.label_boxes_by_row.pop(row)
                self._remove_box_from_collections(box)

        if removed:
            if self.img_path:
                self.Show_Exists()
            self._persist_labels_after_edit()

        return removed

    def clear_all_annotations(self):
        self.clicked_event = False
        self.paint_event = False
        self.save = True
        self.clear_label_list()
        self.list_labels = []
        self.clicked_save = []
        self.paint_save = []
        self.labels = []
        if self.img_path:
            self.show_qt(self.img_path)
        self.ui.label_4.mousePressEvent = self.mouse_press_event
        self.ui.label_4.setCursor(Qt.ArrowCursor)

        base_path = Path(self.save_path) / self.image_name if self.save_path and self.image_name else None
        if base_path:
            xml_path = base_path.with_suffix(".xml")
            txt_path = base_path.with_suffix(".txt")
            if xml_path.exists():
                xml_path.unlink()
            if txt_path.exists():
                txt_path.unlink()
            
    def get_dir(self):
        self.clear_label_list()
        if self.cap:
            self.timer_camera.stop()
            self.clear_label_list()  # 清空listWidget
        self.directory = QtWidgets.QFileDialog.getExistingDirectory()
        if self.directory:
            self.image_files = list_images_in_directory(self.directory)
            self.current_index = 0
            self.show_path_image()
            self.Change_Enable(method="MakeTag",state=True)
            self.Change_Enable(method="ShowVideo",state=False)
            # 禁用开始检测打标按钮
            self.ui.pushButton_start_marking.setEnabled(False)
            # 鼠标点击触发
            self.ui.label_4.mousePressEvent = self.mouse_press_event
        else:
            self.ui.currentImageLabel.setText("")

    def show_path_image(self):
        if self.image_files:
            self.image_path = self.image_files[self.current_index]
            # print(self.image_path)
            self.img_path = self.image_path
            self.image_name = os.path.basename(self.image_path).split('.')[0]
            # print(self.image_name)

            self.img_path, self.img_width, self.img_height = Change_image_Size(self.img_path)
            self.image = cv2.imread(self.img_path)
            self.AT.Set_Image(self.image)
            self.show_qt(self.img_path)
            self.Exists_Labels_And_Boxs()
            self.ui.currentImageLabel.setText(f"{os.path.basename(self.image_path)}")
        else:
            self.ui.currentImageLabel.setText("")

    # 展示已保存所有标签
    def Exists_Labels_And_Boxs(self):
        self.list_labels = []
        self.labels = []
        self.clicked_save = []
        self.paint_save = []
        self.clear_label_list()

        if not self.save_path or not self.image_name:
            return

        base_path = Path(self.save_path) / self.image_name
        preferred_formats = [self.annotation_format]
        fallback = "YOLO" if self.annotation_format == "XML" else "XML"
        preferred_formats.append(fallback)

        for fmt in preferred_formats:
            if fmt == "YOLO":
                txt_path = base_path.with_suffix(".txt")
                labels, boxes, names = load_yolo_labels(txt_path, self.img_width or 0, self.img_height or 0)
                if not labels:
                    continue
                self.labels = labels
                normalized_boxes = [self._normalized_box(box[0], box[1], box[2], box[3]) for box in boxes]
                self.paint_save = normalized_boxes.copy()
                for box, name in zip(normalized_boxes, names):
                    self.ui.listWidget.addItem(name)
                    self.label_boxes_by_row.append(box)
                self.Show_Exists()
                return
            else:
                xml_path = base_path.with_suffix(".xml")
                if not xml_path.exists():
                    continue
                self.labels = get_labels(str(xml_path))
                self.list_labels, list_box = list_label(str(xml_path))
                normalized_boxes = [self._normalized_box(box[0], box[1], box[2], box[3]) for box in list_box]
                self.paint_save = normalized_boxes.copy()
                for label, box in zip(self.list_labels, normalized_boxes):
                    self.ui.listWidget.addItem(label)
                    self.label_boxes_by_row.append(box)
                self.Show_Exists()
                return

    def show_qt(self, img_path):
        if img_path != None:
            Qt_Gui = QtGui.QPixmap(img_path)
            self.ui.label_3.setFixedSize(self.img_width, self.img_height)
            self.ui.label_3.setPixmap(Qt_Gui)

    def next_img(self):
        if self.img_path and not self.clicked_event and not self.paint_event:
            if self.image_files and self.current_index < len(self.image_files) - 1:
                self.current_index += 1
                print(self.current_index)
                self.Other_Img()
            else:
                upWindowsh("这是最后一张")

    def prev_img(self):
        if self.img_path and not self.clicked_event and not self.paint_event:
            if self.image_files and self.current_index > 0:
                self.current_index -= 1
                self.Other_Img()
                
            else:
                upWindowsh("这是第一张")
                
    def Other_Img(self):
        self.labels = []
        self.paint_save = []
        self.clicked_save = []
        self.clear_label_list()
        self.show_path_image()

    def set_save_path(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory()
        if directory:
            self.save_path = directory
            if self.img_path:
                self.Exists_Labels_And_Boxs()

# ########################################################################################################################
    # seg
    def mouse_press_event(self, event):
        try:
            if self.img_path:
                self.clicked_event = True
                x = event.x()
                y = event.y()

                if event.button() == Qt.LeftButton:
                    self.clicked_x, self.clicked_y, self.method = x, y, 1
                if event.button() == Qt.RightButton:
                    self.clicked_x, self.clicked_y, self.method = x, y, 0

                image = self.image.copy()
                self.AT.Set_Clicked([x, y], self.method)
                self.AT.Create_Mask()
                image = self.AT.Draw_Mask(self.AT.mask, image)

                h,w,channels=image.shape
                bytes_per_line = channels * w
                q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

                Qt_Gui = QtGui.QPixmap(q_image)
                self.ui.label_3.setFixedSize(self.img_width, self.img_height)
                self.ui.label_3.setPixmap(Qt_Gui)

                self.save = False
        except Exception as e:
            print(f"Error in mouse_press_event: {str(e)}")

# ########################################################################################################################
# 重写QWidget类的keyPressEvent方法
    def keyPressEvent(self, event):
        if self.img_path:
            if self.clicked_event and not self.paint_event:
                image = self.AT.Key_Event(event.key())

            if self.video_path:
                if self.clicked_event or self.paint_event:
                    if (event.key() == 83):
                        self.save = True
                        self.dialog = LabelInputDialog(self)
                        self.dialog.show()
                        self.dialog.confirmed.connect(self.video_on_dialog_confirmed)
                        
                        # 禁用label4的鼠标事件
                        self.ui.label_4.mousePressEvent = None

                    if (event.key() == 81):
                        self.clicked_event = False
                        self.paint_event = False
                        self.save = True
                        self.Show_Exists()
                        self.ui.label_4.mousePressEvent = self.mouse_press_event
                        self.ui.label_4.setCursor(Qt.ArrowCursor)
            else:
                if self.clicked_event or self.paint_event:
                    if (event.key() == 83):
                        self.save = True
                        self.dialog = LabelInputDialog(self)
                        self.dialog.show()
                        self.dialog.confirmed.connect(self.on_dialog_confirmed)

                    if (event.key() == 81):
                        self.clicked_event = False
                        self.paint_event = False
                        self.save = True
                        self.Show_Exists()
                        self.ui.label_4.mousePressEvent = self.mouse_press_event
                        self.ui.label_4.setCursor(Qt.ArrowCursor)

            if event.key() == Qt.Key_Delete:
                if self.remove_selected_labels():
                    return
                if not self.ui.listWidget.selectedIndexes():
                    self.clear_all_annotations()
                return


    
    def on_dialog_confirmed(self, text):
        if not self.save_path:
            upWindowsh("请选择保存路径")

        elif text and self.clicked_event:
            self.ui.listWidget.addItem(text)
            result, file_path, size = xml_message(self.save_path, self.image_name, self.img_width, self.img_height,
                                                  text, self.AT.x, self.AT.y, self.AT.w, self.AT.h)
            self.labels.append(result)
            box = self._normalized_box(self.AT.x, self.AT.y, self.AT.w + self.AT.x, self.AT.h + self.AT.y)
            self.clicked_save.append(box)
            self.label_boxes_by_row.append(box)
            self.save_annotation_files(self.image_path, self.image_name, size, self.labels)

        elif text and self.paint_event:
            self.paint_event = False
            self.clicked_event = True
            self.ui.listWidget.addItem(text)
            result, file_path, size = xml_message(self.save_path, self.image_name, self.img_width, self.img_height,
                                                  text, self.x0, self.y0, abs(self.x1 - self.x0),
                                                  abs(self.y1 - self.y0))
            self.labels.append(result)
            box = self._normalized_box(self.x0, self.y0, self.x1, self.y1)
            self.paint_save.append(box)
            self.label_boxes_by_row.append(box)
            self.save_annotation_files(self.image_path, self.image_name, size, self.labels)

            self.ui.label_4.mousePressEvent = self.mouse_press_event
            self.ui.label_4.setCursor(Qt.ArrowCursor)
            
        self.clicked_event = False
        self.paint_event = False

        self.Show_Exists()

    def video_on_dialog_confirmed(self, text):
        self.text = text
        print(self.text)
        if not self.save_path:
            upWindowsh("请选择保存路径")
        elif text and self.clicked_event:
            self.ui.listWidget.addItem(text)
            result, file_path, size = xml_message(self.save_path, self.image_name, self.img_width, self.img_height,
                                                    text, self.AT.x, self.AT.y, self.AT.w, self.AT.h)
            self.labels.append(result)
            box = self._normalized_box(self.AT.x, self.AT.y, self.AT.w + self.AT.x, self.AT.h + self.AT.y)
            self.clicked_save.append(box)
            self.label_boxes_by_row.append(box)
            self.save_annotation_files(self.image_path, self.image_name, size, self.labels)
            # 启用"开始检测打标"按钮
            self.ui.pushButton_start_marking.setEnabled(True)
        self.clicked_event = False
        self.paint_event = False

        self.Show_Exists()
        

    # 显示已存在框
    def Show_Exists(self):
        image = cv2.imread(self.img_path)
        if self.clicked_save == [] and self.paint_save == []:
            self.show_qt(self.img_path)
        else:
            if self.clicked_save != []:
                for i in self.clicked_save:
                    image = cv2.rectangle(image, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 2)
            if self.paint_save != []:
                for i in self.paint_save:
                    image = cv2.rectangle(image, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)

            h,w,channels=image.shape
            bytes_per_line = channels * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            Qt_Gui = QtGui.QPixmap(q_image)
            self.ui.label_3.setFixedSize(self.img_width, self.img_height)
            self.ui.label_3.setPixmap(Qt_Gui)

# ##################################################################################################
    # 手动打标
    def mousePaint(self):
        if self.img_path != None:
            self.paint_event = True
            self.clicked_event = False
            if self.save:
                self.ui.label_4.mousePressEvent = self.mousePressEvent
                self.ui.label_4.mouseMoveEvent = self.mouseMoveEvent
                self.ui.label_4.mouseReleaseEvent = self.mouseReleaseEvent
                self.ui.label_4.paintEvent = self.paintEvent
                self.ui.label_4.setCursor(Qt.CrossCursor)
                self.save = False
            else:
                upWindowsh("请先输入标签")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.flag = True
            self.show_qt(self.img_path)
            self.x0, self.y0 = event.pos().x(), event.pos().y()
            self.x1, self.y1 = self.x0, self.y0
            self.ui.label_4.update()

    def mouseReleaseEvent(self, event):
        if self.flag:
            self.saveAndUpdate()
        self.flag = False
        self.save = False

    def mouseMoveEvent(self, event):
        if self.flag:
            self.x1, self.y1 = event.pos().x(), event.pos().y()
            self.ui.label_4.update()

    def paintEvent(self, event):
        super(MainFunc, self).paintEvent(event)
        if self.flag and self.x0 != 0 and self.y0 != 0 and self.x1 != 0 and self.y1 != 0:
            painter = QPainter(self.ui.label_4)
            painter.setPen(QPen(Qt.red, 4, Qt.SolidLine))
            painter.drawRect(QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0)))

    def saveAndUpdate(self):
        try:

            # 获取当前label上的QPixmap对象
            if self.ui.label_3.pixmap():
                pixmap = self.ui.label_3.pixmap()
                image = QImage(pixmap.size(), QImage.Format_ARGB32)
                painter = QPainter(image)

                # 绘制原始图像
                painter.drawPixmap(0, 0, pixmap)

                # 绘制矩形框
                if self.x0 != 0 and self.y0 != 0 and self.x1 != 0 and self.y1 != 0:
                    painter.setPen(QPen(Qt.red, 4, Qt.SolidLine))
                    painter.drawRect(QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0)))

                painter.end()
        except Exception as e:
            print(f"Error saving and updating image: {e}")

# ##################################################################################################
    # 获取视频
    def get_video(self):
        self.clear_label_list()  # 清空listWidget
        self.image_files = None
        self.img_path = None
        self.num = 0
        video_save_path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片保存文件夹")
        if video_save_path:
            self.video_save_path = video_save_path
        
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            "选择视频", 
            "", 
            "Video Files (*.mp4 *.mpg)"
        )
        
        if video_path and video_save_path:
            self.video_path = video_path  # 保存视频路径以供重播使用
            self.cap = cv2.VideoCapture(video_path)
            # 获取视频总帧数
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # 设置滑块范围
            self.ui.horizontalSlider.setRange(0, self.total_frames)
            self.timer_camera.start(33)
            self.timer_camera.timeout.connect(self.OpenFrame)
            # 初始禁用重播按钮
            self.ui.pushButton_4.setEnabled(False)
        
            self.Change_Enable(method="ShowVideo", state=True)
            self.Change_Enable(method="MakeTag", state=False)
            # 禁用开始检测打标按钮
            self.ui.pushButton_start_marking.setEnabled(False)
        else:
            upWindowsh("请先选择视频和保存路径")


    def OpenFrame(self):
        if not self.sld_video_pressed:  # 只在未拖动时更新
            ret, image = self.cap.read()
            if ret:
                # 更新当前帧位置
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                # 更新滑块位置
                self.ui.horizontalSlider.setValue(self.current_frame)
                
                # 调整视频帧大小
                height, width = image.shape[:2]
                ratio = 1300 / width
                new_width = 1300
                new_height = int(height * ratio)
                
                if new_height > 850:
                    ratio = 850 / new_height
                    new_height = 850
                    new_width = int(new_width * ratio)
                
                # 调整图像大小
                image = cv2.resize(image, (new_width, new_height))
                
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    vedio_img = QImage(image.data, new_width, new_height, image.strides[0], QImage.Format_RGB888)
                elif len(image.shape) == 1:
                    vedio_img = QImage(image.data, new_width, new_height, QImage.Format_Indexed8)
                else:
                    vedio_img = QImage(image.data, new_width, new_height, image.strides[0], QImage.Format_RGB888)
                self.vedio_img = vedio_img
                
                # 调整label大小以适应新的图像尺寸
                self.ui.label_3.setFixedSize(new_width, new_height)
                self.ui.label_3.setPixmap(QPixmap(self.vedio_img))
                self.ui.label_3.setScaledContents(True)
            else:
                self.cap.release()
                self.timer_camera.stop()
                # 视频结束时启用重播按钮
                self.ui.pushButton_4.setEnabled(True)


    def Btn_Start(self):
        try:
            # 尝试断开之前的连接
            self.timer_camera.timeout.disconnect(self.OpenFrame)
        except TypeError:
            # 如果没有连接，直接忽略错误
            pass
        # 重新连接并启动定时器
        self.timer_camera.timeout.connect(self.OpenFrame)
        self.timer_camera.start(33)

    def Btn_Stop(self):
        self.timer_camera.stop()
        try:
            # 尝试断开定时器连接
            self.timer_camera.timeout.disconnect(self.OpenFrame)
        except TypeError:
            pass

    def Btn_Save(self):
        self.num += 1
        save_path = f'{self.video_save_path}/image{str(self.num)}.jpg'
        self.vedio_img.save(save_path)
        # 将保存信息添加到listWidget
        save_info = f'image{str(self.num)}.jpg保存成功！'
        self.ui.listWidget.addItem(save_info)
        print(f'{save_path}保存成功！')

    
    def moveSlider(self, position):
        """处理滑块移动"""
        if self.cap and self.total_frames > 0:
            # 设置视频帧位置
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            # 读取并显示新位置的帧
            ret, image = self.cap.read()
            if ret:
                # 调整视频帧大小
                height, width = image.shape[:2]
                ratio = 1300 / width
                new_width = 1300
                new_height = int(height * ratio)
                
                if new_height > 850:
                    ratio = 850 / new_height
                    new_height = 850
                    new_width = int(new_width * ratio)
                
                # 调整图像大小
                image = cv2.resize(image, (new_width, new_height))
                
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    vedio_img = QImage(image.data, new_width, new_height, image.strides[0], QImage.Format_RGB888)
                elif len(image.shape) == 1:
                    vedio_img = QImage(image.data, new_width, new_height, QImage.Format_Indexed8)
                else:
                    vedio_img = QImage(image.data, new_width, new_height, image.strides[0], QImage.Format_RGB888)
                self.vedio_img = vedio_img
                
                # 调整label大小以适应新的图像尺寸
                self.ui.label_3.setFixedSize(new_width, new_height)
                self.ui.label_3.setPixmap(QPixmap(self.vedio_img))
                self.ui.label_3.setScaledContents(True)

    def pressSlider(self):
        self.sld_video_pressed = True
        self.timer_camera.stop()  # 暂停视频播放

    def releaseSlider(self):
        self.sld_video_pressed = False
        try:
            # 尝试断开之前的连接
            self.timer_camera.timeout.disconnect(self.OpenFrame)
        except TypeError:
            pass
        # 重新连接并启动定时器
        self.timer_camera.timeout.connect(self.OpenFrame)
        self.timer_camera.start(33)

    def clean_up(self):
        file_path = 'GUI/history.txt'
        if os.path.exists(file_path):
            os.remove(file_path)

    def Btn_Replay(self):
        """重新播放视频"""
        if hasattr(self, 'video_path'):
            # 重新打开视频文件
            self.cap = cv2.VideoCapture(self.video_path)
            # 重置滑块位置
            self.ui.horizontalSlider.setValue(0)
            # 开始播放
            self.timer_camera.start(33)                 
            # 禁用重播按钮
            self.ui.pushButton_4.setEnabled(False)

    def Btn_Auto(self):
        if self.video_path and self.video_save_path:
            output_dir,saved_count = self.AVT.extract_frames_from_video(self.video_path, self.video_save_path, fps=2)
            content = f"已从视频中提取 {saved_count} 帧\n保存至 {output_dir}"
            print(content)
            self.ui.listWidget.addItem(content)
        else:
            upWindowsh("请先选择视频和保存路径")

    def video_marking(self):
        self.directory = None
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.mpg)")
        self.video_path = video_path

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片保存文件夹")
        self.output_dir = output_dir
        self.clear_label_list()
        if self.video_path and self.output_dir:
            self.Change_Enable(method="MakeTag",state=False)
            self.Change_Enable(method="ShowVideo",state=False)
            if self.cap:
                self.cap.release()
                self.timer_camera.stop()

            # 读取视频第一帧
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(f"{self.output_dir}/0.jpg", frame)
            cap.release()
            if self.output_dir:
                self.image_files = list_images_in_directory(self.output_dir)
                if self.image_files:
                    self.image_path = self.image_files[0]
                    # print(self.image_path)
                    self.img_path = self.image_path
                    self.image_name = os.path.basename(self.image_path).split('.')[0]
                    # print(self.image_name)

                    self.img_path, self.img_width, self.img_height = Change_image_Size(self.img_path)
                    print(self.img_path, self.img_width, self.img_height)
                    self.image = cv2.imread(self.img_path)

                    self.AT.Set_Image(self.image)
                    # 转换为QPixmap并显示
                    Qt_Gui = QtGui.QPixmap(self.img_path)
                    # 设置label大小为图片原始大小
                    self.ui.label_3.setFixedSize(self.img_width, self.img_height)
                    self.ui.label_3.setPixmap(Qt_Gui)
                    self.ui.currentImageLabel.setText(f"当前图片：{os.path.basename(self.image_path)}")

            # 鼠标点击触发
            self.ui.label_4.mousePressEvent = self.mouse_press_event
        else:
            upWindowsh("请先选择视频和保存路径")


    def on_video_processing_complete(self):
        self.worker_thread.deleteLater()
        self.xml_messages = self.worker_thread.xml_messages
        # print(self.xml_messages)
        
        # 遍历输出目录中的图片
        for img_file in os.listdir(self.output_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):  # 检查图片文件扩展名
                # 获取不带扩展名的文件名
                img_name = os.path.splitext(img_file)[0]
                img_file  = os.path.join(self.output_dir,img_file)

                # 在xml_messages中查找对应的消息
                for msg in self.xml_messages:
                    self.labels = []
                    if len(msg) > 1:  # 确保msg有足够的元素
                        xml_path = msg[1]  # 获取索引值为1的路径
                        xml_filename = os.path.splitext(os.path.basename(xml_path))[0]

                        # 如果文件名匹配，则复制XML文件到save_path
                        if xml_filename == img_name and self.save_path:
                            result = msg[0]
                            file_path = msg[1]
                            size = msg[2]
                            self.labels.append(result)
                            self.save_annotation_files(img_file, img_name, size, self.labels)
        self.ui.listWidget.addItem("检测打标完成！")
        print("检测打标完成！")

    def on_save_type_triggered(self, fmt, checked):
        if checked:
            self.on_annotation_format_changed(fmt)

    def on_annotation_format_changed(self, text):
        fmt = (text or "").strip().upper() or "XML"
        if fmt == self.annotation_format:
            return

        self.annotation_format = fmt
        for action_fmt, action in self.annotation_format_actions.items():
            block = action.blockSignals(True)
            action.setChecked(action_fmt == fmt)
            action.blockSignals(block)

        self.Exists_Labels_And_Boxs()

    def save_annotation_files(self, image_path, image_name, size, labels):
        if not self.save_path:
            return

        base_path = Path(self.save_path) / str(image_name)

        if self.annotation_format == "YOLO":
            write_yolo_labels(base_path, size, labels)
            xml_path = base_path.with_suffix(".xml")
            if xml_path.exists():
                xml_path.unlink()
        else:
            xml_path = base_path.with_suffix(".xml")
            xml(image_path, str(xml_path), size, labels)
            txt_path = base_path.with_suffix(".txt")
            if txt_path.exists():
                txt_path.unlink()
                            

    def Btn_Start_Marking(self):
        # 禁用开始检测打标按钮
        self.ui.pushButton_start_marking.setEnabled(False)
        if self.video_path and self.output_dir:
            # 创建并启动工作线程
            self.worker_thread = VideoProcessingThread(self.AVT, self.video_path, self.output_dir,self.clicked_x, self.clicked_y, self.method,self.text,self.save_path)
            self.worker_thread.finished.connect(self.on_video_processing_complete)
            self.worker_thread.start()
            

        else:
            upWindowsh("请先选择视频和保存路径")

        

