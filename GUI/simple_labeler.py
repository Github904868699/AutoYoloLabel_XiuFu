"""Simplified annotation interface built on top of the original tooling.

This window keeps the fast click-based segmentation workflow from the
existing application but reduces the amount of chrome on screen.  Users can
open an image directory, click to add positive/negative points, and commit the
mask to either XML or YOLO annotations with a single text field.
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from sampro.LabelQuick_TW import Anything_TW
from util.QtFunc import list_images_in_directory, upWindowsh
from util.xmlfile import (
    get_labels,
    load_yolo_labels,
    write_yolo_labels,
    xml,
    xml_message,
)


MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 820


class ImageCanvas(QtWidgets.QLabel):
    """Clickable canvas used to capture SAM prompts."""

    clicked = QtCore.pyqtSignal(int, int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background-color: #202020; border: 1px solid #3c3c3c;")

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self.pixmap() is None:
            return
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(event.x(), event.y(), 1)
        elif event.button() == QtCore.Qt.RightButton:
            self.clicked.emit(event.x(), event.y(), 0)
        super().mousePressEvent(event)


class SimpleLabelerWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("简洁快速标注")
        self.resize(1280, 820)

        self.annotation_format = "XML"
        self.image_files: List[str] = []
        self.current_index: int = -1
        self.current_image_path: Optional[str] = None
        self.save_path: Optional[Path] = None

        self.original_image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self.display_scale: float = 1.0
        self.original_size: Tuple[int, int, int] = (0, 0, 3)

        self.current_labels: List[dict] = []
        self.display_rects: List[Tuple[int, int, int, int]] = []
        self.pending_mask: Optional[np.ndarray] = None
        self.pending_bbox_display: Optional[Tuple[int, int, int, int]] = None
        self.pending_bbox_original: Optional[Tuple[int, int, int, int]] = None

        self._current_qimage_buffer: Optional[np.ndarray] = None

        self.segmentor = Anything_TW()

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        main_layout.addWidget(splitter)

        # Image list -----------------------------------------------------
        list_container = QtWidgets.QWidget()
        list_layout = QtWidgets.QVBoxLayout(list_container)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(8)

        controls_row = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("打开图片夹")
        self.save_dir_button = QtWidgets.QPushButton("标注保存位置")
        controls_row.addWidget(self.open_button)
        controls_row.addWidget(self.save_dir_button)
        list_layout.addLayout(controls_row)

        format_row = QtWidgets.QHBoxLayout()
        format_label = QtWidgets.QLabel("标注格式：")
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["XML", "YOLO"])
        format_row.addWidget(format_label)
        format_row.addWidget(self.format_combo)
        format_row.addStretch(1)
        list_layout.addLayout(format_row)

        self.image_list = QtWidgets.QListWidget()
        self.image_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        list_layout.addWidget(self.image_list, 1)

        navigation_row = QtWidgets.QHBoxLayout()
        self.prev_button = QtWidgets.QPushButton("上一张")
        self.next_button = QtWidgets.QPushButton("下一张")
        navigation_row.addWidget(self.prev_button)
        navigation_row.addWidget(self.next_button)
        list_layout.addLayout(navigation_row)

        splitter.addWidget(list_container)

        # Right panel ----------------------------------------------------
        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.current_image_label = QtWidgets.QLabel("")
        self.current_image_label.setStyleSheet("font-weight: 600;")
        right_layout.addWidget(self.current_image_label)

        self.canvas = ImageCanvas()
        self.canvas.setMinimumSize(640, 480)
        right_layout.addWidget(self.canvas, 1)

        self.hint_label = QtWidgets.QLabel(
            "左键添加前景点，右键添加背景点。满意后输入标签名称点击保存。"
        )
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #666666;")
        right_layout.addWidget(self.hint_label)

        form_layout = QtWidgets.QHBoxLayout()
        self.label_edit = QtWidgets.QLineEdit()
        self.label_edit.setPlaceholderText("标签名称…")
        self.save_button = QtWidgets.QPushButton("保存标注")
        self.clear_button = QtWidgets.QPushButton("撤销本次")
        form_layout.addWidget(self.label_edit, 1)
        form_layout.addWidget(self.save_button)
        form_layout.addWidget(self.clear_button)
        right_layout.addLayout(form_layout)

        self.annotation_list = QtWidgets.QListWidget()
        self.annotation_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        right_layout.addWidget(self.annotation_list, 1)

        splitter.addWidget(right_container)
        splitter.setStretchFactor(1, 1)

    def _connect_signals(self) -> None:
        self.open_button.clicked.connect(self._open_directory)
        self.save_dir_button.clicked.connect(self._select_save_directory)
        self.image_list.itemSelectionChanged.connect(self._on_image_selected)
        self.prev_button.clicked.connect(self._go_previous)
        self.next_button.clicked.connect(self._go_next)
        self.canvas.clicked.connect(self._on_canvas_clicked)
        self.save_button.clicked.connect(self._save_annotation)
        self.clear_button.clicked.connect(self._clear_pending_annotation)
        self.format_combo.currentTextChanged.connect(self._on_format_changed)

    # -------------------------------------------------------------- helpers ---
    def _open_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not directory:
            return
        self.image_files = sorted(list_images_in_directory(directory))
        if not self.image_files:
            upWindowsh("该文件夹下未找到图片")
            return
        self.current_index = 0
        self.image_list.clear()
        for path in self.image_files:
            self.image_list.addItem(os.path.relpath(path, directory))
        self.image_list.setCurrentRow(0)
        if self.save_path is None:
            self.save_path = Path(directory)

    def _select_save_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择保存路径")
        if directory:
            self.save_path = Path(directory)

    def _go_previous(self) -> None:
        if not self.image_files:
            return
        if self.current_index <= 0:
            upWindowsh("已经是第一张")
            return
        self.current_index -= 1
        self.image_list.setCurrentRow(self.current_index)

    def _go_next(self) -> None:
        if not self.image_files:
            return
        if self.current_index >= len(self.image_files) - 1:
            upWindowsh("已经是最后一张")
            return
        self.current_index += 1
        self.image_list.setCurrentRow(self.current_index)

    def _on_format_changed(self, text: str) -> None:
        self.annotation_format = text.strip().upper() or "XML"
        self._load_existing_annotations()

    # ------------------------------------------------------------- loading ---
    def _on_image_selected(self) -> None:
        row = self.image_list.currentRow()
        if row < 0 or row >= len(self.image_files):
            return
        self.current_index = row
        path = self.image_files[row]
        self._load_image(path)

    def _load_image(self, path: str) -> None:
        image = cv2.imread(path)
        if image is None:
            upWindowsh("无法读取图片：" + path)
            return

        self.current_image_path = path
        self.original_image = image
        h, w = image.shape[:2]
        self.original_size = (w, h, image.shape[2] if image.ndim == 3 else 1)

        scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1.0)
        if scale != 1.0:
            display = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            display = image.copy()
        self.display_scale = scale
        self.display_image = display
        self.segmentor.Set_Image(display.copy())

        self.current_labels = []
        self.display_rects = []
        self.pending_mask = None
        self.pending_bbox_display = None
        self.pending_bbox_original = None
        self.annotation_list.clear()
        self.label_edit.clear()

        self.current_image_label.setText(f"{os.path.basename(path)}")
        self._show_on_canvas(display)
        self._load_existing_annotations()

    def _show_on_canvas(self, image: np.ndarray) -> None:
        if image is None:
            self.canvas.clear()
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self._current_qimage_buffer = rgb.copy()
        q_image = QtGui.QImage(
            self._current_qimage_buffer.data,
            w,
            h,
            self._current_qimage_buffer.strides[0],
            QtGui.QImage.Format_RGB888,
        )
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.canvas.setFixedSize(w, h)
        self.canvas.setPixmap(pixmap)

    def _render_with_overlays(self) -> None:
        if self.display_image is None:
            return
        canvas = self.display_image.copy()
        for x1, y1, x2, y2 in self.display_rects:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self._show_on_canvas(canvas)

    # ---------------------------------------------------------- annotations ---
    def _on_canvas_clicked(self, x: int, y: int, method: int) -> None:
        if self.display_image is None:
            return
        self.segmentor.Set_Clicked([x, y], method)
        self.segmentor.Create_Mask()
        mask_image = self.segmentor.Draw_Mask(self.segmentor.mask, self.display_image.copy())
        self.pending_mask = mask_image
        bbox = (self.segmentor.x, self.segmentor.y, self.segmentor.w, self.segmentor.h)
        self.pending_bbox_display = bbox
        if self.display_scale:
            scale = self.display_scale
        else:
            scale = 1.0
        x_orig = int(round(self.segmentor.x / scale))
        y_orig = int(round(self.segmentor.y / scale))
        w_orig = int(round(self.segmentor.w / scale))
        h_orig = int(round(self.segmentor.h / scale))
        self.pending_bbox_original = (x_orig, y_orig, w_orig, h_orig)
        self._show_on_canvas(mask_image)

    def _clear_pending_annotation(self) -> None:
        if self.pending_bbox_display is None:
            return
        self.segmentor.Key_Event(16777219)  # Backspace
        self.pending_mask = None
        self.pending_bbox_display = None
        self.pending_bbox_original = None
        self.label_edit.clear()
        self._render_with_overlays()

    def _save_annotation(self) -> None:
        if not self.pending_bbox_original or not self.pending_bbox_display:
            upWindowsh("请先点击图片生成标注")
            return
        label_text = self.label_edit.text().strip()
        if not label_text:
            upWindowsh("请输入标签名称")
            return
        if self.save_path is None or self.current_image_path is None:
            upWindowsh("请先设置保存路径")
            return

        x_disp, y_disp, w_disp, h_disp = self.pending_bbox_display
        rect = (x_disp, y_disp, x_disp + w_disp, y_disp + h_disp)
        self.display_rects.append(rect)
        self.annotation_list.addItem(label_text)

        x_orig, y_orig, w_orig, h_orig = self.pending_bbox_original
        result, file_path, size = xml_message(
            str(self.save_path),
            Path(self.current_image_path).stem,
            self.original_size[0],
            self.original_size[1],
            label_text,
            x_orig,
            y_orig,
            w_orig,
            h_orig,
        )
        self.current_labels.append(result)
        self._persist_annotations(Path(self.current_image_path), Path(file_path).stem, size, self.current_labels)

        self.segmentor.Key_Event(83)
        self.pending_mask = None
        self.pending_bbox_display = None
        self.pending_bbox_original = None
        self.label_edit.clear()
        self._render_with_overlays()

    def _persist_annotations(
        self,
        image_path: Path,
        image_name: Path,
        size: Tuple[int, int, int],
        labels: List[dict],
    ) -> None:
        if self.save_path is None:
            return
        base_path = self.save_path / image_name
        if self.annotation_format == "YOLO":
            write_yolo_labels(base_path, size, labels)
            xml_path = base_path.with_suffix(".xml")
            if xml_path.exists():
                xml_path.unlink()
        else:
            xml_path = base_path.with_suffix(".xml")
            xml(str(image_path), str(xml_path), size, labels)
            txt_path = base_path.with_suffix(".txt")
            if txt_path.exists():
                txt_path.unlink()

    def _load_existing_annotations(self) -> None:
        if self.save_path is None or self.current_image_path is None:
            self._render_with_overlays()
            return
        image_name = Path(self.current_image_path).stem
        base_path = self.save_path / image_name
        preferred = [self.annotation_format]
        fallback = "YOLO" if self.annotation_format == "XML" else "XML"
        preferred.append(fallback)

        labels: List[dict] = []
        rects: List[Tuple[int, int, int, int]] = []

        for fmt in preferred:
            if fmt == "YOLO":
                loaded, boxes, names = load_yolo_labels(
                    base_path.with_suffix(".txt"),
                    self.original_size[0],
                    self.original_size[1],
                )
                if not loaded:
                    continue
                labels = loaded
                rects = [
                    (
                        int(round(x1 * self.display_scale)),
                        int(round(y1 * self.display_scale)),
                        int(round(x2 * self.display_scale)),
                        int(round(y2 * self.display_scale)),
                    )
                    for x1, y1, x2, y2 in boxes
                ]
                self.annotation_list.clear()
                for name in names:
                    self.annotation_list.addItem(name)
                break
            else:
                xml_path = base_path.with_suffix(".xml")
                if not xml_path.exists():
                    continue
                labels = get_labels(str(xml_path))
                self.annotation_list.clear()
                for label in labels:
                    self.annotation_list.addItem(label["name"])
                rects = []
                for raw in labels:
                    xmin, ymin, xmax, ymax = raw["bndbox"]
                    width = xmax - xmin if xmax > xmin else xmax
                    height = ymax - ymin if ymax > ymin else ymax
                    x1_disp = int(round(xmin * self.display_scale))
                    y1_disp = int(round(ymin * self.display_scale))
                    x2_disp = int(round((xmin + width) * self.display_scale))
                    y2_disp = int(round((ymin + height) * self.display_scale))
                    rects.append((x1_disp, y1_disp, x2_disp, y2_disp))
                    raw["bndbox"] = [xmin, ymin, width, height]
                break

        self.current_labels = labels
        self.display_rects = rects
        self.pending_mask = None
        self.pending_bbox_display = None
        self.pending_bbox_original = None
        self.label_edit.clear()
        self._render_with_overlays()


def main() -> None:
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = SimpleLabelerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
