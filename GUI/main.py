"""Main application window for the streamlined AutoYoloLabel tool.

This module rewrites the original UI logic around a hand-crafted layout so
that the code is easier to follow and extend.  The window is split into two
panels: the left side contains directory/format controls and the image list
while the right side hosts the interactive canvas together with annotation
management widgets.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from sampro.LabelQuick_TW import Anything_TW
from util.QtFunc import get_labels, list_images_in_directory, upWindowsh
from util.xmlfile import load_yolo_labels, write_yolo_labels, xml, xml_message


MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 820


class ImageCanvas(QtWidgets.QLabel):
    """Clickable QLabel used as a drawing surface for segmentation prompts."""

    clicked = QtCore.pyqtSignal(int, int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background-color: #1f1f1f; border: 1px solid #3c3c3c;")

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self.pixmap() is None:
            return
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(event.x(), event.y(), 1)
        elif event.button() == QtCore.Qt.RightButton:
            self.clicked.emit(event.x(), event.y(), 0)
        super().mousePressEvent(event)


class LabelerMainWindow(QtWidgets.QMainWindow):
    """Main window that wires the segmentation workflow together."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("AutoYoloLabel")
        self.resize(MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT)

        # Data/state --------------------------------------------------------
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
        self.pending_mask: Optional[np.ndarray] = None
        self.pending_bbox_display: Optional[Tuple[int, int, int, int]] = None
        self.pending_bbox_original: Optional[Tuple[int, int, int, int]] = None

        self._current_qimage_buffer: Optional[np.ndarray] = None

        self.segmentor = Anything_TW()

        # UI ----------------------------------------------------------------
        self._build_ui()
        self._connect_signals()
        self._setup_shortcuts()

        self.statusBar().showMessage("准备就绪")

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        control_row = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("打开图片夹")
        self.save_dir_button = QtWidgets.QPushButton("标注保存位置")
        control_row.addWidget(self.open_button)
        control_row.addWidget(self.save_dir_button)

        control_row.addSpacing(12)
        control_row.addWidget(QtWidgets.QLabel("标注格式："))
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["XML", "YOLO"])
        control_row.addWidget(self.format_combo)
        control_row.addStretch(1)

        self.current_image_label = QtWidgets.QLabel("未加载图片")
        self.current_image_label.setStyleSheet("font-weight: 600;")
        control_row.addWidget(self.current_image_label)
        main_layout.addLayout(control_row)

        self.hint_label = QtWidgets.QLabel(
            "提示：A 上一张，D 下一张，Q 撤销当前掩膜。左键前景，右键背景。"
        )
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #666666;")
        main_layout.addWidget(self.hint_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        main_layout.addWidget(splitter, 1)

        # Left column -----------------------------------------------------
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.image_list = QtWidgets.QListWidget()
        self.image_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.image_list.setAlternatingRowColors(True)
        left_layout.addWidget(self.image_list, 1)

        navigation_row = QtWidgets.QHBoxLayout()
        self.prev_button = QtWidgets.QPushButton("上一张")
        self.next_button = QtWidgets.QPushButton("下一张")
        navigation_row.addWidget(self.prev_button)
        navigation_row.addWidget(self.next_button)
        left_layout.addLayout(navigation_row)

        splitter.addWidget(left_panel)

        # Right column ----------------------------------------------------
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.canvas_scroll = QtWidgets.QScrollArea()
        self.canvas_scroll.setWidgetResizable(True)
        self.canvas_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.canvas_scroll.setAlignment(QtCore.Qt.AlignCenter)

        self.canvas = ImageCanvas()
        self.canvas_scroll.setWidget(self.canvas)
        right_layout.addWidget(self.canvas_scroll, 3)

        form_row = QtWidgets.QHBoxLayout()
        self.label_edit = QtWidgets.QLineEdit()
        self.label_edit.setPlaceholderText("标签名称…")
        form_row.addWidget(self.label_edit, 1)
        self.save_button = QtWidgets.QPushButton("保存标注")
        form_row.addWidget(self.save_button)
        self.clear_button = QtWidgets.QPushButton("撤销掩膜")
        form_row.addWidget(self.clear_button)
        right_layout.addLayout(form_row)

        annotation_header = QtWidgets.QHBoxLayout()
        annotation_header.addWidget(QtWidgets.QLabel("已有标注"))
        annotation_header.addStretch(1)
        self.delete_button = QtWidgets.QPushButton("删除选中")
        self.delete_button.setEnabled(False)
        annotation_header.addWidget(self.delete_button)
        right_layout.addLayout(annotation_header)

        self.annotation_list = QtWidgets.QListWidget()
        self.annotation_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.annotation_list.setAlternatingRowColors(True)
        right_layout.addWidget(self.annotation_list, 2)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
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
        self.delete_button.clicked.connect(self._delete_selected_annotation)
        self.annotation_list.itemSelectionChanged.connect(self._on_annotation_selected)
        self.format_combo.currentTextChanged.connect(self._on_format_changed)

    def _setup_shortcuts(self) -> None:
        self._shortcuts: List[QtWidgets.QShortcut] = []
        for key, handler in (
            ("A", self._go_previous),
            ("D", self._go_next),
            ("Q", self._clear_pending_annotation),
        ):
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.ApplicationShortcut)
            shortcut.activated.connect(handler)
            self._shortcuts.append(shortcut)

    # -------------------------------------------------------------- helpers ---
    def _set_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def _reset_pending_state(self, clear_label: bool = True) -> None:
        self.pending_mask = None
        self.pending_bbox_display = None
        self.pending_bbox_original = None
        if clear_label and hasattr(self, "label_edit"):
            self.label_edit.clear()

    def _available_canvas_space(self) -> Tuple[int, int]:
        if not hasattr(self, "canvas_scroll") or self.canvas_scroll is None:
            return MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT
        viewport = self.canvas_scroll.viewport()
        width = viewport.width()
        height = viewport.height()
        if width <= 0 or height <= 0:
            width = max(self.canvas_scroll.width(), 1)
            height = max(self.canvas_scroll.height(), 1)
        return max(width, 1), max(height, 1)

    def _update_display_image(self, preserve_label: bool = True) -> None:
        if self.original_image is None:
            self.canvas.clear()
            return

        self._reset_pending_state(clear_label=not preserve_label)

        h, w = self.original_image.shape[:2]
        avail_w, avail_h = self._available_canvas_space()
        if avail_w <= 0 or avail_h <= 0:
            avail_w, avail_h = MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT
        scale = min(avail_w / w, avail_h / h, 1.0)
        if scale <= 0:
            scale = 1.0

        display_size = (int(round(w * scale)), int(round(h * scale)))
        if scale != 1.0:
            display = cv2.resize(
                self.original_image,
                display_size,
                interpolation=cv2.INTER_AREA,
            )
        else:
            display = self.original_image.copy()

        self.display_scale = scale
        self.display_image = display
        self.segmentor.Set_Image(display.copy())
        self._render_with_overlays()

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
        self._set_status(f"共 {len(self.image_files)} 张图片")

    def _select_save_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择保存路径")
        if directory:
            self.save_path = Path(directory)
            self._set_status(f"标注保存至：{directory}")

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
        channels = image.shape[2] if image.ndim == 3 else 1
        self.original_size = (w, h, channels)

        self.current_labels = []
        self.annotation_list.clear()
        self.delete_button.setEnabled(False)

        self.current_image_label.setText(os.path.basename(path))
        self._update_display_image(preserve_label=False)
        self._load_existing_annotations()
        self._set_status("等待点击生成掩膜…")

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
        self.canvas.setPixmap(pixmap)
        self.canvas.resize(pixmap.size())
        self.canvas.update()

    def _render_with_overlays(self) -> None:
        if self.display_image is None:
            return
        canvas = self.display_image.copy()
        scale = self.display_scale or 1.0
        for label in self.current_labels:
            x_min, y_min, width_or_xmax, height_or_ymax = label["bndbox"][:4]
            width = width_or_xmax
            height = height_or_ymax
            if width <= 0 and width_or_xmax > x_min:
                width = width_or_xmax - x_min
            if height <= 0 and height_or_ymax > y_min:
                height = height_or_ymax - y_min
            if width <= 0 or height <= 0:
                continue
            x1 = int(round(x_min * scale))
            y1 = int(round(y_min * scale))
            x2 = int(round((x_min + width) * scale))
            y2 = int(round((y_min + height) * scale))
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

        scale = self.display_scale or 1.0
        x_orig = int(round(self.segmentor.x / scale))
        y_orig = int(round(self.segmentor.y / scale))
        w_orig = int(round(self.segmentor.w / scale))
        h_orig = int(round(self.segmentor.h / scale))
        self.pending_bbox_original = (x_orig, y_orig, w_orig, h_orig)

        self._show_on_canvas(mask_image)
        self._set_status("已生成候选掩膜，填写标签后保存")

    def _clear_pending_annotation(self) -> None:
        if self.pending_bbox_display is None and self.pending_mask is None:
            self._set_status("没有掩膜可以撤销")
            return
        # 16777219 == Qt.Key_Backspace to mirror the original behaviour
        self.segmentor.Key_Event(16777219)
        self._reset_pending_state(clear_label=False)
        self._render_with_overlays()
        self._set_status("已撤销当前掩膜")

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

        self.segmentor.Key_Event(83)  # Qt.Key_S: confirm the mask inside SAM
        self._reset_pending_state()
        self._render_with_overlays()
        self._set_status("已保存标注")

    def _delete_selected_annotation(self) -> None:
        row = self.annotation_list.currentRow()
        if row < 0 or row >= len(self.current_labels):
            return

        del self.current_labels[row]
        self.annotation_list.takeItem(row)

        if self.current_image_path and self.save_path:
            if self.current_labels:
                base = Path(self.current_image_path)
                self._persist_annotations(base, base.stem, self.original_size, self.current_labels)
            else:
                self._remove_annotation_files()
        self._render_with_overlays()
        self._set_status("已删除标注")
        self.delete_button.setEnabled(False)

    def _on_annotation_selected(self) -> None:
        has_selection = self.annotation_list.currentRow() >= 0
        self.delete_button.setEnabled(has_selection)

    def _remove_annotation_files(self) -> None:
        if self.save_path is None or self.current_image_path is None:
            return
        base_path = self.save_path / Path(self.current_image_path).stem
        xml_path = base_path.with_suffix(".xml")
        txt_path = base_path.with_suffix(".txt")
        for path in (xml_path, txt_path):
            if path.exists():
                path.unlink()

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
            xml_labels: List[dict] = []
            for label in labels:
                x_min, y_min, width, height = label["bndbox"][:4]
                xmax = x_min + width
                ymax = y_min + height
                xml_labels.append({**label, "bndbox": [x_min, y_min, xmax, ymax]})
            xml(str(image_path), str(xml_path), size, xml_labels)
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
        self.annotation_list.clear()
        for fmt in preferred:
            if fmt == "YOLO":
                loaded, _boxes, names = load_yolo_labels(
                    base_path.with_suffix(".txt"),
                    self.original_size[0],
                    self.original_size[1],
                )
                if not loaded:
                    continue
                labels = loaded
                for name in names:
                    self.annotation_list.addItem(name)
                break
            else:
                xml_path = base_path.with_suffix(".xml")
                if not xml_path.exists():
                    continue
                labels = get_labels(str(xml_path))
                for label in labels:
                    self.annotation_list.addItem(label["name"])
                for raw in labels:
                    xmin, ymin, xmax, ymax = raw["bndbox"]
                    width = xmax - xmin if xmax > xmin else xmax
                    height = ymax - ymin if ymax > ymin else ymax
                    raw["bndbox"] = [xmin, ymin, width, height]
                break

        self.current_labels = labels
        self._reset_pending_state()
        self.annotation_list.clearSelection()
        self.delete_button.setEnabled(False)
        self._render_with_overlays()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self.original_image is not None:
            self._update_display_image()


def main() -> None:
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = LabelerMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
