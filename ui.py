import logging

import tifffile

from PyQt5.QtWidgets import (QApplication, QMainWindow, QScrollArea, QVBoxLayout,
                             QWidget, QScrollBar, QLabel, QProgressBar, QHBoxLayout,
                             QMessageBox, QListWidget, QListWidgetItem, QPushButton, QDialog)
from PyQt5.QtCore import Qt
from typing import List, Optional

from canvas import MplCanvas
from loader import TiffLoaderThread
from utils import should_preload
from imageproc import ensure_uint8_rgb, apply_color_selection
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class ColorSelector(QDialog):
    def __init__(self, num_frames: int, default_colors: Optional[List[str]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Colors to TIFF Frames")
        self.setModal(True)
        if not default_colors:
            default_colors = ["Red", "Green", "Blue", "Alpha"]
        self.color_list = default_colors * num_frames
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Drag to reorder colors for each frame"))
        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QListWidget.InternalMove)
        for color in self.color_list:
            self.list_widget.addItem(QListWidgetItem(color))
        layout.addWidget(self.list_widget)
        confirm_button = QPushButton("Confirm")
        confirm_button.clicked.connect(self.accept)
        layout.addWidget(confirm_button)

    def get_selected_colors(self) -> List[str]:
        return [self.list_widget.item(i).text().strip()
                for i in range(self.list_widget.count())]

class MainWindow(QMainWindow):
    def __init__(self, tiff_path: str, width: int, height: int, dpi: float):
        super().__init__()
        self.setWindowTitle("TIFF Frame Viewer - Landscape Mode")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(20)
        self.first_draw_done = False
        self.data_ready = False
        scroll_area = QScrollArea()
        self.canvas = MplCanvas(width, height, dpi)
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area = scroll_area
        self.width_ = width
        self.height_ = height
        self.dpi = dpi
        layout = QVBoxLayout()
        layout.addWidget(scroll_area)
        self.scroll_bar = QScrollBar(Qt.Horizontal)
        self.scroll_bar.setStyleSheet("""
            QScrollBar:horizontal { background: #333; min-width: 20px; border-radius: 8px; }
        """)
        self.scroll_bar.valueChanged.connect(self.display_frame)
        layout.addWidget(self.scroll_bar)
        layout.addWidget(self.progress_bar)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.preload_ = should_preload(tiff_path)
        self.load_tiff(tiff_path)

    def showEvent(self, event):
        super().showEvent(event)
        if (not self.first_draw_done and self.data_ready
            and hasattr(self, 'scroll_bar')):
            self.first_draw_done = True
            self.display_frame(self.scroll_bar.value())
            self.show_axes_size_popup()

    def load_tiff(self, path: str):
        self.tiff_path = path
        if self.preload_:
            logger.info("Loading entire TIFF into memory...")
            self.progress_bar.setVisible(True)
            self.loader_thread = TiffLoaderThread(path)
            self.loader_thread.progress.connect(self.progress_bar.setValue)
            self.loader_thread.finished.connect(self.on_tiff_loaded)
            self.loader_thread.start()
        else:
            self.tif = tifffile.TiffFile(path)
            self.frames_count = len(self.tif.pages)
            self.data_ready = True
            sample = self.tif.pages[0].asarray()
            self.buffer = np.empty_like(sample)
            dialog = ColorSelector(self.frames_count, parent=self)
            self.selected_colors = dialog.get_selected_colors() if dialog.exec_() else ["Gray"] * self.frames_count
            self.selected_colors = [c.strip().title() for c in self.selected_colors]
            self.scroll_bar.setMaximum(max(0, self.frames_count - 1))
            self.scroll_bar.setValue(0)
            self.display_frame(0)
            self.show_axes_size_popup()

    def on_tiff_loaded(self, frames, count):
        self.frames = frames
        self.frames_count = count
        self.data_ready = True
        self.progress_bar.setVisible(False)
        dialog = ColorSelector(self.frames_count, parent=self)
        self.selected_colors = dialog.get_selected_colors() if dialog.exec_() else ["Gray"] * self.frames_count
        self.selected_colors = [c.strip().title() for c in self.selected_colors]
        self.scroll_bar.setMaximum(max(0, self.frames_count - 1))
        self.scroll_bar.setValue(0)
        if self.isVisible() and not self.first_draw_done:
            self.first_draw_done = True
            self.display_frame(self.scroll_bar.value())
            self.show_axes_size_popup()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if (hasattr(self, 'scroll_bar') and hasattr(self, 'frames_count')
            and self.scroll_bar.value() == 0
            and (hasattr(self, 'frames') or hasattr(self, 'tif'))):
            self.display_frame(self.scroll_bar.value())

    def closeEvent(self, event):
        if hasattr(self, 'tif') and self.tif:
            self.tif.close()
        event.accept()

    def display_frame(self, index=None):
        # 1) Use scroll-area width for main image, leave room for preview bar (~260px)
        try:
            vp_w = int(self.scroll_area.viewport().width())
        except AttributeError:
            vp_w = self.width_
        preview_w = getattr(self.canvas, "preview_w", 240)
        effective_w = max(200, vp_w - preview_w - 16)

        # 2) Pick frame
        if index is None:
            index = int(self.scroll_bar.value())
        if self.preload_:
            frame = self.frames[index]
        else:
            frame = self.tif.pages[index].asarray(out=self.buffer)

        # 3) Normalize to uint8 (simple min-max, no brightness/contrast)
        if frame.dtype != np.uint8:
            fmin, fmax = frame.min(), frame.max()
            if fmax > fmin:
                frame = ((frame - fmin) / (fmax - fmin) * 255).astype(np.uint8)
            else:
                frame = np.zeros_like(frame, dtype=np.uint8)

        h_orig, w_orig = frame.shape[:2]
        color = self.selected_colors[index] if index < len(self.selected_colors) else "Gray"
        color_norm = (color or "").strip().title()

        # 4) Build RGB (no window/level adjustment)
        ndim = frame.ndim
        rgb_full = apply_color_selection(frame, color_norm, ndim)
        h, w = rgb_full.shape[:2]

        # 5) Resize to fit width, keep aspect ratio
        new_w = effective_w
        new_h = max(1, int(h * new_w / float(w)))
        frame_resized_rgb = cv2.resize(rgb_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = ensure_uint8_rgb(frame_resized_rgb)

        # 6) Send to canvas; keep original for magnifier/preview
        self.canvas.show_frame(rgb, color_norm, original_color=rgb_full)
        # 7) Let scroll-area control vertical scrolling, width fixed to effective_w
        self.canvas.setMinimumWidth(effective_w)
        self.canvas.setMaximumWidth(effective_w)
        self.canvas.resize(effective_w, new_h)
        self.canvas.draw_idle()

    def show_axes_size_popup(self):
        main_w = self.canvas.width()
        main_h = self.canvas.height()
        prev_w = getattr(self.canvas, 'preview_img_w', 0)
        prev_h = getattr(self.canvas, 'preview_img_h', 0)
        QMessageBox.information(
            self, "Image Dimensions",
            f"MAIN IMAGE AREA (red border):\nWidth: {main_w}px, Height: {main_h}px\n\n"
            f"PREVIEW IMAGE (inside red bar on right):\nWidth: {prev_w}px, Height: {prev_h}px")
