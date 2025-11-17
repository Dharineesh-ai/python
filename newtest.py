# full_tiff_viewer_with_autofix.py
import sys
import time
from datetime import datetime
import psutil

import numpy as np
import cv2
import tifffile
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from PyQt5.QtWidgets import QProgressBar, QSizePolicy
from PyQt5.QtCore import QThread, pyqtSignal, QSize

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QScrollArea, QVBoxLayout, QWidget, QScrollBar,
    QListWidgetItem, QPushButton, QLabel, QListWidget, QDialog
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap

import matplotlib.patches as patches

import os

# Skip GUI + TIFF loading when running in GitHub Actions
if os.getenv("CI") == "true":
    print("Running in CI mode — skipping TIFF loading and GUI.")
    exit(0)

# Ensure non-interactive backend for faster rendering when needed
matplotlib.use("Agg")


def auto_fix_bgr_rgb(img_rgb):
    """
    Auto-detect if a uint8 3-channel image is actually BGR and flip to RGB.
    Returns the (possibly flipped) image.
    Works best on uint8 images; if dtype differs it will still attempt to evaluate.
    """
    if img_rgb is None:
        return None
    arr = np.ascontiguousarray(img_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return arr
    # convert to float for mean calculation safely
    a = arr.astype(np.float32)
    # compute mean intensity per channel (assume channel order is currently R,G,B or B,G,R)
    # We detect if the blue channel is significantly higher than red channel (common sign of BGR ordering)
    # Note: the indexing below treats arr[:,:,0] as R-like by convention; if it's BGR, channel0 is B.
    r_mean = float(a[:, :, 0].mean())
    g_mean = float(a[:, :, 1].mean())
    b_mean = float(a[:, :, 2].mean())
    # Heuristic: if blue channel is significantly larger than red channel, image likely BGR
    if b_mean > r_mean * 1.4 and (b_mean - r_mean) > 10:
        # flip channel order BGR -> RGB
        fixed = arr[:, :, ::-1].copy()
        print("⚠ Auto-fix: detected probable BGR ordering — converted to RGB")
        return fixed
    return arr


def ensure_uint8_rgb(img):
    """
    Convert an image (any dtype, 1/3/4 channels) into a uint8 RGB image.
    - Normalizes numeric range to 0..255 if dtype != uint8
    - Converts single-channel -> RGB
    - Converts 4-channel -> RGBA -> RGB
    - If channels are identical (pure grayscale stored as 3-channel),
      collapses to a proper RGB copy (still colorless but proper dtype).
    Also applies auto_fix_bgr_rgb before returning 3-channel arrays.
    """
    if img is None:
        return None
    img = np.ascontiguousarray(img)

    # Convert to float for safe normalization if not uint8
    if img.dtype != np.uint8:
        f = img.astype(np.float32)
        mn = float(np.min(f))
        mx = float(np.max(f))
        if mx > mn:
            f = (f - mn) / (mx - mn) * 255.0
        else:
            f = np.zeros_like(f, dtype=np.float32)
        img8 = np.clip(f, 0, 255).astype(np.uint8)
    else:
        img8 = img

    # Now handle channels
    if img8.ndim == 2:
        rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
        return auto_fix_bgr_rgb(rgb)
    elif img8.ndim == 3:
        ch = img8.shape[2]
        if ch == 1:
            rgb = cv2.cvtColor(img8[:, :, 0], cv2.COLOR_GRAY2RGB)
            return auto_fix_bgr_rgb(rgb)
        if ch == 3:
            # detect pure-grayscale stored in 3 channels (all channels identical)
            if np.array_equal(img8[:, :, 0], img8[:, :, 1]) and np.array_equal(img8[:, :, 1], img8[:, :, 2]):
                rgb = cv2.cvtColor(img8[:, :, 0], cv2.COLOR_GRAY2RGB)
                return auto_fix_bgr_rgb(rgb)
            # else assume RGB or BGR; apply auto-fix heuristic
            return auto_fix_bgr_rgb(img8)
        if ch == 4:
            try:
                # try BGRA -> RGB then check
                candidate = cv2.cvtColor(img8, cv2.COLOR_BGRA2RGB)
                # if candidate appears grayscale, try RGBA->RGB instead
                if np.array_equal(candidate[:, :, 0], candidate[:, :, 1]) and np.array_equal(candidate[:, :, 1], candidate[:, :, 2]):
                    candidate2 = cv2.cvtColor(img8, cv2.COLOR_RGBA2RGB)
                    return auto_fix_bgr_rgb(candidate2)
                return auto_fix_bgr_rgb(candidate)
            except Exception:
                return auto_fix_bgr_rgb(img8[:, :, :3])
    # else unexpected shape -> attempt to coerce to RGB by flattening
    flat = img8
    if flat.ndim == 1:
        flat = flat.reshape((1, -1))
        return ensure_uint8_rgb(flat)
    return img8


class ColorSelector(QDialog):
    """Dialog for assigning colors to TIFF frames."""
    def __init__(self, num_frames, default_colors=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Colors to TIFF Frames")
        self.setModal(True)

        if not default_colors:
            default_colors = ["Red", "Green", "Blue", "Alpha"]

        self.color_list = default_colors[:num_frames]

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Drag to reorder colors for each frame:"))

        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QListWidget.InternalMove)
        for color in self.color_list:
            self.list_widget.addItem(QListWidgetItem(color))
        layout.addWidget(self.list_widget)

        confirm_button = QPushButton("Confirm")
        confirm_button.clicked.connect(self.accept)
        layout.addWidget(confirm_button)

    def get_selected_colors(self):
        return [self.list_widget.item(i).text()
                for i in range(self.list_widget.count())]


class DraggableOverlay(QLabel):
    """
    QLabel-based overlay that displays a zoomed QPixmap and supports dragging.
    Parent should be the FigureCanvas (MplCanvas) so overlay coordinates align with canvas.
    """
    def __init__(self, parent=None, size=400):
        super().__init__(parent)
        self.setWindowFlags(Qt.Widget)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #666;
                background: rgba(0,0,0,0.8);
                border-radius: 6px;
            }
        """)
        self.setScaledContents(True)
        self.overlay_size = size
        self.resize(self.overlay_size, self.overlay_size)
        self._dragging = False
        self._drag_start_pos = QPoint(0, 0)
        self.setVisible(False)  # start hidden until mouse over image

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start_pos = event.globalPos() - self.pos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            new_pos = event.globalPos() - self._drag_start_pos
            parent_pos = self.parent().mapFromGlobal(new_pos)
            pw = self.parent().width()
            ph = self.parent().height()
            ow = self.width()
            oh = self.height()
            x = max(0, min(parent_pos.x(), pw - ow))
            y = max(0, min(parent_pos.y(), ph - oh))
            self.move(x, y)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def update_image_from_ndarray(self, arr):
        """
        Accepts a numpy array (grayscale or color) and updates the QLabel pixmap.
        For 3-channel arrays, expects RGB order (this matches tifffile/Matplotlib).
        """
        if arr is None or arr.size == 0:
            return

        # Ensure it's uint8 RGB (or grayscale uint8)
        arr_rgb = ensure_uint8_rgb(arr)

        if arr_rgb is None:
            return

        if arr_rgb.ndim == 3 and arr_rgb.shape[2] == 3:
            qimg = QImage(arr_rgb.data.tobytes(), arr_rgb.shape[1], arr_rgb.shape[0], arr_rgb.strides[0], QImage.Format_RGB888)
        elif arr_rgb.ndim == 2:
            qimg = QImage(arr_rgb.data.tobytes(), arr_rgb.shape[1], arr_rgb.shape[0], arr_rgb.strides[0], QImage.Format_Grayscale8)
        else:
            gray = cv2.cvtColor(arr_rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
            qimg = QImage(gray.data.tobytes(), gray.shape[1], gray.shape[0], gray.strides[0], QImage.Format_Grayscale8)

        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.overlay_size, self.overlay_size, Qt.KeepAspectRatio)
        self.setPixmap(pix)
        self.setVisible(True)
        self.raise_()


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for displaying single TIFF frames with magnifier overlay."""
    def __init__(self, width, height, dpi):
        fig_width_in = width / dpi
        fig_height_in = height / dpi
        self.fig = Figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])

        super().__init__(self.fig)

        # event connections
        self.mpl_connect("draw_event", self.on_draw_event)
        self.mpl_connect("motion_notify_event", self.on_mouse_motion)
        self.mpl_connect("button_release_event", self.update_overview_rect)
        self.mpl_connect("key_press_event", self.on_key_press)
        self.mpl_connect("resize_event", self.on_resize)
        self.mpl_connect("button_press_event", self.on_mouse_press)

        self.draw_start = time.time()

        # colormaps
        self.red_cmap = LinearSegmentedColormap.from_list("red_map", [(0, "black"), (1, "red")])
        self.green_cmap = LinearSegmentedColormap.from_list("green_map", [(0, "black"), (1, "green")])

        # state
        self.main_img = None
        self.original_img = None  # uint8 RGB copy for overlay/overview
        self.overview_img = None
        self.overview_rect = None
        self.mag_patch = None
        self.img_shape = None
        self.current_cmap = "gray"

        # overlay widget
        self.overlay = DraggableOverlay(parent=self, size=400)
        # initial overlay position
        self.overlay.move(max(10, self.width() - self.overlay.overlay_size - 10),
                          max(10, self.height() - self.overlay.overlay_size - 10))

        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def on_resize(self, event):
        try:
            w, h = self.get_width_height()
            dpi = self.fig.dpi
            self.fig.set_size_inches(w / dpi, h / dpi)
        except Exception:
            pass
        # make sure overlay remains visible inside bounds
        try:
            pw = self.width()
            ph = self.height()
            ow = self.overlay.width()
            oh = self.overlay.height()
            x = min(self.overlay.x(), max(0, pw - ow))
            y = min(self.overlay.y(), max(0, ph - oh))
            self.overlay.move(max(0, x), max(0, y))
        except Exception:
            pass

    def show_frame(self, display_img, label="", original_color=None):
        """
        display_img: 2D grayscale or 3-channel image to show on main axis (we show grayscale by default)
        original_color: full-color image (any dtype) used for overlay and overview (converted to uint8 RGB)
        """
        # choose cmap
        if label == "Red":
            cmap = self.red_cmap
            self.current_cmap = cmap
        elif label == "Green":
            cmap = self.green_cmap
            self.current_cmap = cmap
        else:
            cmap = "gray"
            self.current_cmap = cmap

        # convert display_img -> uint8 grayscale (disp8)
        if display_img is None:
            return
        if display_img.dtype != np.uint8:
            f = display_img.astype(np.float32)
            mn, mx = float(np.min(f)), float(np.max(f))
            if mx > mn:
                f = (f - mn) / (mx - mn) * 255.0
            else:
                f = np.zeros_like(f)
            disp8 = np.clip(f, 0, 255).astype(np.uint8)
        else:
            disp8 = display_img

        # set or update main imshow
        if self.main_img is None:
            self.ax.clear()
            if disp8.ndim == 2:
                self.main_img = self.ax.imshow(disp8, cmap=cmap, interpolation="nearest", animated=True)
            else:
                self.main_img = self.ax.imshow(disp8, interpolation="nearest", animated=True)
            self.ax.axis("off")
        else:
            self.main_img.set_data(disp8)
            if disp8.ndim == 2:
                self.main_img.set_cmap(cmap)

        self.ax.set_title(label)
        self.img_shape = disp8.shape[:2]

        # store original color version (ensure uint8 RGB and auto-fix BGR if needed)
        if original_color is not None:
            try:
                oc = ensure_uint8_rgb(original_color)
                self.original_img = oc
            except Exception:
                self.original_img = None

        # update overview inset (portrait mode only as before)
        h, w = disp8.shape[:2]
        if h > w:
            if self.original_img is not None:
                over_img = self.original_img.copy()
            else:
                over_img = disp8.copy()

            # resize preview
            ratio = (self.fig.bbox.height / 2) / h
            new_w = max(1, int(ratio * w))
            new_h = max(1, int(ratio * h))
            over_img_resized = cv2.resize(over_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            over_img_resized = ensure_uint8_rgb(over_img_resized)

            new_width = over_img_resized.shape[1] / self.fig.bbox.width
            if not hasattr(self, "over_ax") or self.over_ax is None:
                self.over_ax = self.fig.add_axes([1 - new_width, 0.5, new_width, 0.5])
                self.over_ax.axis("off")
                self.overview_img = None
                self.overview_rect = None
            else:
                self.over_ax.set_position([1 - new_width, 0.5, new_width, 0.5])
                self.over_ax.set_visible(True)

            if self.overview_img is None:
                self.over_ax.clear()
                if over_img_resized.ndim == 3:
                    self.overview_img = self.over_ax.imshow(over_img_resized, interpolation="nearest")
                else:
                    self.overview_img = self.over_ax.imshow(over_img_resized, cmap="gray", interpolation="nearest")
                self.over_ax.axis("off")
            else:
                self.overview_img.set_data(over_img_resized)

            if self.overview_rect is None:
                self.overview_rect = patches.Rectangle((0, 0),
                                                      over_img_resized.shape[1],
                                                      over_img_resized.shape[0],
                                                      edgecolor="red",
                                                      facecolor="none",
                                                      linewidth=1)
                self.over_ax.add_patch(self.overview_rect)
        else:
            if hasattr(self, "over_ax") and self.over_ax:
                self.over_ax.clear()
                self.over_ax.set_visible(False)
            self.overview_img = None
            self.overview_rect = None

        # reset view
        self.ax.set_xlim(0, self.img_shape[1])
        self.ax.set_ylim(self.img_shape[0], 0)

        self.draw_start = time.time()
        self.draw_idle()

    def on_draw_event(self, event):
        label = self.ax.get_title()
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        draw_time_ms = (time.time() - self.draw_start) * 1000
        print(f"[{label}] draw() completed at {timestamp} (draw time: {draw_time_ms:.2f} ms)")

    def on_key_press(self, event):
        key = event.key
        if key is None:
            return
        if 'ctrl' in str(key):
            if '=' in key or '+' in key:
                self.zoom_in()
            elif '-' in key:
                self.zoom_out()
        else:
            if key in ('+', 'equal'):
                self.zoom_in()
            elif key in ('-', 'minus'):
                self.zoom_out()

    def zoom_in(self, factor=0.7):
        if self.img_shape is None:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        width = (x1 - x0) * factor
        height = (y1 - y0) * factor
        self.ax.set_xlim(cx - width / 2.0, cx + width / 2.0)
        self.ax.set_ylim(cy + height / 2.0, cy - height / 2.0)
        self.update_overview_rect()
        self.draw_idle()

    def zoom_out(self, factor=1.4):
        if self.img_shape is None:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        width = (x1 - x0) * factor
        height = (y1 - y0) * factor
        width = min(width, self.img_shape[1])
        height = min(height, self.img_shape[0])
        self.ax.set_xlim(max(0, cx - width / 2.0), min(self.img_shape[1], cx + width / 2.0))
        self.ax.set_ylim(min(self.img_shape[0], cy + height / 2.0), max(0, cy - height / 2.0))
        self.update_overview_rect()
        self.draw_idle()

    def on_mouse_press(self, event):
        pass

    def on_mouse_motion(self, event):
        if event.inaxes != self.ax or self.main_img is None:
            if self.mag_patch:
                self.mag_patch.set_visible(False)
                self.draw_idle()
                self.update_overview_rect(event)
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        img = self.main_img.get_array()
        h, w = img.shape[:2]

        region_half = 50
        xi = int(round(x))
        yi = int(round(y))
        left = max(0, xi - region_half)
        right = min(w, xi + region_half)
        top = max(0, yi - region_half)
        bottom = min(h, yi + region_half)

        if (right - left) < 10 or (bottom - top) < 10:
            left = max(0, xi - region_half)
            right = min(w, xi + region_half)
            top = max(0, yi - region_half)
            bottom = min(h, yi + region_half)

        color_crop = None
        if getattr(self, "original_img", None) is not None:
            try:
                color_crop = self.original_img[top:bottom, left:right]
            except Exception:
                color_crop = None

        if color_crop is None or color_crop.size == 0:
            crop = img[top:bottom, left:right]
            if crop.size == 0:
                return
            if crop.ndim == 2:
                color_crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            else:
                color_crop = crop.copy()

        color_crop = ensure_uint8_rgb(color_crop)

        try:
            zoomed_size = self.overlay.overlay_size
            zoomed = cv2.resize(color_crop, (zoomed_size, zoomed_size), interpolation=cv2.INTER_CUBIC)
        except Exception:
            zoomed = cv2.resize(color_crop, (self.overlay.overlay_size, self.overlay.overlay_size),
                                interpolation=cv2.INTER_NEAREST)

        self.overlay.update_image_from_ndarray(zoomed)

        if self.mag_patch is None:
            self.mag_patch = patches.Rectangle((left, top), right - left, bottom - top,
                                               facecolor='none', edgecolor='yellow', linewidth=1)
            self.ax.add_patch(self.mag_patch)
        else:
            self.mag_patch.set_bounds(left, top, right - left, bottom - top)
            self.mag_patch.set_visible(True)

        self.update_overview_rect(event)
        self.draw_idle()

    def update_overview_rect(self, event=None):
        if not (hasattr(self, "overview_img") and self.overview_img and self.overview_rect and self.main_img):
            return

        try:
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
        except Exception:
            return

        main_h, main_w = self.main_img.get_array().shape[:2]
        ov_arr = self.overview_img.get_array()
        ov_h, ov_w = ov_arr.shape[:2]

        rect_x = max(0, x0 / main_w * ov_w)
        rect_y = max(0, y1 / main_h * ov_h)
        rect_w = (x1 - x0) / main_w * ov_w
        rect_h = (y0 - y1) / main_h * ov_h

        rect_x = max(0, min(ov_w, rect_x))
        rect_y = max(0, min(ov_h, rect_y))
        rect_w = max(1, min(ov_w - rect_x, rect_w))
        rect_h = max(1, min(ov_h - rect_y, rect_h))

        try:
            self.overview_rect.set_bounds(rect_x, rect_y, rect_w, rect_h)
        except Exception:
            pass


class MainWindow(QMainWindow):
    """Main TIFF viewer window with integrated magnifier/overview."""
    def __init__(self, tiff_path, width, height, dpi):
        super().__init__()
        self.setWindowTitle("TIFF Frame Viewer")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        scroll_area = QScrollArea()
        self.canvas = MplCanvas(width, height, dpi)
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)

        self.width = width
        self.height = height
        self.dpi = dpi

        layout = QVBoxLayout()
        layout.addWidget(scroll_area)

        self.scroll_bar = QScrollBar(Qt.Horizontal)
        self.scroll_bar.setStyleSheet("""
            QScrollBar:horizontal {
                background: #ddd;
            }
            QScrollBar::handle:horizontal {
                background: blue;
                min-width: 20px;
            }
        """)
        self.scroll_bar.setMinimum(0)
        self.scroll_bar.setPageStep(1)
        self.scroll_bar.setSingleStep(1)
        self.scroll_bar.valueChanged.connect(self.display_frame)
        layout.addWidget(self.scroll_bar)

        layout.addWidget(self.progress_bar)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.preload = self.should_preload(tiff_path)

        self.load_tiff(tiff_path)

    def should_preload(self, path):
        file_size = os.path.getsize(path)
        available_ram = psutil.virtual_memory().available

        file_gb = file_size / (1024**3)
        ram_gb = available_ram / (1024**3)

        print(f"TIFF size: {file_gb:.2f} GB, Available RAM: {ram_gb:.2f} GB")

        ratio = file_size / available_ram

        if ratio < 0.25:
            print("➡ Preload mode (safe: TIFF <25% of free RAM)")
            return True
        elif ratio < 0.5:
            print("⚠ TIFF is between 25–50% of free RAM — preload is possible, "
                  "but may cause swapping if other apps are running.")
            return True
        else:
            print("➡ Lazy mode (TIFF too large for safe preload)")
            return False

    def load_tiff(self, path):
        self.tiff_path = path

        if self.preload:
            print("▶ Loading TIFF in background thread...")
            self.progress_bar.setVisible(True)

            self.loader_thread = TiffLoaderThread(path)
            self.loader_thread.progress.connect(self.progress_bar.setValue)
            self.loader_thread.finished.connect(self.on_tiff_loaded)
            self.loader_thread.start()
        else:
            self.tif = tifffile.TiffFile(path)
            self.frames_count = len(self.tif.pages)
            sample = self.tif.pages[0].asarray()
            self.buffer = np.empty_like(sample)

            dialog = ColorSelector(self.frames_count, parent=self)
            if dialog.exec_():
                self.selected_colors = dialog.get_selected_colors()
            else:
                self.selected_colors = ["Gray"] * self.frames_count

            self.scroll_bar.setMaximum(max(0, self.frames_count - 1))
            self.scroll_bar.setValue(0)

    @staticmethod
    def get_tiff_frame_count(tiff_path):
        frames = tifffile.imread(tiff_path)
        if frames.ndim in (3, 4):
            return frames.shape[0]
        raise ValueError("Unsupported TIFF shape.")

    def display_frame(self, index=None):
        index = self.scroll_bar.value()
        access_start = time.time()

        if self.preload:
            frame = self.frames[index]
        else:
            frame = self.tif.pages[index].asarray(out=self.buffer)

        orig_h, orig_w = frame.shape[:2]
        scale_factor = self.width / orig_w
        target_height = max(1, int(orig_h * scale_factor))

        frame_resized = cv2.resize(frame, (self.width, target_height), interpolation=cv2.INTER_AREA)

        access_time = (time.time() - access_start) * 1000
        print(f"Frame {index} access time: {access_time:.2f} ms")

        try:
            color = self.selected_colors[index]
        except Exception:
            color = "Gray"

        # Prepare display_img (grayscale) and original_color (uint8 RGB)
        if frame_resized.ndim == 2:
            display_img = frame_resized.copy()
            original_color = ensure_uint8_rgb(frame_resized)
        else:
            original_color = ensure_uint8_rgb(frame_resized)
            # convert to grayscale for main display (from RGB)
            try:
                display_img = cv2.cvtColor(original_color, cv2.COLOR_RGB2GRAY)
            except Exception:
                display_img = cv2.cvtColor(original_color, cv2.COLOR_BGR2GRAY)

        self.canvas.show_frame(display_img, color, original_color=original_color)

    def on_tiff_loaded(self, frames, count):
        print("✅ TIFF loaded in background.")
        self.frames = frames
        self.frames_count = count
        self.progress_bar.setVisible(False)

        dialog = ColorSelector(self.frames_count, parent=self)
        if dialog.exec_():
            self.selected_colors = dialog.get_selected_colors()
        else:
            self.selected_colors = ["Gray"] * self.frames_count

        self.scroll_bar.setMaximum(max(0, self.frames_count - 1))
        self.scroll_bar.setValue(0)
        if self.frames_count == 1:
            self.display_frame(0)


class TiffLoaderThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray, int)  # Emits (frames, frame_count)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        with tifffile.TiffFile(self.path) as tif:
            num_pages = len(tif.pages)
            frames = []
            for i, page in enumerate(tif.pages):
                frames.append(page.asarray())
                percent = int(((i + 1) / num_pages) * 100)
                self.progress.emit(percent)
            stacked = np.stack(frames, axis=0)
            self.finished.emit(stacked, num_pages)


if __name__ == "__main__":
    # Update path as required
    tiff_path = (r"D:\python-master\2024.08.01_cLift-Kontrolle_A0206-02_Rl50Gh35_1677.tif")

    app = QApplication(sys.argv)
    screen = app.primaryScreen()

    width = screen.size().width()
    height = screen.size().height()
    dpi = screen.logicalDotsPerInch()

    window = MainWindow(tiff_path, width, height, dpi)
    window.show()

    sys.exit(app.exec_())
