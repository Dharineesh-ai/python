# full_tiff_viewer_with_autofix.py
import sys
import time
from datetime import datetime
import os
import psutil
import numpy as np
import cv2
import tifffile

# If running in CI/headless, use Agg; otherwise use Qt5Agg
import matplotlib
if os.getenv("CI") == "true":
    print("Running in CI mode — using non-interactive Agg backend.")
    matplotlib.use("Agg")
else:
    matplotlib.use("Qt5Agg")

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QScrollArea, QVBoxLayout, QWidget, QScrollBar,
    QListWidgetItem, QPushButton, QLabel, QListWidget, QDialog, QProgressBar,
    QSizePolicy, QHBoxLayout
)
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap

# Skip GUI + TIFF loading when running in GitHub Actions
if os.getenv("CI") == "true" and __name__ == "__main__":
    print("Running in CI mode — skipping TIFF loading and GUI.")
    sys.exit(0)


def auto_fix_bgr_rgb(img_rgb):
    if img_rgb is None:
        return None
    arr = np.ascontiguousarray(img_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return arr
    a = arr.astype(np.float32)
    r_mean = float(a[:, :, 0].mean())
    g_mean = float(a[:, :, 1].mean())
    b_mean = float(a[:, :, 2].mean())
    if b_mean > r_mean * 1.4 and (b_mean - r_mean) > 10:
        fixed = arr[:, :, ::-1].copy()
        print("Warning: Auto-fix: detected probable BGR ordering — converted to RGB")
        return fixed
    return arr


def ensure_uint8_rgb(img):
    if img is None:
        return None
    img = np.ascontiguousarray(img)
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
    if img8.ndim == 2:
        rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
        return auto_fix_bgr_rgb(rgb)
    elif img8.ndim == 3:
        ch = img8.shape[2]
        if ch == 1:
            rgb = cv2.cvtColor(img8[:, :, 0], cv2.COLOR_GRAY2RGB)
            return auto_fix_bgr_rgb(rgb)
        if ch == 3:
            if np.array_equal(img8[:, :, 0], img8[:, :, 1]) and np.array_equal(img8[:, :, 1], img8[:, :, 2]):
                rgb = cv2.cvtColor(img8[:, :, 0], cv2.COLOR_GRAY2RGB)
                return auto_fix_bgr_rgb(rgb)
            return auto_fix_bgr_rgb(img8)
        if ch == 4:
            try:
                candidate = cv2.cvtColor(img8, cv2.COLOR_BGRA2RGB)
                if np.array_equal(candidate[:, :, 0], candidate[:, :, 1]) and np.array_equal(candidate[:, :, 1], candidate[:, :, 2]):
                    candidate2 = cv2.cvtColor(img8, cv2.COLOR_RGBA2RGB)
                    return auto_fix_bgr_rgb(candidate2)
                return auto_fix_bgr_rgb(candidate)
            except Exception:
                return auto_fix_bgr_rgb(img8[:, :, :3])
    return img8


class ColorSelector(QDialog):
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
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())]


class DraggableOverlay(QLabel):
    def __init__(self, parent=None, size=400):
        super().__init__(parent)
        self.setWindowFlags(Qt.Widget)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setStyleSheet("QLabel { border: 2px solid #666; background: rgba(0,0,0,0.8); border-radius: 6px; }")
        self.setScaledContents(True)
        self.overlay_size = size
        self.resize(self.overlay_size, self.overlay_size)
        self._dragging = False
        self._drag_start_pos = QPoint(0, 0)
        self.setVisible(False)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start_pos = event.globalPos() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            new_pos = event.globalPos() - self._drag_start_pos
            parent_pos = self.parent().mapFromGlobal(new_pos)
            pw, ph = self.parent().width(), self.parent().height()
            ow, oh = self.width(), self.height()
            x = max(0, min(parent_pos.x(), pw - ow))
            y = max(0, min(parent_pos.y(), ph - oh))
            self.move(x, y)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False

    def update_image_from_ndarray(self, arr):
        if arr is None or arr.size == 0:
            return
        arr_rgb = ensure_uint8_rgb(arr)
        if arr_rgb is None:
            return
        if arr_rgb.ndim == 3 and arr_rgb.shape[2] == 3:
            qimg = QImage(arr_rgb.data.tobytes(), arr_rgb.shape[1], arr_rgb.shape[0], arr_rgb.strides[0], QImage.Format_RGB888)
        else:
            gray = cv2.cvtColor(arr_rgb[:, :, :3], cv2.COLOR_RGB2GRAY) if arr_rgb.ndim == 3 else arr_rgb
            qimg = QImage(gray.data.tobytes(), gray.shape[1], gray.shape[0], gray.strides[0], QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg).scaled(self.overlay_size, self.overlay_size, Qt.KeepAspectRatio)
        self.setPixmap(pix)
        self.setVisible(True)
        self.raise_()


class MplCanvas(FigureCanvas):
    def __init__(self, width, height, dpi):
        fig_width_in = width / dpi
        fig_height_in = height / dpi
        self.fig = Figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        super().__init__(self.fig)

        self.mpl_connect("draw_event", self.on_draw_event)
        self.mpl_connect("motion_notify_event", self.on_mouse_motion)
        self.mpl_connect("button_release_event", self.update_overview_rect)
        self.mpl_connect("key_press_event", self.on_key_press)
        self.mpl_connect("resize_event", self.on_resize)

        self.draw_start = time.time()
        self.red_cmap = LinearSegmentedColormap.from_list("red_map", [(0, "black"), (1, "red")])
        self.green_cmap = LinearSegmentedColormap.from_list("green_map", [(0, "black"), (1, "green")])

        self.main_img = None
        self.original_img = None
        self.overview_img = None
        self.overview_rect = None
        self.mag_patch = None
        self.img_shape = None
        self.current_cmap = "gray"
        self.over_ax = None

        self.overlay = DraggableOverlay(parent=self, size=400)
        self.overlay.move(max(10, self.width() - 410), max(10, self.height() - 410))
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def on_resize(self, event):
        try:
            w, h = self.get_width_height()
            dpi = self.fig.dpi
            self.fig.set_size_inches(w / dpi, h / dpi)
        except: pass
        try:
            pw, ph = self.width(), self.height()
            ow, oh = self.overlay.width(), self.overlay.height()
            x = min(self.overlay.x(), max(0, pw - ow))
            y = min(self.overlay.y(), max(0, ph - oh))
            self.overlay.move(x, y)
        except: pass

    def show_frame(self, display_img, label="", original_color=None):
        if label == "Red":
            cmap = self.red_cmap
        elif label == "Green":
            cmap = self.green_cmap
        else:
            cmap = "gray"
        self.current_cmap = cmap

        if display_img is None:
            return

        if display_img.dtype != np.uint8:
            f = display_img.astype(np.float32)
            mn, mx = float(np.min(f)), float(np.max(f))
            if mx > mn:
                f = (f - mn) / (mx - mn) * 255.0
            disp8 = np.clip(f, 0, 255).astype(np.uint8)
        else:
            disp8 = display_img

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

        if original_color is not None:
            self.original_img = ensure_uint8_rgb(original_color)
            if hasattr(self, "over_ax") and self.over_ax:
                self.over_ax.clear()
                self.over_ax.set_visible(False)
            self.overview_img = None
            self.overview_rect = None
        else:
            over_img = disp8.copy()
            h, w = over_img.shape[:2]
            ratio = (self.fig.bbox.height / 2) / max(1, h)
            new_w = max(1, int(ratio * w))
            new_h = max(1, int(ratio * h))
            over_img_resized = cv2.resize(over_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            over_img_resized = ensure_uint8_rgb(over_img_resized)
            new_width = over_img_resized.shape[1] / max(1.0, self.fig.bbox.width)

            if not hasattr(self, "over_ax") or self.over_ax is None:
                self.over_ax = self.fig.add_axes([1 - new_width, 0.5, new_width, 0.5])
                self.over_ax.axis("off")
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
                self.overview_rect = patches.Rectangle((0, 0), over_img_resized.shape[1], over_img_resized.shape[0],
                                                       edgecolor="red", facecolor="none", linewidth=1)
                self.over_ax.add_patch(self.overview_rect)
            else:
                self.overview_img.set_data(over_img_resized)

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
        if event.key in ('+', '='):
            self.zoom_in()
        elif event.key in ('-', '_'):
            self.zoom_out()

    def zoom_in(self, factor=0.7):
        if not self.img_shape: return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w = (x1 - x0) * factor
        h = (y1 - y0) * factor
        self.ax.set_xlim(cx - w/2, cx + w/2)
        self.ax.set_ylim(cy + h/2, cy - h/2)
        self.update_overview_rect()
        self.draw_idle()

    def zoom_out(self, factor=1.4):
        if not self.img_shape: return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w = min((x1 - x0) * factor, self.img_shape[1])
        h = min((y1 - y0) * factor, self.img_shape[0])
        self.ax.set_xlim(max(0, cx - w/2), min(self.img_shape[1], cx + w/2))
        self.ax.set_ylim(min(self.img_shape[0], cy + h/2), max(0, cy - h/2))
        self.update_overview_rect()
        self.draw_idle()

    def on_mouse_motion(self, event):
        if not event.inaxes or not self.main_img:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None: return
        img = self.main_img.get_array()
        h, w = img.shape[:2]
        region = 50
        xi, yi = int(round(x)), int(round(y))
        l = max(0, xi - region)
        r = min(w, xi + region)
        t = max(0, yi - region)
        b = min(h, yi + region)

        if getattr(self, "original_img", None) is not None:
            crop = self.original_img[t:b, l:r]
        else:
            crop = img[t:b, l:r]
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

        crop = ensure_uint8_rgb(crop)
        zoomed = cv2.resize(crop, (self.overlay.overlay_size, self.overlay.overlay_size), interpolation=cv2.INTER_CUBIC)
        self.overlay.update_image_from_ndarray(zoomed)

        if self.mag_patch is None:
            self.mag_patch = patches.Rectangle((l, t), r-l, b-t, facecolor='none', edgecolor='yellow', linewidth=1)
            self.ax.add_patch(self.mag_patch)
        else:
            self.mag_patch.set_bounds(l, t, r-l, b-t)
            self.mag_patch.set_visible(True)

        self.update_overview_rect(event)
        self.draw_idle()

    def update_overview_rect(self, event=None):
        if not (self.overview_img and self.overview_rect and self.main_img):
            return
        try:
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            main_h, main_w = self.main_img.get_array().shape[:2]
            ov_h, ov_w = self.overview_img.get_array().shape[:2]
            rx = max(0, x0 / main_w * ov_w)
            ry = max(0, y1 / main_h * ov_h)
            rw = (x1 - x0) / main_w * ov_w
            rh = (y0 - y1) / main_h * ov_h
            self.overview_rect.set_bounds(rx, ry, max(1, rw), max(1, rh))
        except: pass


class TiffLoaderThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, int)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        with tifffile.TiffFile(self.path) as tif:
            num_pages = len(tif.pages)
            frames = []
            for i, page in enumerate(tif.pages):
                frames.append(page.asarray())
                self.progress.emit(int((i + 1) / num_pages * 100))
            try:
                stacked = np.stack(frames, axis=0)
            except:
                stacked = frames
            self.finished.emit(stacked, num_pages)


class MainWindow(QMainWindow):
    def __init__(self, tiff_path, width, height, dpi):
        super().__init__()
        self.setWindowTitle("TIFF Frame Viewer — Landscape Mode")
        self.rotation_mode = 0  # 0 = Landscape, 1 = Portrait (90° clockwise)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(20)

        # Canvas
        scroll_area = QScrollArea()
        self.canvas = MplCanvas(width, height, dpi)
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)

        self.width = width
        self.height = height
        self.dpi = dpi

        # Layout
        layout = QVBoxLayout()

        # Orientation buttons
        btn_layout = QHBoxLayout()
        self.landscape_btn = QPushButton("Landscape")
        self.portrait_btn = QPushButton("Portrait")
        self.landscape_btn.setCheckable(True)
        self.portrait_btn.setCheckable(True)
        self.landscape_btn.setChecked(True)

        self.landscape_btn.setShortcut("L")
        self.portrait_btn.setShortcut("P")

        self.landscape_btn.clicked.connect(self.set_landscape)
        self.portrait_btn.clicked.connect(self.set_portrait)

        style = "QPushButton:checked { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px; }"
        self.landscape_btn.setStyleSheet(style.replace("#4CAF50", "#4CAF50"))
        self.portrait_btn.setStyleSheet(style.replace("#4CAF50", "#2196F3"))

        btn_layout.addWidget(QLabel("Orientation:"))
        btn_layout.addWidget(self.landscape_btn)
        btn_layout.addWidget(self.portrait_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addWidget(scroll_area)

        # Frame slider
        self.scroll_bar = QScrollBar(Qt.Horizontal)
        self.scroll_bar.setStyleSheet("QScrollBar::handle:horizontal { background: #555; min-width: 20px; border-radius: 8px; }")
        self.scroll_bar.valueChanged.connect(self.display_frame)
        layout.addWidget(self.scroll_bar)

        layout.addWidget(self.progress_bar)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.preload = self.should_preload(tiff_path)
        self.load_tiff(tiff_path)

    def set_landscape(self):
        if self.rotation_mode == 0: return
        self.rotation_mode = 0
        self.landscape_btn.setChecked(True)
        self.portrait_btn.setChecked(False)
        self.setWindowTitle("TIFF Frame Viewer — Landscape Mode")
        print("Orientation → Landscape (0°)")
        self.display_current_frame()

    def set_portrait(self):
        if self.rotation_mode == 1: return
        self.rotation_mode = 1
        self.landscape_btn.setChecked(False)
        self.portrait_btn.setChecked(True)
        self.setWindowTitle("TIFF Frame Viewer — Portrait Mode")
        print("Orientation → Portrait (90° clockwise)")
        self.display_current_frame()

    def display_current_frame(self):
        current = self.scroll_bar.value()
        self.scroll_bar.blockSignals(True)
        self.display_frame(current)
        self.scroll_bar.blockSignals(False)

    def should_preload(self, path):
        file_size = os.path.getsize(path)
        available_ram = psutil.virtual_memory().available
        file_gb = file_size / (1024**3)
        ram_gb = available_ram / (1024**3)
        print(f"TIFF size: {file_gb:.2f} GB, Free RAM: {ram_gb:.2f} GB")
        return file_size < available_ram * 0.5

    def load_tiff(self, path):
        self.tiff_path = path
        if self.preload:
            print("Loading entire TIFF into memory...")
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
            self.selected_colors = dialog.get_selected_colors() if dialog.exec_() else ["Gray"] * self.frames_count
            self.scroll_bar.setMaximum(max(0, self.frames_count - 1))
            self.scroll_bar.setValue(0)

    def on_tiff_loaded(self, frames, count):
        print("TIFF fully loaded into RAM")
        self.frames = frames
        self.frames_count = count
        self.progress_bar.setVisible(False)
        dialog = ColorSelector(self.frames_count, parent=self)
        self.selected_colors = dialog.get_selected_colors() if dialog.exec_() else ["Gray"] * self.frames_count
        self.scroll_bar.setMaximum(max(0, self.frames_count - 1))
        self.scroll_bar.setValue(0)
        self.display_frame(0)

    def display_frame(self, _=None):
        index = int(self.scroll_bar.value())
        t0 = time.time()

        if self.preload:
            container = getattr(self, "frames", None)
            if container is None: return
            frame = container[index] if isinstance(container, (list, np.ndarray)) and len(container) > index else container[index]
        else:
            frame = self.tif.pages[index].asarray(out=self.buffer)

        # APPLY ROTATION
        if self.rotation_mode == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        h, w = frame.shape[:2]
        scale = self.width / max(1, w)
        target_h = max(1, int(h * scale))
        frame_resized = cv2.resize(frame, (self.width, target_h), interpolation=cv2.INTER_AREA)

        print(f"Frame {index} (+ rotate) time: {(time.time()-t0)*1000:.1f} ms")

        color = self.selected_colors[index] if index < len(self.selected_colors) else "Gray"

        if frame_resized.ndim == 2:
            display_img = frame_resized.copy()
            original_color = ensure_uint8_rgb(frame_resized)
        else:
            original_color = ensure_uint8_rgb(frame_resized)
            display_img = cv2.cvtColor(original_color, cv2.COLOR_RGB2GRAY)

        self.canvas.show_frame(display_img, color, original_color=original_color)


if __name__ == "__main__":
    # Change this path to your TIFF file
    tiff_path = r"D:\python-master\2024.08.01_cLift-Kontrolle_A0206-02_Rl50Gh35_1677.tif"

    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    width = screen.size().width()
    height = screen.size().height()
    dpi = screen.logicalDotsPerInch()

    window = MainWindow(tiff_path, width, height, dpi)
    window.showMaximized()
    sys.exit(app.exec_())