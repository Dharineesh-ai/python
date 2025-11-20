# newtest.py
# TIFF viewer (Option B) — image width always fits viewport width (no horizontal scroll)
# Restores correct Red / Green / Blue channel behaviour (handles 2D and 3D frames)
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
        # Normalize stored color names to Title case for consistency
        return [self.list_widget.item(i).text().strip() for i in range(self.list_widget.count())]


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
        except:
            pass
        try:
            pw, ph = self.width(), self.height()
            ow, oh = self.overlay.width(), self.overlay.height()
            x = min(self.overlay.x(), max(0, pw - ow))
            y = min(self.overlay.y(), max(0, ph - oh))
            self.overlay.move(x, y)
        except:
            pass

    def show_frame(self, display_img, label="", original_color=None):
        """
        FINAL WORKING VERSION
        Shows bright Red / Green / Blue in BOTH Landscape and Portrait modes
        """
        # ------------------------------------------------------------------
        # 1. Normalize to uint8 if needed
        # ------------------------------------------------------------------
        if display_img.dtype != np.uint8:
            img = display_img.astype(np.float32)
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) / (mx - mn) * 255.0
            else:
                img = np.zeros_like(img)
            disp8 = np.clip(img, 0, 255).astype(np.uint8)
        else:
            disp8 = display_img

        # ------------------------------------------------------------------
        # 2. True colour mode? (Red / Green / Blue channel)
        # Make the check case-insensitive to avoid label mismatches
        # ------------------------------------------------------------------
        label_norm = (label or "").strip().lower()
        true_colour = label_norm in ("red", "green", "blue")

        # ------------------------------------------------------------------
        # 3. First draw – create the imshow object
        # ------------------------------------------------------------------
        if self.main_img is None:
            self.ax.clear()
            if true_colour and disp8.ndim == 3:
                # Show real RGB – NO colormap!
                self.main_img = self.ax.imshow(disp8, interpolation="nearest", animated=True)
            else:
                # Grayscale path
                gray = disp8 if disp8.ndim == 2 else cv2.cvtColor(disp8, cv2.COLOR_RGB2GRAY)
                self.main_img = self.ax.imshow(gray, cmap="gray", interpolation="nearest", animated=True)
            self.ax.axis("off")

        # ------------------------------------------------------------------
        # 4. Subsequent frames – just update data (fast!)
        # ------------------------------------------------------------------
        else:
            if true_colour and disp8.ndim == 3:
                # Keep real RGB
                self.main_img.set_data(disp8)
                self.main_img.set_cmap(None)        # VERY IMPORTANT
            else:
                # Grayscale
                gray = disp8 if disp8.ndim == 2 else cv2.cvtColor(disp8, cv2.COLOR_RGB2GRAY)
                self.main_img.set_data(gray)
                self.main_img.set_cmap("gray")

        # ------------------------------------------------------------------
        # 5. Title + original colour for magnifier
        # ------------------------------------------------------------------
        # Show title in Title Case for clarity
        self.ax.set_title(label.title() if label else "")
        self.img_shape = disp8.shape[:2]

        if original_color is not None:
            self.original_img = ensure_uint8_rgb(original_color)
            if hasattr(self, "over_ax") and self.over_ax:
                self.over_ax.clear()
                self.over_ax.set_visible(False)
            self.overview_img = None
            self.overview_rect = None

        # ------------------------------------------------------------------
        # 6. Limits & redraw
        # ------------------------------------------------------------------
        try:
            self.ax.set_xlim(0, self.img_shape[1])
            self.ax.set_ylim(self.img_shape[0], 0)
        except:
            pass

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
        # kept for compatibility; overview not used in this simplified pipeline
        return


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
        # For Option B we want the canvas to be free-size vertically, but we hide horizontal scroll
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area = scroll_area

        self.width = width
        self.height = height
        self.dpi = dpi

        # Layout
        layout = QVBoxLayout()

        # Orientation buttons removed as requested
        # (Landscape / Portrait buttons and their callbacks were removed)

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
            # Normalize selected colors to Title case to be safe
            self.selected_colors = [c.strip().title() for c in self.selected_colors]
            self.scroll_bar.setMaximum(max(0, self.frames_count - 1))
            self.scroll_bar.setValue(0)
            # display first frame immediately
            self.display_frame(0)

    def on_tiff_loaded(self, frames, count):
        print("TIFF fully loaded into RAM")
        self.frames = frames
        self.frames_count = count
        self.progress_bar.setVisible(False)
        dialog = ColorSelector(self.frames_count, parent=self)
        self.selected_colors = dialog.get_selected_colors() if dialog.exec_() else ["Gray"] * self.frames_count
        self.selected_colors = [c.strip().title() for c in self.selected_colors]
        self.scroll_bar.setMaximum(max(0, self.frames_count - 1))
        self.scroll_bar.setValue(0)
        self.display_frame(0)

    def resizeEvent(self, event):
        """
        When main window resizes, re-display current frame so image width matches new viewport width.
        """
        try:
            super().resizeEvent(event)
        except:
            pass
        try:
            # call display_frame directly (previous display_current_frame removed)
            self.display_frame()
        except:
            pass

    def display_frame(self, _=None):
        index = int(self.scroll_bar.value())
        if self.preload:
            frame = self.frames[index]
        else:
            frame = self.tif.pages[index].asarray(out=self.buffer)

        # ---- NORMALIZE TO uint8 ----
        if frame.dtype != np.uint8:
            fmin, fmax = frame.min(), frame.max()
            if fmax > fmin:
                frame = ((frame - fmin) / (fmax - fmin) * 255).astype(np.uint8)
            else:
                frame = np.zeros_like(frame, dtype=np.uint8)

        # ---- RESIZE TO FIT VIEWPORT WIDTH (no horizontal scroll) ----
        try:
            viewport_w = int(self.scroll_area.viewport().width())
        except:
            viewport_w = 0

        if viewport_w <= 1:
            try:
                canvas_w = int(self.canvas.get_width_height()[0])
            except:
                canvas_w = max(100, self.width)
            viewport_w = canvas_w

        h, w = frame.shape[:2]
        new_w = max(1, int(viewport_w))
        new_h = max(1, int(h * (new_w / float(w))))
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # ---- COLOUR ASSIGNMENT: robust to 2D (gray) or 3D (color) frames ----
        color = self.selected_colors[index] if index < len(self.selected_colors) else "Gray"
        color_norm = (color or "").strip().title()  # "Green", "Red", "Blue", "Gray", etc.

        # Build rgb output safely
        if frame_resized.ndim == 2:
            # grayscale -> create channels from gray
            if color_norm == "Red":
                rgb = np.zeros((frame_resized.shape[0], frame_resized.shape[1], 3), dtype=np.uint8)
                rgb[..., 0] = frame_resized
            elif color_norm == "Green":
                rgb = np.zeros((frame_resized.shape[0], frame_resized.shape[1], 3), dtype=np.uint8)
                rgb[..., 1] = frame_resized
            elif color_norm == "Blue":
                rgb = np.zeros((frame_resized.shape[0], frame_resized.shape[1], 3), dtype=np.uint8)
                rgb[..., 2] = frame_resized
            else:
                rgb = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2RGB)
        else:
            # color image (H,W,3 or H,W,>=3). Use appropriate channels
            if frame_resized.shape[2] >= 3:
                if color_norm == "Red":
                    rgb = np.zeros_like(frame_resized[..., :3])
                    rgb[..., 0] = frame_resized[..., 0]
                elif color_norm == "Green":
                    rgb = np.zeros_like(frame_resized[..., :3])
                    rgb[..., 1] = frame_resized[..., 1]
                elif color_norm == "Blue":
                    rgb = np.zeros_like(frame_resized[..., :3])
                    rgb[..., 2] = frame_resized[..., 2]
                else:
                    rgb = frame_resized[..., :3]
            else:
                # fallback: convert to RGB
                rgb = cv2.cvtColor(frame_resized[..., 0], cv2.COLOR_GRAY2RGB) if frame_resized.ndim == 3 else cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2RGB)

        rgb = ensure_uint8_rgb(rgb)

        # --- Update canvas and ensure the FigureCanvas widget reports the pixel size so scrollbars appear ---
        # Pass a Label that matches the color (title-case) so show_frame can detect true_colour
        self.canvas.show_frame(rgb, color_norm, original_color=rgb)

        # Set the canvas minimum size to the image pixel dimensions so scroll area can scroll vertically only
        h_img, w_img = rgb.shape[:2]
        # Force canvas width to viewport width (avoid horizontal scroll)
        try:
            viewport_w = int(self.scroll_area.viewport().width())
            if viewport_w > 1:
                w_img = viewport_w
        except:
            pass

        self.canvas.setMinimumSize(w_img, h_img)
        self.canvas.resize(w_img, h_img)

        # Force a draw to update the widget size / scrollbars
        self.canvas.draw_idle()


if __name__ == "__main__":
    # Change this path to your TIFF file before running
    tiff_path = r"D:\python-master\2024.08.01_cLift-Kontrolle_A0206-02_Rl50Gh35_1677.tif"

    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    width = screen.size().width()
    height = screen.size().height()
    dpi = screen.logicalDotsPerInch()

    window = MainWindow(tiff_path, width, height, dpi)
    window.showMaximized()
    sys.exit(app.exec_())
