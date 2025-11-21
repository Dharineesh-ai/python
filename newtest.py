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
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal, QSize, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPainterPath, QPen, QColor, QBrush

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


class MagnifierOverlay(QLabel):
    """
    Circular magnifier overlay (follows cursor).
    Not draggable. Has white border and circular mask.
    Parent should be QScrollArea.viewport() so it remains visible while scrolling.
    """
    def __init__(self, parent=None, size=220, border=4):
        super().__init__(parent)
        self.setWindowFlags(Qt.Widget)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setScaledContents(False)
        self.overlay_size = int(size)
        self.border_px = int(border)
        self.resize(self.overlay_size, self.overlay_size)
        self.setVisible(False)
        self._last_pixmap = None

    def update_image_from_ndarray(self, arr):
        """
        arr: uint8 RGB numpy array scaled to overlay_size x overlay_size (H,W,3)
        Convert to QPixmap and draw circular clipped image + white border.
        """
        if arr is None or arr.size == 0:
            return
        arr_rgb = ensure_uint8_rgb(arr)
        if arr_rgb is None:
            return

        h, w = arr_rgb.shape[:2]
        qimg = QImage(arr_rgb.data.tobytes(), w, h, arr_rgb.strides[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.overlay_size, self.overlay_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

        out = QPixmap(self.overlay_size, self.overlay_size)
        out.fill(QColor(0, 0, 0, 0))
        painter = QPainter(out)
        painter.setRenderHint(QPainter.Antialiasing, True)

        path = QPainterPath()
        path.addEllipse(0, 0, self.overlay_size, self.overlay_size)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pix)

        painter.setClipping(False)
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(self.border_px)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(self.border_px//2, self.border_px//2, self.overlay_size - self.border_px, self.overlay_size - self.border_px)

        painter.end()

        self._last_pixmap = out
        self.setPixmap(out)
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
        self.original_img = None        # full-resolution original RGB (for magnifier + preview)
        self.overview_img = None
        self.overview_rect = None
        self.mag_patch = None
        self.img_shape = None
        self.current_cmap = "gray"
        self.over_ax = None

        # We'll create the magnifier later once we have a parent (parent is set after widget added to scroll area)
        self.overlay = None

        # Static preview label (top-right). Will show full original image and current view rect.
        self.preview_w = 240
        self.preview_h = 180
        self.preview_label = QLabel(parent=self)
        self.preview_label.setFixedSize(self.preview_w, self.preview_h)
        self.preview_label.setStyleSheet("QLabel { border: 2px solid white; background: black; }")
        self.preview_label.setVisible(False)
        self.preview_label.setScaledContents(False)
        self._position_preview()

        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def _ensure_overlay_created(self):
        """
        Create overlay with the appropriate parent (scrollarea.viewport()) so it remains visible
        while scrolling. If no parent available, fall back to canvas itself.
        """
        if self.overlay is not None:
            return
        # Prefer parent() (which should be the QScrollArea.viewport() when setWidget() is used)
        overlay_parent = self.parent() if self.parent() is not None else self
        self.overlay = MagnifierOverlay(parent=overlay_parent, size=220, border=4)
        # position initially centered top-right-ish
        try:
            pw = overlay_parent.width()
            ph = overlay_parent.height()
            ow = self.overlay.width()
            oh = self.overlay.height()
            x = max(10, pw - ow - 10)
            y = 10
            self.overlay.move(x, y)
        except Exception:
            pass

    def _position_preview(self):
        try:
            pw, ph = self.width(), self.height()
            x = max(10, pw - self.preview_w - 10)
            y = 10
            self.preview_label.move(x, y)
            self.preview_label.raise_()
        except:
            pass

    def on_resize(self, event):
        try:
            w, h = self.get_width_height()
            dpi = self.fig.dpi
            self.fig.set_size_inches(w / dpi, h / dpi)
        except:
            pass
        try:
            if self.overlay:
                parent_widget = self.overlay.parent()
                if parent_widget is not None:
                    ow, oh = self.overlay.width(), self.overlay.height()
                    # ensure overlay inside parent bounds
                    x = min(self.overlay.x(), max(0, parent_widget.width() - ow))
                    y = min(self.overlay.y(), max(0, parent_widget.height() - oh))
                    self.overlay.move(x, y)
            self._position_preview()
        except:
            pass

    def show_frame(self, display_img, label="", original_color=None):
        """
        display_img: image sized to viewport (uint8 RGB or gray)
        original_color: full-resolution RGB image (uint8) OR None
        """
        # ------------------------------------------------------------------
        # 1. Normalize to uint8 if needed (display_img already expected uint8)
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
        # ------------------------------------------------------------------
        label_norm = (label or "").strip().lower()
        true_colour = label_norm in ("red", "green", "blue")

        # ------------------------------------------------------------------
        # 3. First draw – create the imshow object
        # ------------------------------------------------------------------
        if self.main_img is None:
            self.ax.clear()
            if true_colour and disp8.ndim == 3:
                self.main_img = self.ax.imshow(disp8, interpolation="nearest", animated=True)
            else:
                gray = disp8 if disp8.ndim == 2 else cv2.cvtColor(disp8, cv2.COLOR_RGB2GRAY)
                self.main_img = self.ax.imshow(gray, cmap="gray", interpolation="nearest", animated=True)
            self.ax.axis("off")
        else:
            if true_colour and disp8.ndim == 3:
                self.main_img.set_data(disp8)
                self.main_img.set_cmap(None)
            else:
                gray = disp8 if disp8.ndim == 2 else cv2.cvtColor(disp8, cv2.COLOR_RGB2GRAY)
                self.main_img.set_data(gray)
                self.main_img.set_cmap("gray")

        # ------------------------------------------------------------------
        # 5. Title + original colour for magnifier / preview
        # ------------------------------------------------------------------
        self.ax.set_title(label.title() if label else "")
        self.img_shape = disp8.shape[:2]

        if original_color is not None:
            self.original_img = ensure_uint8_rgb(original_color)
            # ensure overlay exists and preview updated
            self._ensure_overlay_created()
            self._update_preview()
        else:
            # keep existing
            self._ensure_overlay_created()

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
        """
        Called frequently — build magnifier crop from full-resolution original_img if present.
        Also update preview rectangle.

        Fixed behavior:
        - magnifier centers at cursor (Option A)
        - overlay parent is scrollarea.viewport() so magnifier stays visible while scrolling
        - mapping between display coordinates and original image is correct so magnifier follows same direction
        """
        if not event.inaxes or not self.main_img:
            return
        # Ensure overlay exists
        self._ensure_overlay_created()

        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        # Displayed image array and its size
        img = self.main_img.get_array()
        h_disp, w_disp = img.shape[:2]

        # small region around cursor in display/image coordinates
        region = 60
        xi, yi = int(round(xdata)), int(round(ydata))
        l = max(0, xi - region)
        r = min(w_disp, xi + region)
        t = max(0, yi - region)
        b = min(h_disp, yi + region)

        # Crop from full-resolution original if available; scale correctly
        if getattr(self, "original_img", None) is not None:
            orig = self.original_img
            h_orig, w_orig = orig.shape[:2]
            # Compute scale between displayed image and original image
            scale_x = (w_orig / float(w_disp)) if w_disp > 0 else 1.0
            scale_y = (h_orig / float(h_disp)) if h_disp > 0 else 1.0
            l_o = max(0, int(round(l * scale_x)))
            r_o = min(w_orig, int(round(r * scale_x)))
            t_o = max(0, int(round(t * scale_y)))
            b_o = min(h_orig, int(round(b * scale_y)))
            crop = orig[t_o:b_o, l_o:r_o]
        else:
            crop = img[t:b, l:r]
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

        crop = ensure_uint8_rgb(crop)
        if crop is None or crop.size == 0:
            return

        # Upscale to overlay size for smooth magnification
        zoomed = cv2.resize(crop, (self.overlay.overlay_size, self.overlay.overlay_size), interpolation=cv2.INTER_CUBIC)
        self.overlay.update_image_from_ndarray(zoomed)

        # Position overlay centered at cursor, using proper mapping to overlay parent coordinates
        try:
            # event.x, event.y are widget display coordinates relative to the canvas widget
            canvas_x = int(round(event.x))
            canvas_y = int(round(event.y))

            # Map canvas coordinates to overlay parent coordinates (usually scrollarea.viewport())
            parent_widget = self.overlay.parent()
            if parent_widget is not None and parent_widget is not self:
                # mapToParent will transform from canvas coords to parent's coords
                mapped_point = self.mapToParent(QPoint(canvas_x, canvas_y))
                px = mapped_point.x() - (self.overlay.width() // 2)
                py = mapped_point.y() - (self.overlay.height() // 2)
                # clamp to parent bounds
                px = max(0, min(px, parent_widget.width() - self.overlay.width()))
                py = max(0, min(py, parent_widget.height() - self.overlay.height()))
                self.overlay.move(px, py)
            else:
                # fallback: parent is canvas itself
                px = canvas_x - (self.overlay.width() // 2)
                py = canvas_y - (self.overlay.height() // 2)
                px = max(0, min(px, self.width() - self.overlay.width()))
                py = max(0, min(py, self.height() - self.overlay.height()))
                self.overlay.move(px, py)
        except Exception:
            pass

        # Draw rectangle patch on the main axes indicating the magnified region (display coords)
        if self.mag_patch is None:
            self.mag_patch = patches.Rectangle((l, t), r - l, b - t, facecolor='none', edgecolor='yellow', linewidth=1)
            self.ax.add_patch(self.mag_patch)
        else:
            self.mag_patch.set_bounds(l, t, r - l, b - t)
            self.mag_patch.set_visible(True)

        # Update preview overlay rectangle and repaint preview
        self._update_preview(draw_rect=True)

        # Redraw canvas (only the figure)
        self.draw_idle()

    def update_overview_rect(self, event=None):
        # compatibility stub; main preview logic handled in _update_preview
        return

    def _update_preview(self, draw_rect=True):
        """
        Build preview pixmap from self.original_img (full-resolution).
        Draw a rectangle representing the current viewport (if possible).
        """
        if getattr(self, "original_img", None) is None:
            self.preview_label.setVisible(False)
            return

        orig = self.original_img
        h_orig, w_orig = orig.shape[:2]
        qimg = QImage(orig.data.tobytes(), w_orig, h_orig, orig.strides[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.preview_w, self.preview_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        out = QPixmap(self.preview_w, self.preview_h)
        out.fill(QColor(0, 0, 0))
        painter = QPainter(out)
        painter.setRenderHint(QPainter.Antialiasing, True)

        pix_x = (self.preview_w - pix.width()) // 2
        pix_y = (self.preview_h - pix.height()) // 2
        painter.drawPixmap(pix_x, pix_y, pix)

        if draw_rect and self.img_shape is not None:
            try:
                x0, x1 = self.ax.get_xlim()
                y0, y1 = self.ax.get_ylim()
                # Convert to top-left origin coords
                left = int(round(min(x0, x1)))
                right = int(round(max(x0, x1)))
                top = int(round(min(y0, y1)))
                bottom = int(round(max(y0, y1)))
                view_w = right - left
                view_h = bottom - top

                if view_w > 0 and view_h > 0:
                    scale_x = pix.width() / float(w_orig)
                    scale_y = pix.height() / float(h_orig)
                    rect_x = pix_x + int(round(left * scale_x))
                    rect_y = pix_y + int(round(top * scale_y))
                    rect_w = max(1, int(round(view_w * scale_x)))
                    rect_h = max(1, int(round(view_h * scale_y)))
                    pen = QPen(QColor(255, 255, 0))
                    pen.setWidth(2)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(rect_x, rect_y, rect_w, rect_h)
            except Exception:
                pass

        painter.end()
        self.preview_label.setPixmap(out)
        self.preview_label.setVisible(True)
        self.preview_label.raise_()
        self._position_preview()


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

        # Keep a full-resolution copy in RGB (for preview + magnifier)
        h_orig, w_orig = frame.shape[:2]
        color = self.selected_colors[index] if index < len(self.selected_colors) else "Gray"
        color_norm = (color or "").strip().title()

        # Build rgb_full from original 'frame' (not resized)
        if frame.ndim == 2:
            if color_norm == "Red":
                rgb_full = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
                rgb_full[..., 0] = frame
            elif color_norm == "Green":
                rgb_full = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
                rgb_full[..., 1] = frame
            elif color_norm == "Blue":
                rgb_full = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
                rgb_full[..., 2] = frame
            else:
                rgb_full = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            if frame.shape[2] >= 3:
                if color_norm == "Red":
                    rgb_full = np.zeros_like(frame[..., :3])
                    rgb_full[..., 0] = frame[..., 0]
                elif color_norm == "Green":
                    rgb_full = np.zeros_like(frame[..., :3])
                    rgb_full[..., 1] = frame[..., 1]
                elif color_norm == "Blue":
                    rgb_full = np.zeros_like(frame[..., :3])
                    rgb_full[..., 2] = frame[..., 2]
                else:
                    rgb_full = frame[..., :3].copy()
            else:
                rgb_full = cv2.cvtColor(frame[..., 0], cv2.COLOR_GRAY2RGB) if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        rgb_full = ensure_uint8_rgb(rgb_full)

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

        h, w = rgb_full.shape[:2]
        new_w = max(1, int(viewport_w))
        new_h = max(1, int(h * (new_w / float(w))))
        frame_resized_rgb = cv2.resize(rgb_full, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rgb = ensure_uint8_rgb(frame_resized_rgb)

        # --- Update canvas and ensure the FigureCanvas widget reports the pixel size so scrollbars appear ---
        # Pass original full-resolution rgb_full so magnifier and preview use the original data
        self.canvas.show_frame(rgb, color_norm, original_color=rgb_full)

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
