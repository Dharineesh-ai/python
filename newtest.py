# newtest.py - FIXED TIFF VIEWER WITH TOP-LEFT 5x5 GRID + PERCENTILE B&C + BLOBS
# + LIVE CURSOR COORD DISPLAY (TOP-RIGHT)

import sys
import time
from datetime import datetime

from PyQt5.QtWidgets import (
    QMessageBox, QMenuBar, QMenu, QAction, QVBoxLayout,
    QHBoxLayout, QWidget, QLabel, QScrollBar, QApplication,
    QMainWindow, QScrollArea, QListWidgetItem, QPushButton,
    QListWidget, QDialog, QProgressBar, QSizePolicy, QFileDialog
)

import os
import psutil
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage import io

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

from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal, QSize, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPainterPath, QPen, QColor, QBrush

# Skip GUI + TIFF loading when running in GitHub Actions
if os.getenv("CI") == "true" and __name__ == "__main__":
    print("Running in CI mode — skipping TIFF loading and GUI.")
    sys.exit(0)

# ---------------- Utility functions ----------------

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
        rgb = np.stack([img8]*3, axis=2)
        return auto_fix_bgr_rgb(rgb)
    elif img8.ndim == 3:
        ch = img8.shape[2]
        if ch == 1:
            rgb = np.stack([img8[:, :, 0]]*3, axis=2)
            return auto_fix_bgr_rgb(rgb)
        if ch == 3:
            if np.array_equal(img8[:, :, 0], img8[:, :, 1]) and np.array_equal(img8[:, :, 1], img8[:, :, 2]):
                rgb = np.stack([img8[:, :, 0]]*3, axis=2)
                return auto_fix_bgr_rgb(rgb)
            return auto_fix_bgr_rgb(img8)
        if ch == 4:
            try:
                candidate = img8[:, :, :3].copy()
                if (np.array_equal(candidate[:, :, 0], candidate[:, :, 1]) and
                        np.array_equal(candidate[:, :, 1], candidate[:, :, 2])):
                    return auto_fix_bgr_rgb(img8[:, :, :3])
                return auto_fix_bgr_rgb(candidate)
            except Exception:
                return auto_fix_bgr_rgb(img8[:, :, :3])
    return img8

def compute_percentile_range(img, vmin_pct=3.0, vmax_pct=97.0):
    if img is None or img.size == 0:
        return 0.0, 255.0
    flat = img.flatten()
    vmin = np.percentile(flat, vmin_pct)
    vmax = np.percentile(flat, vmax_pct)
    return vmin, vmax

# ---------------- Top-left blob detection ----------------

def detect_log_blobs_topleft(frame, roi_size=100):
    """
    ULTRA-FAST blob detection in top-left region only
    Blue circles for blank spots (5-25px radius)
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame.astype(np.uint8)

    h, w = gray.shape
    roi_w = min(roi_size * 10, w)
    roi_h = min(roi_size * 10, h)
    roi = gray[:roi_h, :roi_w]

    _, thresh = cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blob_centers = []
    blob_radii = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 80 < area < 400:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy, radius = int(cx), int(cy), int(radius)
            if 6 <= radius <= 20:
                blob_centers.append((cx, cy))
                blob_radii.append(radius)

    print(f"✓ Top-left detection: Found {len(blob_centers)} blank spots")
    return blob_centers, blob_radii

# ---------------- Color selector dialog ----------------
class GALSpot:
    def __init__(self, spot_id, x, y, radius, is_blank):
        self.spot_id = spot_id
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.is_blank = bool(is_blank)


class GALDataModel:
    """
    Parse your GAL.csv (GenePix style) and expose 25 spots.
    Uses 'ID' column: if it contains 'Empty' → blank, else filled.
    """
    def __init__(self):
        self.spots = []

    def clear(self):
        self.spots = []

    def load_from_file(self, path, default_radius=3.0):
        import csv
        self.spots = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            in_table = False
            for row in reader:
                if not row:
                    continue
                # Find header row
                if not in_table:
                    if len(row) >= 6 and row[0].strip() == "Block" and row[5].strip().startswith("x"):
                        in_table = True
                    continue
                # Data rows: Block,Row,Column,ID,Name,"x,y"
                if len(row) < 6:
                    continue
                block, r, c, spot_id, name, xy = row[:6]
                xy = xy.strip().strip('"')
                if "," not in xy:
                    continue
                x_str, y_str = [p.strip() for p in xy.split(",", 1)]
                x = float(x_str)
                y = float(y_str)
                # blank if ID contains 'Empty'
                is_blank = "empty" in (spot_id or "").lower()
                self.spots.append(GALSpot(spot_id, x, y, default_radius, is_blank))

    def __len__(self):
        return len(self.spots)

    def __iter__(self):
        return iter(self.spots)

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
        return [self.list_widget.item(i).text().strip()
                for i in range(self.list_widget.count())]

# ---------------- Magnifier overlay ----------------

class MagnifierOverlay(QLabel):
    def __init__(self, parent=None, size=260, border=4):
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
        if arr is None or arr.size == 0:
            return
        arr_rgb = ensure_uint8_rgb(arr)
        if arr_rgb is None:
            return

        h, w = arr_rgb.shape[:2]
        qimg = QImage(
            arr_rgb.data.tobytes(), w, h,
            arr_rgb.strides[0], QImage.Format_RGB888
        )
        pix = QPixmap.fromImage(qimg).scaled(
            self.overlay_size, self.overlay_size,
            Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        )

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
        painter.drawEllipse(
            self.border_px // 2,
            self.border_px // 2,
            self.overlay_size - self.border_px,
            self.overlay_size - self.border_px
        )
        painter.end()

        self._last_pixmap = out
        self.setPixmap(out)
        self.setVisible(True)
        self.raise_()

# ---------------- Matplotlib canvas ----------------

class MplCanvas(FigureCanvas):
    def __init__(self, width, height, dpi, parent_window=None):
        fig_width_in = width / dpi
        fig_height_in = height / dpi
        self.fig = Figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        super().__init__(self.fig)

        # Keep reference to MainWindow (for coord label updates)
        self.parent_window = parent_window

        self.mpl_connect("draw_event", self.on_draw_event)
        self.mpl_connect("motion_notify_event", self.on_mouse_motion)
        self.mpl_connect("button_release_event", self.update_overview_rect)
        self.mpl_connect("key_press_event", self.on_key_press)

        self.draw_start = time.time()
        self.red_cmap = LinearSegmentedColormap.from_list(
            "red_map", [(0, "black"), (1, "red")]
        )
        self.green_cmap = LinearSegmentedColormap.from_list(
            "green_map", [(0, "black"), (1, "green")]
        )

        self.main_img = None
        self.original_img = None
        self.overview_img = None
        self.overview_rect = None
        self.mag_patch = None
        self.img_shape = None
        self.current_cmap = "gray"
        self.over_ax = None
        self.overlay = None

        # Blob detection - few circles only
        self.blob_circles = []
        self.show_blobs = True

        # Grid display toggle
        #self.show_grid = True

        # Static preview label
        self.preview_w = 240
        self.preview_label = QLabel(parent=None)
        self.preview_label.setFixedWidth(self.preview_w)
        self.preview_label.setStyleSheet(
            "QLabel { border: 2px solid white; background: black; }"
        )
        self.preview_label.setVisible(False)
        self.preview_label.setScaledContents(False)
        self._position_preview()
        self.preview_img_w = 0
        self.preview_img_h = 0

        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.last_view_rect = None

    # ----- helper for magnifier -----

    def _ensure_overlay_created(self):
        if self.overlay is not None:
            return
        overlay_parent = self.parent() if self.parent() is not None else self
        self.overlay = MagnifierOverlay(parent=overlay_parent, size=220, border=4)
        try:
            pw = overlay_parent.width()
            ph = overlay_parent.height()
            ow = self.overlay.width()
            oh = self.overlay.height()
            x = max(10, pw - ow - 10)
            y = max(10, min(100, ph - oh - 10))
            self.overlay.move(x, y)
        except AttributeError:
            pass

    def _position_preview(self):
        try:
            parent_widget = self.parent()
            if parent_widget is not None and self.preview_label.parent() is not parent_widget:
                self.preview_label.setParent(parent_widget)
                self.preview_label.setFixedHeight(parent_widget.height())
                self.preview_label.setFixedWidth(self.preview_w)
            if self.preview_label.parent() is not None:
                p = self.preview_label.parent()
                self.preview_label.setFixedHeight(p.height())
                x = max(5, p.width() - self.preview_w - 5)
                y = 0
                self.preview_label.move(x, y)
                self.preview_label.raise_()
        except AttributeError:
            pass

    def clear_blob_circles(self):
        self.blob_circles = []
        for patch in list(self.ax.patches):
            if hasattr(patch, '_blob_marker'):
                patch.remove()
        self.draw_idle()

    # ----- show frame -----

    def show_frame(self, display_img, label="", original_color=None):
        """
        Shows the image + overlays:
        - optional detected blank-spot blue circles
        - 5x5 grid overlay (white for filled, blue for blanks)
        """
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

        label_norm = (label or "").strip().lower()
        true_colour = label_norm in ("red", "green", "blue")

        self.ax.clear()
        if true_colour and disp8.ndim == 3:
            self.main_img = self.ax.imshow(
                disp8, interpolation="nearest", animated=True
            )
        else:
            gray = disp8 if disp8.ndim == 2 else cv2.cvtColor(
                disp8, cv2.COLOR_RGB2GRAY
            )
            self.main_img = self.ax.imshow(
                gray, cmap="gray", interpolation="nearest", animated=True
            )

        self.ax.axis("off")
        self.ax.set_title(label.title() if label else "")
        self.img_shape = disp8.shape[:2]

        if original_color is not None:
            self.original_img = ensure_uint8_rgb(original_color)
            self._ensure_overlay_created()
            self._update_preview()
        else:
            self._ensure_overlay_created()

        try:
            self.ax.set_xlim(0, self.img_shape[1])
            self.ax.set_ylim(self.img_shape[0], 0)
            self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        except AttributeError:
            pass

        # Draw blob circles
        if self.blob_circles and self.show_blobs:
            for (cx, cy), radius in self.blob_circles:
                circle = Circle(
                    (cx, cy), radius,
                    edgecolor='blue', facecolor='none',
                    linewidth=0.7, alpha=0.9
                )
                circle._blob_marker = True
                self.ax.add_patch(circle)

        

        self.draw_start = time.time()
        self.draw_idle()

    # ----- events -----

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
        elif event.key == 'b':
            self.show_blobs = not self.show_blobs
            self.draw_idle()
        elif event.key == 'g':
            self.show_grid = not self.show_grid
            self.draw_idle()

    def zoom_in(self, factor=0.7):
        if not self.img_shape:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w = (x1 - x0) * factor
        h = (y1 - y0) * factor
        self.ax.set_xlim(cx - w/2, cx + w/2)
        self.ax.set_ylim(cy + h/2, cy - h/2)
        self.update_overview_rect()
        self._update_preview(draw_rect=True)
        self.draw_idle()

    def zoom_out(self, factor=1.4):
        if not self.img_shape:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w = min((x1 - x0) * factor, self.img_shape[1])
        h = min((y1 - y0) * factor, self.img_shape[0])
        self.ax.set_xlim(max(0, cx - w/2), min(self.img_shape[1], cx + w/2))
        self.ax.set_ylim(min(self.img_shape[0], cy + h/2), max(0, cy - h/2))
        self.update_overview_rect()
        self._update_preview(draw_rect=True)
        self.draw_idle()

    def on_mouse_motion(self, event):
        # Update magnifier & preview as before
        if not event.inaxes or not self.main_img:
            return

        # Also update live coordinate label in MainWindow (image coordinates)
        if self.parent_window is not None and event.xdata is not None and event.ydata is not None:
            self.parent_window.update_coords(int(round(event.xdata)), int(round(event.ydata)))

        self._ensure_overlay_created()
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        img = self.main_img.get_array()
        h_disp, w_disp = img.shape[:2]
        region = 20
        xi, yi = int(round(xdata)), int(round(ydata))
        l = max(0, xi - region)
        r = min(w_disp, xi + region)
        t = max(0, yi - region)
        b = min(h_disp, yi + region)

        if getattr(self, "original_img", None) is not None:
            orig = self.original_img
            h_orig, w_orig = orig.shape[:2]
            scale_x = (w_orig / float(w_disp)) if w_disp > 0 else 1.0
            scale_y = (h_orig / float(h_disp)) if h_disp > 0 else 1.0
            l_o = max(0, int(round(l * scale_x)))
            r_o = min(w_orig, int(round(r * scale_x)))
            t_o = max(0, int(round(t * scale_y)))
            b_o = min(h_orig, int(round(b * scale_y)))
            self.last_view_rect = (l_o, t_o, r_o, b_o)
            crop = orig[t_o:b_o, l_o:r_o]
        else:
            crop = img[t:b, l:r]

        if crop.ndim == 2:
            crop = np.stack([crop]*3, axis=2)
        crop = ensure_uint8_rgb(crop)
        if crop is None or crop.size == 0:
            return

        target_size = self.overlay.overlay_size * 2
        zoomed = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        self.overlay.update_image_from_ndarray(zoomed)

        try:
            canvas_x = int(round(event.x))
            canvas_y = int(round(event.y))
            parent_widget = self.overlay.parent()
            if parent_widget is not None and parent_widget is not self:
                mapped_point = self.mapToParent(QPoint(canvas_x, canvas_y))
                px = mapped_point.x() - (self.overlay.width() // 2)
                py = mapped_point.y() - (self.overlay.height() // 2)
                px = max(0, min(px, parent_widget.width() - self.overlay.width()))
                py = max(0, min(py, parent_widget.height() - self.overlay.height()))
                self.overlay.move(px, py)
            else:
                px = canvas_x - (self.overlay.width() // 2)
                py = canvas_y - (self.overlay.height() // 2)
                px = max(0, min(px, self.width() - self.overlay.width()))
                py = max(0, min(py, self.height() - self.overlay.height()))
                self.overlay.move(px, py)
        except AttributeError:
            pass

        if self.mag_patch is None:
            self.mag_patch = patches.Rectangle(
                (l, t), r - l, b - t,
                facecolor='none', edgecolor='yellow', linewidth=1
            )
            self.ax.add_patch(self.mag_patch)
        else:
            self.mag_patch.set_bounds(l, t, r - l, b - t)
        self.mag_patch.set_visible(True)

        self._update_preview(draw_rect=True)
        self.draw_idle()

    def update_overview_rect(self, event=None):
        return

    def _update_preview(self, draw_rect=True):
        if getattr(self, "original_img", None) is None:
            self.preview_label.setVisible(False)
            return

        orig = self.original_img
        h_orig, w_orig = orig.shape[:2]
        qimg = QImage(
            orig.data.tobytes(), w_orig, h_orig,
            orig.strides[0], QImage.Format_RGB888
        )

        parent_widget = self.preview_label.parent()
        if parent_widget is None:
            if self.parent() is not None:
                parent_widget = self.parent()
                self.preview_label.setParent(parent_widget)

        preview_h = self.preview_label.height() if self.preview_label.height() > 1 else 180
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview_w, preview_h,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_img_w = pix.width()
        self.preview_img_h = pix.height()

        out = QPixmap(self.preview_w, preview_h)
        out.fill(QColor(0, 0, 0))

        painter = QPainter(out)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pix_x = (self.preview_w - pix.width()) // 2
        pix_y = (preview_h - pix.height()) // 2
        painter.drawPixmap(pix_x, pix_y, pix)

        if draw_rect and self.img_shape is not None:
            try:
                if self.last_view_rect is not None:
                    left, top, right, bottom = self.last_view_rect
                else:
                    x0, x1 = self.ax.get_xlim()
                    y0, y1 = self.ax.get_ylim()
                    left = int(round(min(x0, x1)))
                    right = int(round(max(x0, x1)))
                    top = int(round(min(y0, y1)))
                    bottom = int(round(max(y0, y1)))
                view_w = right - left
                view_h = bottom - top
                if view_w > 0 and view_h > 0:
                    scale_x = pix.width() / float(w_orig)
                    scale_y = pix.height() / float(h_orig)
                    rect_w = max(1, int(round(view_w * scale_x)))
                    rect_h = max(1, int(round(view_h * scale_y)))
                    rect_x = pix_x + int(round(left * scale_x))
                    rect_y = pix_y + int(round(top * scale_y))
                    pen = QPen(QColor(255, 255, 0))
                    pen.setWidth(2)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(rect_x, rect_y, rect_w, rect_h)
            except AttributeError:
                pass

        painter.end()
        self.preview_label.setPixmap(out)
        self.preview_label.setVisible(True)
        self.preview_label.raise_()
        self._position_preview()

# ---------------- TIFF loader thread ----------------

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
            except Exception:
                stacked = frames
        self.finished.emit(stacked, num_pages)

# ---------------- Main window ----------------

class MainWindow(QMainWindow):
    
    def __init__(self, tiff_path, width, height, dpi):
        super().__init__()
        self.setWindowTitle("TIFF Viewer — Top-Left 5x5 Grid")

        self.gal_model = GALDataModel()
        
        self.brightness = 0.0
        self.contrast = 0.0
        self.percentile_vmin = 3.0
        self.percentile_vmax = 97.0
        self.use_percentile = True

        # Blob detection state
        self.blob_centers = []
        self.blob_radii = []

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(20)

        self._first_draw_done = False
        self._data_ready = False

        scroll_area = QScrollArea()
        # Pass self to canvas so it can update coord label
        self.canvas = MplCanvas(width, height, dpi, parent_window=self)
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area = scroll_area

        self.width = width
        self.height = height
        self.dpi = dpi

        self._create_menu_bar()
        

        layout = QVBoxLayout()

        # -------- Coordinates label (top-right) --------
        self.coord_label = QLabel("X: -, Y: -")
        self.coord_label.setStyleSheet(
            """
            QLabel {
                background: rgba(0, 0, 0, 200);
                color: white;
                padding: 3px 8px;
                font-family: monospace;
                font-size: 11px;
                border-radius: 4px;
                min-width: 110px;
            }
            """
        )
        self.coord_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.coord_label)

        layout.addWidget(scroll_area)

        bc_layout = QHBoxLayout()
        self.brightness_slider = QScrollBar(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setFixedHeight(16)
        self.brightness_slider.valueChanged.connect(self.set_brightness)
        bc_layout.addWidget(QLabel("B:"))
        bc_layout.addWidget(self.brightness_slider)

        self.contrast_slider = QScrollBar(Qt.Horizontal)
        self.contrast_slider.setMinimum(-100)
        self.contrast_slider.setMaximum(100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.setFixedHeight(16)
        self.contrast_slider.valueChanged.connect(self.set_contrast)
        bc_layout.addWidget(QLabel("C:"))
        bc_layout.addWidget(self.contrast_slider)

        self.bc_container = QWidget()
        self.bc_container.setLayout(bc_layout)
        self.bc_container.setVisible(False)
        layout.addWidget(self.bc_container)

        self.scroll_bar = QScrollBar(Qt.Horizontal)
        self.scroll_bar.setStyleSheet(
            "QScrollBar::handle:horizontal { background: #555; "
            "min-width: 20px; border-radius: 8px; }"
        )
        self.scroll_bar.valueChanged.connect(self.display_frame)
        layout.addWidget(self.scroll_bar)
        layout.addWidget(self.progress_bar)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.preload = self.should_preload(tiff_path)
        self.load_tiff(tiff_path)


    def apply_gal_to_canvas(self):
        if not self.gal_model or len(self.gal_model) == 0:
            return

        # push GAL spots to canvas
        spots = [((int(s.x), int(s.y)), int(s.radius)) for s in self.gal_model]
        self.canvas.blob_circles = spots
        self.canvas.show_blobs = True

        # redraw only if scrollbar is ready
        if hasattr(self, "scrollbar"):
            idx = self.scrollbar.value()
            self.display_frame(idx)
        else:
            # fallback: when first frame is eventually shown, circles are already set
            self.canvas.draw_idle()


    # ------- coordinates API used by canvas -------

    def update_coords(self, x, y):
        # x,y are image coordinates from MplCanvas
        self.coord_label.setText(f"X: {x}, Y: {y}")
        self.coord_label.raise_()
        # reposition to always stay top-right
        self._position_coord_label()

    def _position_coord_label(self):
        # Move to top-right of the central widget
        cw = self.centralWidget()
        if cw is None:
            return
        cw_rect = cw.rect()
        x = cw_rect.width() - self.coord_label.width() - 8
        y = 4
        self.coord_label.move(x, y)

    # ------- menu bar & actions -------

    def _create_menu_bar(self):
        menubar = self.menuBar()

        gal_menu = menubar.addMenu("GAL")
        open_gal_action = QAction("Open GAL file...", self)
        open_gal_action.setShortcut("Ctrl+O")
        open_gal_action.triggered.connect(self.open_gal_file)
        gal_menu.addAction(open_gal_action)

        adjust_menu = menubar.addMenu("Adjust")
        toggle_sliders_action = QAction("Show B&C Sliders", self)
        toggle_sliders_action.setShortcut("Ctrl+B")
        toggle_sliders_action.setCheckable(True)
        toggle_sliders_action.toggled.connect(self.toggle_bc_sliders)
        adjust_menu.addAction(toggle_sliders_action)

        percentile_menu = adjust_menu.addMenu("Percentile Range")
        self.vmin_action = QAction(f"Vmin: {self.percentile_vmin}%", self)
        self.vmin_action.triggered.connect(lambda: self.change_percentile('vmin'))
        percentile_menu.addAction(self.vmin_action)

        self.vmax_action = QAction(f"Vmax: {self.percentile_vmax}%", self)
        self.vmax_action.triggered.connect(lambda: self.change_percentile('vmax'))
        percentile_menu.addAction(self.vmax_action)

        reset_action = QAction("Reset B&C", self)
        reset_action.setShortcut("Ctrl+R")
        reset_action.triggered.connect(self.reset_bc)
        adjust_menu.addAction(reset_action)

        # Blob Detection Menu
        blob_menu = menubar.addMenu("Blobs")
        detect_action = QAction("Detect Top-Left Blobs", self)
        detect_action.setShortcut("Ctrl+D")
        detect_action.triggered.connect(self.detect_blobs_current_frame)
        blob_menu.addAction(detect_action)

        clear_action = QAction("Clear Blobs", self)
        clear_action.setShortcut("Ctrl+E")
        clear_action.triggered.connect(self.clear_blobs)
        blob_menu.addAction(clear_action)

        toggle_action = QAction("Toggle Blobs", self)
        toggle_action.setShortcut("Ctrl+T")
        toggle_action.triggered.connect(self.toggle_blob_visibility)
        blob_menu.addAction(toggle_action)

    # ------- blob handlers -------

    def detect_blobs_current_frame(self):
        index = int(self.scroll_bar.value())
        if self.preload and hasattr(self, 'frames'):
            frame = self.frames[index]
        elif hasattr(self, 'tif'):
            frame = self.tif.pages[index].asarray()
        else:
            QMessageBox.warning(self, "Error", "No frame data available")
            return

        if frame.dtype != np.uint8:
            pmin, pmax = compute_percentile_range(
                frame, self.percentile_vmin, self.percentile_vmax
            )
            if pmax > pmin:
                frame = ((frame.astype(np.float32) - pmin) /
                         (pmax - pmin) * 255).astype(np.uint8)

        print(f"🔍 Detecting top-left blobs on frame {index}...")
        self.blob_centers, self.blob_radii = detect_log_blobs_topleft(frame)
        self.canvas.blob_circles = list(zip(self.blob_centers, self.blob_radii))
        self.canvas.show_blobs = True
        self.display_frame(index)
        print(f"✅ {len(self.blob_centers)} blue circles added (top-left only)")

    def clear_blobs(self):
        self.blob_centers = []
        self.blob_radii = []
        self.canvas.clear_blob_circles()
        print("🗑️ Blobs cleared")

    def toggle_blob_visibility(self):
        self.canvas.show_blobs = not self.canvas.show_blobs
        self.display_frame(self.scroll_bar.value())
        print(f"👁️ Blobs {'shown' if self.canvas.show_blobs else 'hidden'}")

    def open_gal_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open GAL/CSV file",
            "",
            "GAL/CSV files (*.gal *.csv *.txt);;All files (*.*)",
        )
        if not path:
            return

        try:
            self.gal_model.load_from_file(path, default_radius=3.0)
            print(f"Loaded {len(self.gal_model)} GAL spots from {path}")
            self.apply_gal_to_canvas()
        except Exception as e:
            QMessageBox.critical(self, "GAL Load Error", str(e))


    # ------- B&C and percentile -------

    def toggle_bc_sliders(self, checked):
        self.bc_container.setVisible(checked)

    def change_percentile(self, which):
        if which == 'vmin':
            new_val = 3.0 if self.percentile_vmin == 1.0 else 1.0 if self.percentile_vmin == 5.0 else 5.0
            self.percentile_vmin = new_val
            self.vmin_action.setText(f"Vmin: {new_val}%")
        else:
            new_val = 97.0 if self.percentile_vmax == 99.0 else 99.0 if self.percentile_vmax == 95.0 else 95.0
            self.percentile_vmax = new_val
            self.vmax_action.setText(f"Vmax: {new_val}%")
        print(f"Percentile: vmin={self.percentile_vmin}%, vmax={self.percentile_vmax}%")
        self.display_frame(self.scroll_bar.value())

    def reset_bc(self):
        self.brightness = 0.0
        self.contrast = 0.0
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.display_frame(self.scroll_bar.value())

    def set_brightness(self, value):
        self.brightness = float(value)
        self.display_frame(self.scroll_bar.value())

    def set_contrast(self, value):
        self.contrast = float(value)
        self.display_frame(self.scroll_bar.value())

    # ------- Qt events -------

    def showEvent(self, event):
        super().showEvent(event)
        if (not self._first_draw_done and
                getattr(self, "_data_ready", False) and
                hasattr(self, "scroll_bar")):
            self._first_draw_done = True
            self.display_frame(self.scroll_bar.value())
            self.show_axes_size_popup()

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
            self._data_ready = True
            sample = self.tif.pages[0].asarray()
            self.buffer = np.empty_like(sample)

            dialog = ColorSelector(self.frames_count, parent=self)
            self.selected_colors = (
                dialog.get_selected_colors()
                if dialog.exec_() else ["Gray"] * self.frames_count
            )
            self.selected_colors = [c.strip().title() for c in self.selected_colors]
            self.scroll_bar.setMaximum(max(0, self.frames_count - 1))
            self.scroll_bar.setValue(0)

    def on_tiff_loaded(self, frames, count):
        print("TIFF fully loaded into RAM")
        self.frames = frames
        self.frames_count = count
        self._data_ready = True
        self.progress_bar.setVisible(False)

        dialog = ColorSelector(self.frames_count, parent=self)
        self.selected_colors = (
            dialog.get_selected_colors()
            if dialog.exec_() else ["Gray"] * self.frames_count
        )
        self.selected_colors = [c.strip().title() for c in self.selected_colors]
        self.scroll_bar.setMaximum(max(0, self.frames_count - 1))
        self.scroll_bar.setValue(0)

        if self.isVisible() and not self._first_draw_done:
            self._first_draw_done = True
            self.display_frame(self.scroll_bar.value())
            self.show_axes_size_popup()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_coord_label()
        if (hasattr(self, 'scroll_bar') and hasattr(self, 'frames_count') and
                self.scroll_bar.value() >= 0 and
                (hasattr(self, 'frames') or hasattr(self, 'tif'))):
            self.display_frame(self.scroll_bar.value())

    def closeEvent(self, event):
        if hasattr(self, 'tif') and self.tif:
            self.tif.close()
        event.accept()

    # ------- frame display -------

    def display_frame(self, index=None):
        try:
            viewport_w = int(self.scroll_area.viewport().width())
        except AttributeError:
            viewport_w = self.width

        preview_w = getattr(self.canvas, "preview_w", 240)
        effective_w = max(1, viewport_w - preview_w - 8)
        if effective_w <= 200:
            effective_w = max(1, self.width - preview_w - 8)

        if index is None:
            index = int(self.scroll_bar.value())

        if self.preload:
            frame = self.frames[index]
        else:
            frame = self.tif.pages[index].asarray(out=self.buffer)

        h_orig, w_orig = frame.shape[:2]

        if frame.dtype != np.uint8:
            pmin, pmax = compute_percentile_range(
                frame, self.percentile_vmin, self.percentile_vmax
            )
            if pmax > pmin:
                frame = ((frame.astype(np.float32) - pmin) /
                         (pmax - pmin) * 255).astype(np.uint8)
            else:
                frame = np.zeros_like(frame, dtype=np.uint8)
        else:
            pmin, pmax = compute_percentile_range(
                frame, self.percentile_vmin, self.percentile_vmax
            )
            f = frame.astype(np.float32)
            frame = np.clip(
                ((f - pmin) / (pmax - pmin) * 255),
                0, 255
            ).astype(np.uint8)

        color = self.selected_colors[index] if index < len(self.selected_colors) else "Gray"
        color_norm = (color or "").strip().title()

        f = frame.astype(np.float32)
        c = self.contrast / 100.0
        alpha = 0.1 + 2.9 * (c + 1.0) / 2.0
        beta = self.brightness * 1.28
        f = f * alpha + beta
        f = np.clip(f, 0, 255)
        frame_adjusted = f.astype(np.uint8)

        if frame_adjusted.ndim == 2:
            if color_norm == "Red":
                rgb_full = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
                rgb_full[..., 0] = frame_adjusted
            elif color_norm == "Green":
                rgb_full = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
                rgb_full[..., 1] = frame_adjusted
            elif color_norm == "Blue":
                rgb_full = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
                rgb_full[..., 2] = frame_adjusted
            else:
                rgb_full = np.stack([frame_adjusted]*3, axis=2)
        else:
            if frame_adjusted.shape[2] >= 3:
                if color_norm == "Red":
                    rgb_full = np.zeros_like(frame_adjusted[..., :3])
                    rgb_full[..., 0] = frame_adjusted[..., 0]
                elif color_norm == "Green":
                    rgb_full = np.zeros_like(frame_adjusted[..., :3])
                    rgb_full[..., 1] = frame_adjusted[..., 1]
                elif color_norm == "Blue":
                    rgb_full = np.zeros_like(frame_adjusted[..., :3])
                    rgb_full[..., 2] = frame_adjusted[..., 2]
                else:
                    rgb_full = frame_adjusted[..., :3].copy()
            else:
                rgb_full = np.stack(
                    [frame_adjusted[..., 0]]*3, axis=2
                ) if frame_adjusted.ndim == 3 else np.stack(
                    [frame_adjusted]*3, axis=2
                )

        rgb_full = ensure_uint8_rgb(rgb_full)

        # Draw 5x5 grid directly in the image (for magnifier)
        # Draw GAL spots directly in rgb_full
        if len(self.gal_model) > 0:
            for spot in self.gal_model:
                cx = int(round(spot.x))
                cy = int(round(spot.y))
                rad = int(round(spot.radius))
                if spot.is_blank:
                    cv_color = (255, 0, 0)  # blue (BGR)
                else:
                    cv_color = (255, 255, 255)  # white
                cv2.circle(rgb_full, (cx, cy), rad, cv_color, 1)


        if self.blob_centers:
            self.canvas.blob_circles = list(
                zip(self.blob_centers, self.blob_radii)
            )

        h, w = rgb_full.shape[:2]
        new_w = effective_w
        new_h = max(1, int(h * (new_w / float(w))))
        frame_resized_rgb = cv2.resize(
            rgb_full, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        rgb = ensure_uint8_rgb(frame_resized_rgb)

        self.canvas.show_frame(rgb, color_norm, original_color=rgb_full)

        h_img, w_img = rgb.shape[:2]
        self.canvas.setMinimumWidth(effective_w)
        self.canvas.setMaximumWidth(effective_w)
        self.canvas.resize(effective_w, h_img)
        self.canvas.draw_idle()

    def show_axes_size_popup(self):
        main_w = self.canvas.width()
        main_h = self.canvas.height()
        prev_w = getattr(self.canvas, "preview_img_w", 0)
        prev_h = getattr(self.canvas, "preview_img_h", 0)
        QMessageBox.information(
            self,
            "Image Dimensions",
            f"MAIN IMAGE: {main_w}×{main_h}px\n"
            f"PREVIEW: {prev_w}×{prev_h}px\n"
            f"Percentile: {self.percentile_vmin}%-{self.percentile_vmax}%\n"
            f"Blobs: {len(self.blob_centers)} (top-left)"
        )

# ---------------- main ----------------

if __name__ == "__main__":
    tiff_path = r"D:\python-master\2024.08.01_cLift-Kontrolle_A0206-02_Rl50Gh35_1677.tif"
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    width = screen.size().width()
    height = screen.size().height()
    dpi = screen.logicalDotsPerInch()
    window = MainWindow(tiff_path, width, height, dpi)
    window.showMaximized()
    sys.exit(app.exec_())
