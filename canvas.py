import logging
import time
from datetime import datetime
from typing import Optional, Tuple

import cv2
import matplotlib.patches as patches
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPainterPath, QPen, QColor
from PyQt5.QtWidgets import QLabel

# Import from imageproc for ensure_uint8_rgb
from imageproc import ensure_uint8_rgb

logger = logging.getLogger(__name__)

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
        self.last_pixmap = None

    def update_image_from_ndarray(self, arr: Optional[np.ndarray]):
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
            self.border_px//2, self.border_px//2,
            self.overlay_size - self.border_px*2, self.overlay_size - self.border_px*2
        )
        painter.end()
        self.last_pixmap = out
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
        self.original_img: Optional[np.ndarray] = None
        self.overview_img = None
        self.overview_rect = None
        self.mag_patch = None
        self.img_shape = None
        self.current_cmap = "gray"
        self.over_ax = None
        self.overlay: Optional[MagnifierOverlay] = None

        self.preview_w = 240
        self.preview_label = QLabel(parent=None)
        self.preview_label.setFixedWidth(self.preview_w)
        self.preview_label.setStyleSheet("QLabel { border: 2px solid white; background: black; }")
        self.preview_label.setVisible(False)
        self.preview_label.setScaledContents(False)
        self._position_preview()
        self.preview_img_w = 0
        self.preview_img_h = 0

        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.last_view_rect = None  # (left, top, right, bottom)

    def get_width_height(self):
        return self.width(), self.height()

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

    def on_resize(self, event):
        try:
            w, h = self.get_width_height()
            dpi = self.fig.dpi
            self.fig.set_size_inches(w / dpi, h / dpi)
        except AttributeError:
            pass
        try:
            if self.overlay:
                parent_widget = self.overlay.parent()
                if parent_widget is not None:
                    ow, oh = self.overlay.width(), self.overlay.height()
                    x = min(self.overlay.x(), max(0, parent_widget.width() - ow))
                    y = min(self.overlay.y(), max(0, parent_widget.height() - oh))
                    self.overlay.move(x, y)
            self._position_preview()
        except AttributeError:
            pass

    def show_frame(self, display_img, label: str = "", original_color: Optional[np.ndarray] = None):
        # identical logic to newtest.py
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

        self.draw_start = time.time()
        self.draw_idle()

    def on_draw_event(self, event):
        label = self.ax.get_title()
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        draw_time_ms = (time.time() - self.draw_start) * 1000
        logger.info(f"[{label}] draw() completed at {timestamp} (draw time: {draw_time_ms:.2f} ms)")

    def on_key_press(self, event):
        if event.key in ("+", "="):
            self.zoom_in()
        elif event.key in ("-", "_"):
            self.zoom_out()

    def zoom_in(self, factor=0.7):
        if not self.img_shape:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w = (x1 - x0) * factor
        h = (y1 - y0) * factor
        self.ax.set_xlim(cx - w / 2, cx + w / 2)
        self.ax.set_ylim(cy + h / 2, cy - h / 2)
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
        self.ax.set_xlim(max(0, cx - w / 2), min(self.img_shape[1], cx + w / 2))
        self.ax.set_ylim(min(self.img_shape[0], cy + h / 2), max(0, cy - h / 2))
        self.update_overview_rect()
        self._update_preview(draw_rect=True)
        self.draw_idle()

    def on_mouse_motion(self, event):
        if not event.inaxes or not self.main_img:
            return
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
            crop = np.stack([crop] * 3, axis=2)
        crop = ensure_uint8_rgb(crop)
        if crop is None or crop.size == 0:
            return

        zoomed = cv2.resize(
            crop,
            (self.overlay.overlay_size, self.overlay.overlay_size),
            interpolation=cv2.INTER_CUBIC,
        )
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
                (l, t), r - l, b - t, facecolor="none", edgecolor="yellow", linewidth=1
            )
            self.ax.add_patch(self.mag_patch)
        else:
            self.mag_patch.set_bounds(l, t, r - l, b - t)
            self.mag_patch.set_visible(True)

        self._update_preview(draw_rect=True)
        self.draw_idle()

    def update_overview_rect(self, event=None):
        return

    def _update_preview(self, draw_rect: bool = True):
        if getattr(self, "original_img", None) is None:
            self.preview_label.setVisible(False)
            return

        orig = self.original_img
        h_orig, w_orig = orig.shape[:2]
        qimg = QImage(orig.data.tobytes(), w_orig, h_orig, orig.strides[0], QImage.Format_RGB888)

        parent_widget = self.preview_label.parent()
        if parent_widget is None:
            if self.parent() is not None:
                parent_widget = self.parent()
                self.preview_label.setParent(parent_widget)
        preview_h = self.preview_label.height() if self.preview_label.height() > 1 else 180

        pix = QPixmap.fromImage(qimg).scaled(self.preview_w, preview_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

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
