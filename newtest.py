import sys
import time
from datetime import datetime
import os
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
from PyQt5.QtCore import Qt

import matplotlib.patches as patches

# Ensure non-interactive backend for faster rendering when needed
matplotlib.use("Agg")


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


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for displaying single TIFF frames with magnifier and overview."""
    def __init__(self, width, height, dpi):
        # Figure sized in inches
        fig_width_in = width / dpi
        fig_height_in = height / dpi
        self.fig = Figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
        # main axis covering full figure
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        # overview inset (right side)
        self.over_ax = self.fig.add_axes([0.92, 0.5, 0.08, 0.5])
        self.over_ax.axis("off")
        # magnifier inset (bottom-right)
        self.mag_ax = self.fig.add_axes([0.75, 0.05, 0.2, 0.2])
        self.mag_ax.axis("off")
        self.mag_ax.set_visible(False)

        super().__init__(self.fig)

        # connections
        self.mpl_connect("draw_event", self.on_draw_event)
        self.mpl_connect("motion_notify_event", self.on_mouse_motion)
        self.mpl_connect("button_release_event", self.update_overview_rect)
        self.mpl_connect("key_press_event", self.on_key_press)
        self.mpl_connect("resize_event", self.on_resize)

        self.draw_start = time.time()

        # custom colormaps
        self.red_cmap = LinearSegmentedColormap.from_list("red_map", [(0, "black"), (1, "red")])
        self.green_cmap = LinearSegmentedColormap.from_list("green_map", [(0, "black"), (1, "green")])

        # for reuse
        self.main_img = None
        self.overview_img = None
        self.overview_rect = None
        self.mag_img = None
        self.mag_patch = None  # rectangle on main image
        self.img_shape = None
        self.current_cmap = "gray"

        # focus to receive keyboard events
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def on_resize(self, event):
        # keep figure size consistent with canvas
        try:
            w, h = self.get_width_height()
            dpi = self.fig.dpi
            self.fig.set_size_inches(w / dpi, h / dpi)
        except Exception:
            pass
        # don't force draw here to avoid recursion

    def show_frame(self, img, label=""):
        """Display a single frame with assigned colormap and optionally update overview inset."""
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

        # main image
        if self.main_img is None:
            self.ax.clear()
            self.main_img = self.ax.imshow(img, cmap=cmap, interpolation="nearest", animated=True)
            self.ax.axis("off")
        else:
            self.main_img.set_data(img)
            self.main_img.set_cmap(cmap)

        self.ax.set_title(label)
        self.img_shape = img.shape[:2]  # (h, w)

        # Overview inset: show scaled-down full image
        h, w = img.shape[:2]
        # portrait if height > width
        if h > w:
            over_img = img.copy()
            # compute a resized preview to fit half canvas height
            ratio = (self.fig.bbox.height / 2) / h
            new_w = max(1, int(ratio * w))
            new_h = max(1, int(ratio * h))
            over_img_resized = cv2.resize(over_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            new_width = over_img_resized.shape[1] / self.fig.bbox.width
            # adjust over_ax position to occupy right area
            self.over_ax.set_position([1 - new_width, 0.5, new_width, 0.5])
            self.over_ax.set_visible(True)

            if self.overview_img is None:
                self.over_ax.clear()
                self.overview_img = self.over_ax.imshow(over_img_resized, cmap="gray", interpolation="nearest")
                self.over_ax.axis("off")
            else:
                self.overview_img.set_data(over_img_resized)

            # create or update overview rectangle
            if self.overview_rect is None:
                self.overview_rect = patches.Rectangle((0, 0),
                                                      over_img_resized.shape[1],
                                                      over_img_resized.shape[0],
                                                      edgecolor="red",
                                                      facecolor="none",
                                                      linewidth=1)
                self.over_ax.add_patch(self.overview_rect)
        else:
            # hide overview if not portrait
            self.over_ax.clear()
            self.over_ax.set_visible(False)
            self.overview_img = None
            self.overview_rect = None

        # Reset view to show full image
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
        # allow simple zoom keys: ctrl+'=' or '+' to zoom in, ctrl+'-' or '-' to zoom out
        key = event.key
        # print("DEBUG key:", key)
        if key is None:
            return
        # common representations include 'ctrl++', 'ctrl+-', or just '+', '-'
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
        """Zoom toward center by factor (<1 zooms in)."""
        if self.img_shape is None:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        width = (x1 - x0) * factor
        height = (y1 - y0) * factor
        self.ax.set_xlim(cx - width / 2.0, cx + width / 2.0)
        self.ax.set_ylim(cy + height / 2.0, cy - height / 2.0)  # inverted y-axis
        self.update_overview_rect()
        self.draw_idle()

    def zoom_out(self, factor=1.4):
        """Zoom out by factor (>1 zooms out)."""
        if self.img_shape is None:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        width = (x1 - x0) * factor
        height = (y1 - y0) * factor
        # clamp to image bounds
        width = min(width, self.img_shape[1])
        height = min(height, self.img_shape[0])
        self.ax.set_xlim(max(0, cx - width / 2.0), min(self.img_shape[1], cx + width / 2.0))
        self.ax.set_ylim(min(self.img_shape[0], cy + height / 2.0), max(0, cy - height / 2.0))
        self.update_overview_rect()
        self.draw_idle()

    def on_mouse_motion(self, event):
        """Show magnifier when cursor is on main axis; hide otherwise."""
        if event.inaxes != self.ax:
            # hide magnifier and patch
            self.mag_ax.set_visible(False)
            if self.mag_patch:
                self.mag_patch.set_visible(False)
            self.draw_idle()
            self.update_overview_rect(event)
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None or self.main_img is None:
            return

        img = self.main_img.get_array()
        h, w = img.shape[:2]

        # set target region half-size taking into account current zoom
        region_half = 50  # desired half-size on image coordinates
        # ensure region within bounds
        xi = int(round(x))
        yi = int(round(y))
        left = max(0, xi - region_half)
        right = min(w, xi + region_half)
        top = max(0, yi - region_half)
        bottom = min(h, yi + region_half)

        # If region is degenerate (near edge), pad by expanding the other side
        if (right - left) < 10 or (bottom - top) < 10:
            left = max(0, xi - region_half)
            right = min(w, xi + region_half)
            top = max(0, yi - region_half)
            bottom = min(h, yi + region_half)

        # crop and upscale
        crop = img[top:bottom, left:right]
        if crop.size == 0:
            return

        zoomed_size = 400
        try:
            zoomed = cv2.resize(crop, (zoomed_size, zoomed_size), interpolation=cv2.INTER_CUBIC)
        except Exception:
            # fallback if crop too small
            zoomed = cv2.resize(crop, (zoomed_size, zoomed_size), interpolation=cv2.INTER_NEAREST)

        # update magnifier image
        if self.mag_img is None:
            self.mag_img = self.mag_ax.imshow(zoomed, cmap=self.current_cmap, interpolation='nearest')
            self.mag_ax.axis('off')
        else:
            self.mag_img.set_data(zoomed)
            self.mag_img.set_cmap(self.current_cmap)

        self.mag_ax.set_visible(True)

        # update yellow rectangle on main image
        if self.mag_patch is None:
            self.mag_patch = patches.Rectangle((left, top), right - left, bottom - top,
                                               facecolor='none', edgecolor='yellow', linewidth=1)
            self.ax.add_patch(self.mag_patch)
        else:
            self.mag_patch.set_bounds(left, top, right - left, bottom - top)
            self.mag_patch.set_visible(True)

        # update overview rectangle
        self.update_overview_rect(event)
        self.draw_idle()

    def update_overview_rect(self, event=None):
        """Update overview rectangle to match visible portion of main image."""
        if not (self.overview_img and self.overview_rect and self.main_img):
            return

        try:
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
        except Exception:
            return

        # main image dims
        main_h, main_w = self.main_img.get_array().shape[:2]
        ov_arr = self.overview_img.get_array()
        ov_h, ov_w = ov_arr.shape[:2]

        # compute normalized rectangle in overview coords
        rect_x = max(0, x0 / main_w * ov_w)
        rect_y = max(0, y1 / main_h * ov_h)  # careful: y-axis inverted
        rect_w = (x1 - x0) / main_w * ov_w
        rect_h = (y0 - y1) / main_h * ov_h

        # bounds checks
        rect_x = max(0, min(ov_w, rect_x))
        rect_y = max(0, min(ov_h, rect_y))
        rect_w = max(1, min(ov_w - rect_x, rect_w))
        rect_h = max(1, min(ov_h - rect_y, rect_h))

        try:
            self.overview_rect.set_bounds(rect_x, rect_y, rect_w, rect_h)
        except Exception:
            pass

        # no draw here; caller will draw_idle()

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
        # show canvas inside scroll area
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)

        self.width = width
        self.height = height
        self.dpi = dpi

        # layout
        layout = QVBoxLayout()
        layout.addWidget(scroll_area)

        # horizontal scrollbar for frames
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

        # preload decision
        self.preload = self.should_preload(tiff_path)

        # safe to load TIFF now
        self.load_tiff(tiff_path)

    def should_preload(self, path):
        """Decide preload vs lazy based on TIFF size and available RAM."""
        file_size = os.path.getsize(path)  # bytes
        available_ram = psutil.virtual_memory().available  # bytes

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

            # ask user colors
            dialog = ColorSelector(self.frames_count, parent=self)
            if dialog.exec_():
                self.selected_colors = dialog.get_selected_colors()
            else:
                # fallback default (all gray)
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
        """Render a single TIFF frame."""
        index = self.scroll_bar.value()
        access_start = time.time()

        if self.preload:
            frame = self.frames[index]
        else:
            frame = self.tif.pages[index].asarray(out=self.buffer)

        orig_h, orig_w = frame.shape[:2]
        # scale to fit screen width
        scale_factor = self.width / orig_w
        target_height = max(1, int(orig_h * scale_factor))

        frame = cv2.resize(frame, (self.width, target_height), interpolation=cv2.INTER_AREA)

        access_time = (time.time() - access_start) * 1000
        print(f"Frame {index} access time: {access_time:.2f} ms")

        # color label for this frame
        try:
            color = self.selected_colors[index]
        except Exception:
            color = "Gray"

        # show in canvas
        self.canvas.show_frame(frame, color)

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
