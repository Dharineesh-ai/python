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

# Ensure non-interactive backend for faster rendering
matplotlib.use("Agg")

class ColorSelector(QDialog):
    """Dialog for assigning colors to TIFF frames."""

    def __init__(self, num_frames, default_colors=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Colors to TIFF Frames")
        self.setModal(True)
        print(f"num_frames {num_frames}")

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
    """Matplotlib canvas for displaying single TIFF frames."""

    def __init__(self, width, height, dpi):
        self.width_ = width
        self.height_ = height
        fig_width_in = width / dpi
        fig_height_in = height / dpi

        self.fig = Figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])  # full-figure axes

        # Inset axes for overview (x, y, width, height) in figure coords [0–1]
        self.over_ax = self.fig.add_axes([0.92, 0.5, 0.08, 0.5])  
        self.over_ax.axis("off")

        # Inset axes for magnifier
        self.mag_ax = self.fig.add_axes([0.8, 0.05, 0.2, 0.2])
        self.mag_ax.axis("off")
        self.mag_ax.set_visible(False)
        self.mag_img = None
        self.mag_patch = None

        super().__init__(self.fig)
        
        self.mpl_connect("draw_event", self.on_draw_event)
        self.draw_start = time.time()

        # Custom colormaps
        self.red_cmap = LinearSegmentedColormap.from_list(
            "red_map", [(0, "black"), (1, "red")]
        )
        self.green_cmap = LinearSegmentedColormap.from_list(
            "green_map", [(0, "black"), (1, "green")]
        )
        # 👇 allow key press events
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        # Connect key press handler
        self.mpl_connect("key_press_event", self.on_key_press)

        self.img_shape = None  # will store current image size

        # Connect mouse motion for magnifier and overview
        self.mpl_connect("motion_notify_event", self.on_mouse_motion)
        self.mpl_connect("button_release_event", self.update_overview_rect)

        # Connect resize event for proper figure scaling
        self.mpl_connect("resize_event", self.on_resize)

    def on_resize(self, event):
        w, h = self.get_width_height()
        dpi = self.fig.dpi
        self.fig.set_size_inches(w / dpi, h / dpi)
        self.draw()

    def show_frame(self, img, label=""):
        """Display a single frame with assigned colormap and optional overview inset."""
    
        # --- Choose colormap ---
        if label == "Red":
            cmap = self.red_cmap
        elif label == "Green":
            cmap = self.green_cmap
        else:
            cmap = "gray"
    
        # --- Main image ---
        if hasattr(self, "main_img"):
            # Reuse existing image for speed
            self.main_img.set_data(img)
            self.main_img.set_cmap(cmap)
        else:
            self.ax.clear()
            self.main_img = self.ax.imshow(
                img, cmap=cmap,
                interpolation="nearest", resample=False, animated=True
            )
            self.ax.axis("off")
    
        self.ax.set_title(label)
    
        # --- Overview inset (only for portrait images) ---
        if img.shape[1] > img.shape[0]:  # portrait case
            #over_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            over_img = img
            h, w = over_img.shape[:2]
            ratio = (self.height_ / 2) / h
            over_img_resized = cv2.resize(
                over_img, (int(ratio * w), int(self.height_ / 2)),
                interpolation=cv2.INTER_AREA
            )
    
            new_width = over_img_resized.shape[1] / self.width_
            self.over_ax.set_position([1 - new_width, 0.5, new_width, 0.5])
            self.over_ax.set_visible(True)
    
            if hasattr(self, "overview_img"):
                self.overview_img.set_data(over_img_resized)
            else:
                self.over_ax.clear()
                self.overview_img = self.over_ax.imshow(
                    over_img_resized, cmap="gray",
                    interpolation="nearest", resample=False
                )
                self.over_ax.axis("off")
    
            if not hasattr(self, "overview_rect"):
                self.overview_rect = matplotlib.patches.Rectangle(
                    (0, 0),
                    over_img_resized.shape[1],
                    over_img_resized.shape[0],
                    edgecolor="red", facecolor="none", linewidth=1
                )
                self.over_ax.add_patch(self.overview_rect)
            else:
                self.overview_rect.set_bounds(
                    0, 0,
                    over_img_resized.shape[1],
                    over_img_resized.shape[0]
                )
        else:
            # Hide overview axis if not portrait
            self.over_ax.clear()
            self.over_ax.set_visible(False)
            self.overview_img = None
            self.overview_rect = None
    
        # --- Final draw ---
        self.draw_start = time.time()
        self.draw()
        self.img_shape = img.shape

    def on_draw_event(self, event):
        """Log frame rendering time."""
        label = self.ax.get_title()
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        draw_time_ms = (time.time() - self.draw_start) * 1000
        print(f"[{label}] draw() completed at {timestamp} "
              f"(draw time: {draw_time_ms:.2f} ms)")

    def on_key_press(self, event):
        print("DEBUG key:", event.key)  # keep for testing at first
    
        zoom_in_keys = ["ctrl++", "ctrl+]", "ctrl+equal", "ctrl+add", "ctrl+plus"]
        zoom_out_keys = ["ctrl+-", "ctrl+subtract", "ctrl+minus"]
    
        if event.key in zoom_in_keys:
            self.zoom_in()
        elif event.key in zoom_out_keys:
            self.zoom_out()

    def on_mouse_motion(self, event):
        if event.inaxes != self.ax:
            self.mag_ax.set_visible(False)
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

        # Define region size
        region_half = 50  # 100x100 target crop
        half_x = min(region_half, x, w - 1 - x)
        half_y = min(region_half, y, h - 1 - y)

        left = int(x - half_x)
        right = left + int(2 * half_x)
        top = int(y - half_y)
        bottom = top + int(2 * half_y)

        crop = img[top:bottom, left:right]

        # Upsample to fixed size for consistent display
        zoomed_size = 400
        zoomed = cv2.resize(crop, (zoomed_size, zoomed_size), interpolation=cv2.INTER_CUBIC)

        cmap = self.main_img.get_cmap()

        if self.mag_img is None:
            self.mag_img = self.mag_ax.imshow(zoomed, cmap=cmap, interpolation='nearest')
            self.mag_ax.axis('off')

        else:
            self.mag_img.set_data(zoomed)

        self.mag_ax.set_visible(True)

        # Update patch
        if not self.mag_patch:
            self.mag_patch = patches.Rectangle(
                (left, top), right - left, bottom - top,
                facecolor='none', edgecolor='yellow', linewidth=1
            )
            self.ax.add_patch(self.mag_patch)
        else:
            self.mag_patch.set_bounds(left, top, right - left, bottom - top)
            self.mag_patch.set_visible(True)

        self.draw_idle()
        self.update_overview_rect(event)

    def zoom_out(self):
        # Reset to full image
        if self.img_shape is None:
            return
    
        h, w = self.img_shape[:2]
        print(f" ~~~~~tiff_h, tiff_w {h}, {w}")
        
        self.setFixedSize(QSize(self.width_, self.height_))
        self.draw_idle()
        self.update_overview_rect()
    
    def zoom_in(self):
        print("zoom_in")
        if self.img_shape is None:
            return
    
        h, w = self.img_shape[:2]
        #print(f" ~~~~~tiff_h, tiff_w {h}, {w}")
        bbox = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        screen_w = bbox.width * self.fig.dpi
        screen_h = bbox.height * self.fig.dpi
        #print(f" ~~~~~fig_h, fig_w {screen_h}, {screen_w}")
    
        # Scale so smaller side fits canvas
        scale = min(screen_w / w, screen_h / h)
    
        view_w = screen_w / scale
        view_h = screen_h / scale
        
        self.setFixedSize(QSize(int(view_w), int(view_h)))
        self.draw_idle()
        self.update_overview_rect()
        
    def update_overview_rect(self, *args):
        print("update_overview_rect")
        """Update rectangle to match visible portion of main image."""
        if not (self.overview_img and self.overview_rect):
            return

        # Main image visible range
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()

        # Normalize to overview image coordinates
        ov_xmax, ov_ymax = self.overview_img.get_array().shape[1], self.overview_img.get_array().shape[0]
        rect_x = max(0, x0 / self.main_img.get_array().shape[1] * ov_xmax)
        rect_y = max(0, y0 / self.main_img.get_array().shape[0] * ov_ymax)
        rect_w = (x1 - x0) / self.main_img.get_array().shape[1] * ov_xmax
        rect_h = (y1 - y0) / self.main_img.get_array().shape[0] * ov_ymax

        self.overview_rect.set_bounds(rect_x, rect_y, rect_w, rect_h)
        self.draw_idle()

class MainWindow(QMainWindow):
    """Main TIFF viewer window."""

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
        print(f"Screen size is {width}, {height}")
        self.dpi = dpi
    
        # Layout with canvas + scroll bar
        layout = QVBoxLayout()
        layout.addWidget(scroll_area)
    
        # Create scroll bar FIRST
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
    
        # Add layout to central widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
        # Determine preload mode
        self.preload = self.should_preload(tiff_path)
    
        # Now safe to load TIFF (scroll_bar is already defined)
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
    
            dialog = ColorSelector(self.frames_count, parent=self)
            if dialog.exec_():
                self.selected_colors = dialog.get_selected_colors()
    
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
        
        
        #Rotate + resize in one step (OpenCV faster than np.rot90)
        # if frame.shape[0] > frame.shape[1]:  # height > width
        #     frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        orig_h, orig_w = frame.shape[:2]
        scale_factor = self.width / orig_w
        target_height = int(orig_h * scale_factor)
        
        frame = cv2.resize(frame, (self.width, target_height),
                           interpolation=cv2.INTER_AREA)

        access_time = (time.time() - access_start) * 1000
        print(f"Frame {index} access time: {access_time:.2f} ms")

        color = self.selected_colors[index]
        self.canvas.show_frame(frame, color)
        
    def on_tiff_loaded(self, frames, count):
        print("✅ TIFF loaded in background.")
        self.frames = frames
        self.frames_count = count
        self.progress_bar.setVisible(False)
    
        dialog = ColorSelector(self.frames_count, parent=self)
        if dialog.exec_():
            self.selected_colors = dialog.get_selected_colors()
    
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
    #tiff_path = (r"C:\Users\Raja Chandramohan\Pictures\Jessica\Feedback 4\2024.09.04_Norden Vac Disco_3901-01_700.TIF")
    
    tiff_path = (r"D:\python-master\2024.08.01_cLift-Kontrolle_A0206-02_Rl50Gh35_1677.tif")
    #tiff_path = (r"C:\Users\Raja Chandramohan\Pictures\michael H\Test-TIF.tif")

    app = QApplication(sys.argv)
    screen = app.primaryScreen()

    width = screen.size().width()
    height = screen.size().height()
    dpi = screen.logicalDotsPerInch()

    window = MainWindow(tiff_path, width, height, dpi)
    window.show()

    sys.exit(app.exec_())