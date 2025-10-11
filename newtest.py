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

class FocusDialog(QDialog):
    """Dialog for showing a focused/zoomed region of the image."""

    def __init__(self, focused_img, label="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Focused View: {label}")
        self.setModal(True)
        self.resize(600, 600)  # Fixed size for the dialog

        layout = QVBoxLayout(self)
        self.focus_canvas = MplCanvas(600, 600, 100)  # Smaller DPI for dialog
        layout.addWidget(self.focus_canvas)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        # Show the focused image at 200% zoom
        scale_factor = 2.0
        h, w = focused_img.shape[:2]
        zoomed_img = cv2.resize(focused_img, (int(w * scale_factor), int(h * scale_factor)),
                                interpolation=cv2.INTER_CUBIC)  # Smooth upscale
        self.focus_canvas.show_frame(zoomed_img, label)

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

        # Connect mouse click for focus dialog
        self.mpl_connect("button_press_event", self.on_mouse_click)

        self.img_shape = None  # will store current image size

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

        # Connect for overview updates during pan/zoom
        self.mpl_connect("motion_notify_event", self.update_overview_rect)
        self.mpl_connect("button_release_event", self.update_overview_rect)

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

    def on_mouse_click(self, event):
        """Open focused view on mouse click."""
        if event.inaxes != self.ax or event.button != 1:  # Left-click only on main axes
            return

        # Get click coordinates in image space
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Extract square region around click (20% of image size)
        img_array = self.main_img.get_array()
        h, w = img_array.shape[:2]
        region_size = int(min(h, w) * 0.2)  # 20% crop size
        x_start = max(0, int(x - region_size / 2))
        y_start = max(0, int(y - region_size / 2))
        x_end = min(w, x_start + region_size)
        y_end = min(h, y_start + region_size)

        focused_img = img_array[y_start:y_end, x_start:x_end]

        # Open dialog
        dialog = FocusDialog(focused_img, self.ax.get_title(), parent=self)
        dialog.exec_()

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