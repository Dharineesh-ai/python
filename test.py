import sys, os, numpy as np
import cv2, tifffile
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QScrollArea, QVBoxLayout, 
                             QWidget, QScrollBar, QLabel, QListWidget, QDialog, 
                             QHBoxLayout, QPushButton)
from PyQt5.QtCore import Qt

def ensure_uint8_rgb(img):
    if img is None or img.size == 0: return None
    img = np.ascontiguousarray(img.astype(np.uint8))
    if img.ndim == 2: img = np.stack([img]*3, -1)
    if img.shape[-1] == 3 and img[:,:,0].mean() > img[:,:,2].mean()*1.4:
        img = img[:,:,::-1]  # Auto-fix BGR
    return img

class ColorSelector(QDialog):
    def __init__(self, num_frames, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Colors to Frames")
        self.setModal(True)
        self.color_list = ["Red", "Green", "Blue", "Alpha"][:num_frames]
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Drag to reorder colors for each frame:"))
        
        self.list = QListWidget()
        self.list.setDragDropMode(QListWidget.InternalMove)
        for color in self.color_list: 
            self.list.addItem(color)
        layout.addWidget(self.list)
        
        ok_btn = QPushButton("Confirm")
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)
    
    def get_colors(self): 
        return [self.list.item(i).text() for i in range(self.list.count())]

class TiffViewer(QMainWindow):
    def __init__(self, path):
        super().__init__()
        self.setWindowTitle("TIFF Viewer - Lightweight")
        self.tif = None
        self.frame_idx = 0
        self.brightness = 0
        self.contrast = 1.0
        self.colors = []
        
        # Canvas setup
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.ax.axis('off')
        
        # Controls
        self.scroll = QScrollBar(Qt.Horizontal)
        self.scroll.valueChanged.connect(self.show_frame)
        
        self.b_slider = QScrollBar(Qt.Horizontal)
        self.b_slider.setRange(-100, 100)
        self.b_slider.valueChanged.connect(self.on_brightness)
        
        self.c_slider = QScrollBar(Qt.Horizontal)
        self.c_slider.setRange(0, 200)
        self.c_slider.valueChanged.connect(self.on_contrast)
        
        # Layout
        layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(scroll_area)
        
        layout.addWidget(self.scroll)
        
        bc_layout = QHBoxLayout()
        bc_layout.addWidget(QLabel("Brightness:"))
        bc_layout.addWidget(self.b_slider)
        bc_layout.addStretch()
        bc_layout.addWidget(QLabel("Contrast:"))
        bc_layout.addWidget(self.c_slider)
        layout.addLayout(bc_layout)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        self.load_tiff(path)
    
    def on_brightness(self, value):
        self.brightness = value / 50.0
        self.show_frame()
    
    def on_contrast(self, value):
        self.contrast = 0.5 + value / 100.0
        self.show_frame()
    
    def load_tiff(self, path):
        try:
            self.tif = tifffile.TiffFile(path)
            self.n_frames = len(self.tif.pages)
            print(f"Loaded TIFF with {self.n_frames} frames")
            
            dialog = ColorSelector(self.n_frames, self)
            if dialog.exec_():
                self.colors = dialog.get_colors()
            else:
                self.colors = ["Gray"] * self.n_frames
            
            self.scroll.setMaximum(self.n_frames - 1)
            self.scroll.setValue(0)
            self.show_frame()
        except Exception as e:
            print(f"Error loading TIFF: {e}")
    
    def show_frame(self, idx=None):
        if self.tif is None: return
        
        if idx is None:
            idx = self.scroll.value()
        
        self.frame_idx = idx
        frame = self.tif.pages[idx].asarray()
        
        # Normalize to 0-255
        fmin, fmax = frame.min(), frame.max()
        if fmax > fmin:
            norm_frame = ((frame.astype(float) - fmin) / (fmax - fmin) * 255).astype(np.uint8)
        else:
            norm_frame = np.zeros_like(frame, dtype=np.uint8)
        
        # Apply brightness/contrast
        f = norm_frame.astype(float)
        f = np.clip(f * self.contrast + self.brightness * 2.55, 0, 255).astype(np.uint8)
        
        # Color channel selection
        color = self.colors[idx % len(self.colors)].lower()
        if f.ndim == 2:
            if color == "red":
                rgb = np.stack([f, np.zeros_like(f), np.zeros_like(f)], axis=-1)
            elif color == "green":
                rgb = np.stack([np.zeros_like(f), f, np.zeros_like(f)], axis=-1)
            elif color == "blue":
                rgb = np.stack([np.zeros_like(f), np.zeros_like(f), f], axis=-1)
            else:
                rgb = np.stack([f]*3, axis=-1)
        else:
            rgb = f if f.shape[-1] == 3 else np.stack([f[:,:,0]]*3, axis=-1)
        
        rgb = ensure_uint8_rgb(rgb)
        self.ax.clear()
        self.ax.imshow(rgb)
        self.ax.set_title(f"Frame {idx+1}/{self.n_frames} - {self.colors[idx % len(self.colors)]}")
        self.ax.axis('off')
        self.canvas.draw()
    
    def closeEvent(self, event):
        if self.tif:
            self.tif.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tiff_path = r"D:\python-master\2024.08.01_cLift-Kontrolle_A0206-02_Rl50Gh35_1677.tif"
    viewer = TiffViewer(tiff_path)
    viewer.showMaximized()
    sys.exit(app.exec_())
