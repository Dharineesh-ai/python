import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_btn)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.mouse_position_label = QLabel()
        self.layout.addWidget(self.mouse_position_label)

        self.color_info_label = QLabel()
        self.layout.addWidget(self.color_info_label)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.pixmap = QPixmap(file_name)
            self.image = self.pixmap.toImage()
            self.image_label.setPixmap(self.pixmap)
            self.image_label.setMouseTracking(True)
            self.image_label.mouseMoveEvent = self.mouse_move_event
            self.image_label.mousePressEvent = self.mouse_click_event

    def mouse_move_event(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.mouse_position_label.setText(f"Mouse Position: ({x}, {y})")
        if self.image and 0 <= x < self.image.width() and 0 <= y < self.image.height():
            color = self.image.pixelColor(x, y)
            r, g, b, _ = color.getRgb()
            self.color_info_label.setText(f"RGB: ({r}, {g}, {b}), HEX: #{r:02x}{g:02x}{b:02x}")
            if hasattr(self, 'zoom_window') and self.zoom_window is not None:
                self.zoom_window.update_zoom(x, y)

    def mouse_click_event(self, event):
        if event.button() == Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
            self.open_zoomed_view(x, y)

    def open_zoomed_view(self, x, y):
        self.zoom_window = ZoomWindow(self.pixmap, x, y)
        self.zoom_window.show()

class ZoomWindow(QMainWindow):
    def __init__(self, pixmap, x, y):
        super().__init__()

        self.setWindowTitle("Zoomed View")
        self.setGeometry(100, 100, 400, 400)

        self.image_label = QLabel(self)
        self.setCentralWidget(self.image_label)

        self.pixmap = pixmap
        self.update_zoom(x, y)

    def update_zoom(self, x, y):
        rect_size = 50
        zoom_factor = 4
        rect = self.pixmap.copy(x - rect_size // 2, y - rect_size // 2, rect_size, rect_size)
        zoomed_pixmap = rect.scaled(rect_size * zoom_factor, rect_size * zoom_factor)
        self.image_label.setPixmap(zoomed_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
