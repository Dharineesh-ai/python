import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QScrollArea
from PyQt5.QtCore import Qt
from model import ImageModel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.image_model = ImageModel()

        self.layout = QVBoxLayout()

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_btn)

        self.scroll_area = QScrollArea()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

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
            pixmap, image = self.image_model.load_image(file_name)
            self.image_label.setPixmap(pixmap)
            self.image_label.adjustSize()
            self.image_label.setMouseTracking(True)
            self.image_label.mouseMoveEvent = self.mouse_move_event
            self.image_label.mousePressEvent = self.mouse_click_event

    def mouse_move_event(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.mouse_position_label.setText(f"Mouse Position: ({x}, {y})")
        color = self.image_model.get_pixel_color(x, y)
        if color:
            r, g, b = color
            self.color_info_label.setText(f"RGB: ({r}, {g}, {b}), HEX: #{r:02x}{g:02x}{b:02x}")
            if hasattr(self, 'zoom_window') and self.zoom_window is not None:
                self.zoom_window.update_zoom(x, y)

    def mouse_click_event(self, event):
        if event.button() == Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
            self.open_zoomed_view(x, y)

    def open_zoomed_view(self, x, y):
        self.zoom_window = ZoomWindow(self.image_model, x, y)
        self.zoom_window.show()

class ZoomWindow(QMainWindow):
    def __init__(self, image_model, x, y):
        super().__init__()

        self.setWindowTitle("Zoomed View")
        self.setGeometry(100, 100, 400, 400)

        self.image_label = QLabel(self)
        self.setCentralWidget(self.image_label)

        self.image_model = image_model
        self.update_zoom(x, y)

    def update_zoom(self, x, y):
        zoomed_pixmap = self.image_model.create_zoomed_view(x, y)
        if zoomed_pixmap:
            self.image_label.setPixmap(zoomed_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
