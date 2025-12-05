from PyQt5.QtGui import QPixmap, QImage

class ImageModel:
    def __init__(self):
        self.pixmap = None
        self.image = None

    def load_image(self, file_name):
        self.pixmap = QPixmap(file_name)
        self.image = self.pixmap.toImage()
        return self.pixmap, self.image  # Return both pixmap and image

    def get_pixel_color(self, x, y):
        if self.image and 0 <= x < self.image.width() and 0 <= y < self.image.height():
            color = self.image.pixelColor(x, y)
            r, g, b, _ = color.getRgb()
            return (r, g, b)
        return None

    def create_zoomed_view(self, x, y, rect_size=50, zoom_factor=4):
        if self.pixmap:
            rect = self.pixmap.copy(x - rect_size // 2, y - rect_size // 2, rect_size, rect_size)
            zoomed_pixmap = rect.scaled(rect_size * zoom_factor, rect_size * zoom_factor)
            return zoomed_pixmap
        return None
