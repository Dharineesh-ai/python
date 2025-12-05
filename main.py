import sys
from PyQt5.QtWidgets import QApplication
from utils import setup_logging
from ui import MainWindow

setup_logging()

if __name__ == "__main__":
    import os
    if os.getenv('CI', 'false') == 'true':
        print("Running in CI mode, skipping TIFF loading and GUI.")
        sys.exit(0)
    
    # Change this path to your TIFF file
    tiff_path = r"D:\python-master\2024.08.01_cLift-Kontrolle_A0206-02_Rl50Gh35_1677.tif"  # Update this path
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    width = screen.size().width()
    height = screen.size().height()
    dpi = screen.logicalDotsPerInch()
    window = MainWindow(tiff_path, width, height, dpi)
    window.showMaximized()
    sys.exit(app.exec_())
