import logging
import numpy as np
import tifffile
from PyQt5.QtCore import QThread, pyqtSignal
from typing import Optional

logger = logging.getLogger(__name__)

class TiffLoaderThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, int)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            with tifffile.TiffFile(self.path) as tif:
                num_pages = len(tif.pages)
                frames = []
                for i, page in enumerate(tif.pages):
                    frames.append(page.asarray())
                    self.progress.emit(int((i + 1) / num_pages * 100))
                try:
                    stacked = np.stack(frames, axis=0)
                except:
                    stacked = frames
            logger.info("TIFF fully loaded into RAM")
            self.finished.emit(stacked, num_pages)
        except Exception as e:
            logger.error(f"Failed to load TIFF: {e}")
