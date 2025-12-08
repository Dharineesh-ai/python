import logging
import os
from typing import Optional

def setup_logging() -> None:
    """Configure logging with file and console handlers."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tiff_viewer.log'),
            logging.StreamHandler()
        ]
    )

def should_preload(path: str) -> bool:
    """Check if TIFF should be preloaded based on file size vs available RAM."""
    import psutil
    filesize = os.path.getsize(path)
    available_ram = psutil.virtual_memory().available
    file_gb = filesize / (1024**3)
    ram_gb = available_ram / (1024**3)
    logging.info(f"TIFF size: {file_gb:.2f} GB, Free RAM: {ram_gb:.2f} GB")
    return filesize / available_ram < 0.5
