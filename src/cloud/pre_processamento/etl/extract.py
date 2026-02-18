import zipfile
from pathlib import Path
import logging
from typing import Generator, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataExtractor:
    def __init__(self, rclone_mount_path: str):
        self.mount_path = Path(rclone_mount_path)
        if not self.mount_path.exists():
            logger.warning(f"Mount path {rclone_mount_path} not found. Ensure rclone is mounted.")

    def list_zips(self) -> list[Path]:
        """Lists all ZIP files in the mount directory and subdirectories recursively."""
        zips = sorted(list(self.mount_path.rglob("*.zip")))
        logger.info(f"Found {len(zips)} ZIP files recursively in {self.mount_path}")
        return zips

    def stream_zip_content(self, zip_path: Path) -> Generator[Tuple[str, any], None, None]:
        """
        Opens a ZIP file and yields file-like objects for its content.
        Does not extract to disk.
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                for name in z.namelist():
                    # Accept .json, .csv or .data files
                    if any(name.endswith(ext) for ext in ['.json', '.csv', '.data']):
                        logger.info(f"Streaming {name} from {zip_path.name}")
                        with z.open(name) as f:
                            yield name, f
        except Exception as e:
            logger.error(f"Error reading {zip_path}: {e}")

if __name__ == "__main__":
    # Quick test logic
    extractor = DataExtractor("/workspace/gdrive/My Drive/QuantGod/L2/raw")
    # For local dev testing, it might fail if mount doesn't exist, which is expected.
