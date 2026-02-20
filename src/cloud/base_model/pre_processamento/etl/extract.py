import zipfile
import subprocess
import os
import json
import shutil
from pathlib import Path
import logging
from typing import Generator, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataExtractor:
    def __init__(self, path_or_remote: str, rclone_config: str = None):
        """
        path_or_remote: can be a local path or an rclone remote (e.g., 'drive:path/to/data')
        """
        self.path_or_remote = path_or_remote
        self.rclone_config = rclone_config
        self.is_remote = ":" in path_or_remote
        self.temp_dir = Path("/workspace/data/L2/temp_raw")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _run_rclone(self, args: list) -> str:
        cmd = ["rclone"]
        if self.rclone_config:
            cmd += ["--config", self.rclone_config]
        cmd += args
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Rclone error: {result.stderr}")
        return result.stdout

    def list_zips(self) -> list:
        """Lists all ZIP files recursively logic."""
        if not self.is_remote:
            mount_path = Path(self.path_or_remote)
            zips = sorted(list(mount_path.rglob("*.zip")))
            logger.info(f"Found {len(zips)} ZIP files locally in {self.path_or_remote}")
            return zips
        else:
            logger.info(f"Listing ZIP files from remote: {self.path_or_remote}...")
            # Use rclone lsjson to get all zips recursively
            output = self._run_rclone(["lsjson", "-R", "--include", "*.zip", self.path_or_remote])
            files_data = json.loads(output)
            # Filter only files (not dirs) that end with .zip
            zips = [f['Path'] for f in files_data if not f['IsDir'] and f['Path'].endswith('.zip')]
            logger.info(f"Found {len(zips)} ZIP files in remote.")
            return sorted(zips)

    def cleanup_temp(self):
        """Removes all files in the temp directory to start fresh."""
        if self.temp_dir.exists():
            logger.info(f"🧹 PRE-PROCESS: Cleaning up old temp files in {self.temp_dir}")
            for f in self.temp_dir.glob("*.zip"):
                try:
                    f.unlink()
                except:
                    pass

    def stream_zip_content(self, zip_identifier: str) -> Generator[Tuple[str, any], None, None]:
        """
        Processes a ZIP file. If remote, downloads it to temp, processes it, and ensures it's DELETED.
        """
        local_zip_path = None
        is_remote_download = self.is_remote
        
        try:
            if is_remote_download:
                # remote zip_identifier is a relative path from the root
                remote_full_path = f"{self.path_or_remote}/{zip_identifier}" if not zip_identifier.startswith("/") else f"{self.path_or_remote}{zip_identifier}"
                local_zip_path = self.temp_dir / Path(zip_identifier).name
                
                # Pre-emptive strike: delete if already exists to avoid permission/lock errors
                if local_zip_path.exists():
                    local_zip_path.unlink()
                    
                logger.info(f"📥 DOWNLOAD: {local_zip_path.name}")
                self._run_rclone(["copyto", remote_full_path, str(local_zip_path)])
                target_path = local_zip_path
            else:
                target_path = Path(zip_identifier)

            # Context manager ensures the ZIP file is properly closed before we try to delete it
            with zipfile.ZipFile(target_path, 'r') as z:
                for name in z.namelist():
                    if any(name.endswith(ext) for ext in ['.json', '.csv', '.data']):
                        logger.info(f"📂 STREAMING: {name}")
                        with z.open(name) as f:
                            yield name, f
                            
        except Exception as e:
            logger.error(f"❌ ERROR: Failed to process {zip_identifier}: {e}")
        finally:
            # THIS IS THE CRITICAL PART:
            # We only delete if it was a downloaded file (to not delete your Google Drive data!)
            if is_remote_download and local_zip_path and local_zip_path.exists():
                try:
                    logger.info(f"🗑️ CLEANUP: Deleting processed zip: {local_zip_path.name}")
                    local_zip_path.unlink()
                except Exception as e:
                    logger.warning(f"⚠️ CLEANUP WARNING: Could not delete {local_zip_path}: {e}")

if __name__ == "__main__":
    # Quick test logic
    extractor = DataExtractor("drive:PROJETOS/BTC_USDT_L2_2023_2026", rclone_config="/workspace/rclone.conf")
    # zips = extractor.list_zips()
    # print(zips[:5])
