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
    def __init__(self, path_or_remote: str, rclone_config: str = None, temp_dir: str = "data/L2/temp_raw"):
        """
        path_or_remote: can be a local path or an rclone remote (e.g., 'drive:path/to/data')
        """
        self.path_or_remote = path_or_remote
        self.rclone_config = rclone_config
        # remote if it has : and it's not a single char (Windows drive)
        self.is_remote = ":" in path_or_remote and not (len(path_or_remote.split(":")[0]) == 1 and path_or_remote[1] == ":")
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _run_rclone(self, args: list) -> str:
        # v4.9: Windows fallback - prefer rclone.exe if it exists in project root
        import os
        rclone_bin = "rclone"
        if os.name == 'nt' and Path("rclone.exe").exists():
            rclone_bin = str(Path("rclone.exe").absolute())
        cmd = [rclone_bin]
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
            return [str(p) for p in zips]
        else:
            logger.info(f"Listing ZIP files from remote: {self.path_or_remote}...")
            output = self._run_rclone(["lsjson", "-R", "--include", "*.zip", self.path_or_remote])
            files_data = json.loads(output)
            zips = [f['Path'] for f in files_data if not f['IsDir'] and f['Path'].endswith('.zip')]
            logger.info(f"Found {len(zips)} ZIP files in remote.")
            return sorted(zips)

    def list_trades_csvs(self, trades_remote: str) -> list:
        """Lists all compressed CSV trades recursively."""
        if not self.is_remote:
            mount_path = Path(trades_remote)
            csvs = sorted(list(mount_path.rglob("*.csv.gz")))
            logger.info(f"Found {len(csvs)} CSV files locally in {trades_remote}")
            return [str(p) for p in csvs]
        else:
            logger.info(f"Listing Trades CSV files from remote: {trades_remote}...")
            output = self._run_rclone(["lsjson", "-R", "--include", "*.csv.gz", trades_remote])
            files_data = json.loads(output)
            csvs = [f['Path'] for f in files_data if not f['IsDir'] and f['Path'].endswith('.csv.gz')]
            logger.info(f"Found {len(csvs)} Trades files in remote.")
            return sorted(csvs)

    def cleanup_temp(self):
        """Removes all files in the temp directory to start fresh."""
        if self.temp_dir.exists():
            logger.info(f"🧹 PRE-PROCESS: Cleaning up old temp files in {self.temp_dir}")
            for ext in ["*.zip", "*.csv.gz"]:
                for f in self.temp_dir.glob(ext):
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
            logger.error(f"❌ ERROR: Failed to process ZIP {zip_identifier}: {e}")
        finally:
            if is_remote_download and local_zip_path and local_zip_path.exists():
                try:
                    logger.info(f"🗑️ CLEANUP: Deleting processed zip: {local_zip_path.name}")
                    local_zip_path.unlink()
                except Exception as e:
                    logger.warning(f"⚠️ CLEANUP WARNING: Could not delete {local_zip_path}: {e}")

    def download_file(self, identifier: str, remote_base: str) -> Path:
        """Downloads a specific file (like a CSV or Parquet) from a remote directly to temp_dir."""
        is_remote_download = ":" in remote_base and not (len(remote_base.split(":")[0]) == 1 and remote_base[1] == ":")
        local_path = self.temp_dir / Path(identifier).name
        
        if is_remote_download:
            remote_full_path = f"{remote_base}/{identifier}" if not identifier.startswith("/") else f"{remote_base}{identifier}"
            if local_path.exists():
                local_path.unlink()
                
            logger.info(f"📥 DOWNLOAD FILE: {local_path.name}")
            self._run_rclone(["copyto", remote_full_path, str(local_path)])
        else:
            source_path = Path(remote_base) / identifier
            shutil.copy2(source_path, local_path)
            
        return local_path

if __name__ == "__main__":
    import yaml
    with open("src/cloud/base_model/configs/master_config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # Quick test logic
    extractor = DataExtractor(config['pipeline_paths']['raw_l2_source'], rclone_config="rclone.conf")
    # zips = extractor.list_zips()
    # print(zips[:5])
