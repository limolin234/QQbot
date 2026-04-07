from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

MAX_SNAPSHOTS = 20


class SnapshotService:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.snapshot_root = self.repo_root / ".config_studio_snapshots"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)

    def create_snapshot(self, target_file: Path) -> None:
        if not target_file.exists() or not target_file.is_file():
            return

        rel_path = target_file.relative_to(self.repo_root)
        # Replace both forward and back slashes to get a valid filename on all OSes
        safe_rel = str(rel_path).replace("/", "__").replace("\\", "__")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dst = self.snapshot_root / f"{safe_rel}__{ts}.bak"
        shutil.copy2(target_file, dst)
        self._prune(safe_rel)

    def list_snapshots(self) -> list[str]:
        files = sorted(self.snapshot_root.glob("*.bak"), reverse=True)
        return [str(p.name) for p in files]

    def restore_snapshot(self, snapshot_name: str, target_file: Path) -> None:
        src = self.snapshot_root / snapshot_name
        if not src.exists():
            raise FileNotFoundError(f"snapshot not found: {snapshot_name}")
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target_file)

    def _prune(self, safe_rel: str) -> None:
        matched = sorted(self.snapshot_root.glob(f"{safe_rel}__*.bak"), reverse=True)
        for path in matched[MAX_SNAPSHOTS:]:
            path.unlink(missing_ok=True)
