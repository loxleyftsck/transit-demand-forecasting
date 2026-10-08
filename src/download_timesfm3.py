"""Download the official TimesFM 3.0 checkpoint into the local D: cache."""
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".hf-cache-timesfm3"
REPO_ID = "google/timesfm-3.0-pytorch"
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"


def main() -> None:
    snapshot = snapshot_download(repo_id=REPO_ID, revision=REVISION, cache_dir=str(CACHE_DIR))
    print(f"Downloaded {REPO_ID} to: {snapshot}")
    print("Weights use the TimesFM Non-Commercial License v1.0; see the model repository license.")


if __name__ == "__main__":
    main()
