"""Download the official Chronos-2 checkpoint into the local D: cache."""
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".hf-cache-chronos2"
REPO_ID = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def main() -> None:
    snapshot = snapshot_download(repo_id=REPO_ID, revision=REVISION, cache_dir=str(CACHE_DIR))
    print(f"Downloaded {REPO_ID}@{REVISION} to: {snapshot}")


if __name__ == "__main__":
    main()
