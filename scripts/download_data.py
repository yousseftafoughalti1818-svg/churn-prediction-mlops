"""Download the public IBM Telco Customer Churn dataset (7,043 customers)."""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churn import config  # noqa: E402


def main() -> None:
    config.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if config.RAW_DATA_PATH.exists():
        print(f"Already downloaded: {config.RAW_DATA_PATH}")
        return
    print(f"Downloading {config.DATA_URL} ...")
    urllib.request.urlretrieve(config.DATA_URL, config.RAW_DATA_PATH)
    print(f"Saved to {config.RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
