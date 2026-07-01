import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
LOCAL_SWIFT = PROJECT.parent.parent / "ms-swift"
sys.path.insert(0, str(PROJECT))
if LOCAL_SWIFT.is_dir():
    sys.path.insert(0, str(LOCAL_SWIFT))
