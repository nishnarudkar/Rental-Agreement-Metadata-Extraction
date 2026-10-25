"""Project-wide paths and dataset constants.

Everything that depends on the on-disk layout lives here so the rest of the
pipeline never hard-codes a path or a CSV column name.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXTRACTED_TEXT_DIR = OUTPUTS_DIR / "extracted_text"

SUPPORTED_EXTENSIONS = (".docx", ".png")

# Column names exactly as they appear in train.csv / test.csv.
# Note: the dataset spells "Agreement" as "Aggrement"; we keep it verbatim so
# predictions line up with the reference files.
FILE_NAME_COLUMN = "File Name"
LABEL_COLUMNS = [
    "Aggrement Value",
    "Aggrement Start Date",
    "Aggrement End Date",
    "Renewal Notice (Days)",
    "Party One",
    "Party Two",
]
