from pathlib import Path

from src.config import LABEL_COLUMNS, TEST_DIR, TRAIN_DIR
from src.dataset import document_id, leaked_ids, list_documents, load_split


def test_document_id_strips_double_extension():
    assert document_id(Path("228094620-Rental-Agreement.pdf.docx")) == "228094620-Rental-Agreement"
    assert document_id(Path("24158401-Rental-Agreement.png")) == "24158401-Rental-Agreement"
    assert document_id(Path("18325926-Rental-Agreement-1.docx")) == "18325926-Rental-Agreement-1"


def test_label_columns_match_csv():
    for split in ("train", "test"):
        assert list(load_split(split).columns[2:]) == LABEL_COLUMNS


def test_every_test_label_has_a_test_document():
    labels = load_split("test")
    assert labels["path"].notna().all()
    assert len(labels) == len(list_documents(TEST_DIR)) == 4


def test_train_folder_has_ten_documents():
    assert len(list_documents(TRAIN_DIR)) == 10


def test_known_leak_is_detected():
    # 24158401 is labelled in train.csv but its file lives in test/.
    assert leaked_ids() == {"24158401-Rental-Agreement"}
