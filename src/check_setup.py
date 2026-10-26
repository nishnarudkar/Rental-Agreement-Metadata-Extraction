"""Environment and dataset sanity check.

Run from the project root:

    python -m src.check_setup
"""

import importlib
import shutil
import subprocess
import sys

from src.config import LABEL_COLUMNS, TEST_DIR, TRAIN_DIR
from src.dataset import leaked_ids, list_documents, load_split

REQUIRED_PACKAGES = ["pandas", "docx", "PIL", "pydantic", "ollama", "tqdm", "dotenv", "pytest"]


def check_python() -> bool:
    ok = sys.version_info[:2] == (3, 11)
    print(f"[{'OK' if ok else 'WARN'}] Python {sys.version.split()[0]} ({sys.executable})")
    return ok


def check_packages() -> bool:
    all_ok = True
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
            print(f"[OK] import {name} {getattr(module, '__version__', '')}")
        except ImportError as exc:
            all_ok = False
            print(f"[FAIL] import {name}: {exc}")
    return all_ok


def check_gpu() -> None:
    if shutil.which("nvidia-smi") is None:
        print("[INFO] nvidia-smi not found; GPU not visible")
        return
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print(f"[OK] GPU: {result.stdout.strip()}")


def check_ollama() -> bool:
    from src.llm import LLMError, LocalLLM

    llm = LocalLLM()
    try:
        llm.check()
    except LLMError as exc:
        print(f"[FAIL] Local LLM: {exc}")
        return False
    print(f"[OK] Local LLM: '{llm.config.model}' available at {llm.config.host} (num_ctx={llm.config.num_ctx})")
    return True


def check_data() -> bool:
    all_ok = True
    for split, folder in (("train", TRAIN_DIR), ("test", TEST_DIR)):
        labels = load_split(split)
        docs = list_documents(folder)
        missing = labels[labels["path"].isna()]["File Name"].tolist()
        unlabeled = sorted(set(docs) - {n.strip() for n in labels["File Name"]})
        print(f"[INFO] {split}: {len(labels)} label rows, {len(docs)} documents")
        for name in missing:
            print(f"  [WARN] label row has no file in {folder.name}/: {name}")
        for name in unlabeled:
            print(f"  [WARN] file has no label row: {name}")
        if list(labels.columns[2:]) != LABEL_COLUMNS:
            all_ok = False
            print(f"  [FAIL] unexpected columns: {list(labels.columns)}")

    leaks = leaked_ids()
    for name in sorted(leaks):
        print(f"[WARN] train/test overlap (exclude from few-shot examples): {name}")
    return all_ok


def main() -> int:
    print("== Python ==")
    py_ok = check_python()
    print("\n== Packages ==")
    pkg_ok = check_packages()
    print("\n== Hardware / runtimes ==")
    check_gpu()
    llm_ok = check_ollama()
    print("\n== Dataset ==")
    data_ok = check_data()
    ok = py_ok and pkg_ok and llm_ok and data_ok
    print(f"\nSetup check {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
