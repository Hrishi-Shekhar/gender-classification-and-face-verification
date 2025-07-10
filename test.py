# test.py

import sys
import os

# Import both task pipelines
from taskA_new import run_pipeline as run_task_a
from taskb_new import run_siamese_pipeline as run_task_b # type: ignore

def main():
    if len(sys.argv) != 2:
        print("Usage: python test.py /path/to/test_folder")
        sys.exit(1)

    test_folder = sys.argv[1]

    if not os.path.isdir(test_folder):
        print(f"Provided test directory does not exist: {test_folder}")
        sys.exit(1)

    print("\n========== Running Task A: Face Classification ==========")
    try:
        run_task_a(base_dataset_dir=None, test_dir=test_folder)
    except Exception as e:
        print(f"[ERROR] Task A failed: {e}")
        sys.exit(1)

    print("\n========== Running Task B: Face Verification ==========")
    try:
        run_task_b(train_dir=None, val_dir=None, test_dir=test_folder)
    except Exception as e:
        print(f"[ERROR] Task B failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
