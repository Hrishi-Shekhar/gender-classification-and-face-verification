from task_a import run_pipeline as run_task_a
from task_b import run_siamese_pipeline as run_task_b

if _name_ == "_main_":
    # Set your dataset directories here
    test_dir_task_a = "/path/to/test_images_task_a"  # Task A test images dir
    test_dir_task_b = "/path/to/test_images_task_b"  # Task B test images dir

    print("\n=== Running Task A Pipeline ===")
    run_task_a(test_dir=test_dir_task_a)

    print("\n=== Running Task B (Siamese) Pipeline ===")
    run_task_b(test_dir=test_dir_task_b)
