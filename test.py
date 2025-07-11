from task_a import run_pipeline as run_task_a
from task_b import run_siamese_pipeline as run_task_b

if __name__ == "__main__":

    test_dir_task_a = None  # Replace with task A test images dir path
    test_dir_task_b = None  # Replace with task B test images dir path

    print("\n=== Running Task A Pipeline ===")
    run_task_a(test_dir=test_dir_task_a)

    print("\n=== Running Task B (Siamese) Pipeline ===")
    run_task_b(test_dir=test_dir_task_b)
