import os
from imutils import paths
import imageio.v2 as imageio  # For compatibility with newer imageio versions
import imgaug.augmenters as iaa

def augment_minority_class(class_name='female', augment_count=3, input_dir='Comys_Hackathon5 (1)/Comys_Hackathon5/Task_A'):
    """
    Augments images of a minority class in the training dataset.

    Args:
        class_name (str): Folder name of the class to augment.
        augment_count (int): Number of augmentations per image.
        input_dir (str): Root directory containing 'train' and 'val' folders.
    """
    print(f"[INFO] Augmenting '{class_name}' images in training set...")

    class_dir = os.path.join(input_dir, 'train', class_name)
    image_paths = list(paths.list_images(class_dir))

    # Define augmentation pipeline
    augmenters = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-15, 15), scale=(0.9, 1.1)),
        iaa.AdditiveGaussianNoise(scale=0.02 * 255),
        iaa.Multiply((0.9, 1.1)),
        iaa.LinearContrast((0.8, 1.2))
    ])

    # Apply augmentations and save new images
    for idx, img_path in enumerate(image_paths):
        image = imageio.imread(img_path)
        for aug_idx in range(augment_count):
            aug_image = augmenters(image=image)
            filename = f"aug_{idx}_{aug_idx}.jpg"
            imageio.imwrite(os.path.join(class_dir, filename), aug_image)

    print(f"[INFO] Done augmenting {len(image_paths)} images × {augment_count} times.")

# Example usage
# augment_minority_class(class_name='female', augment_count=3)
