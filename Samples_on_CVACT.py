import tensorflow as tf
import numpy as np
import cv2
import os
import glob

# Define a function to sample an image
def sample_image(image):
    height, width, _ = image.shape
    crop_height, crop_width = height // 2, width // 2

    # Make sure the center is within a 1/4 area of the center of the sampled image
    def get_center_within_quarter_region():
        center_x = np.random.randint(crop_width // 4, 3 * crop_width // 4)
        center_y = np.random.randint(crop_height // 4, 3 * crop_height // 4)
        return center_x, center_y

    # Make sure the center is not within the 1/4 area of the center of the sampled image
    def get_center_outside_quarter_region():
        while True:
            center_x = np.random.randint(0, crop_width)
            center_y = np.random.randint(0, crop_height)
            if center_x < crop_width // 4 or center_x > 3 * crop_width // 4 or center_y < crop_height // 4 or center_y > 3 * crop_height // 4:
                return center_x, center_y

    # Acquire 4 sample images
    sampled_images = []
    centers = []

    # The first image is centered within the 1/4 area
    center_x, center_y = get_center_within_quarter_region()
    x_start = center_x
    y_start = center_y
    sampled_images.append(image[y_start:y_start + crop_height, x_start:x_start + crop_width])
    centers.append((600.0 - x_start, 600.0 - y_start))

    # The other three images are centered outside the 1/4 area
    for _ in range(3):
        center_x, center_y = get_center_outside_quarter_region()
        x_start = center_x
        y_start = center_y
        sampled_images.append(image[y_start:y_start + crop_height, x_start:x_start + crop_width])
        centers.append((600.0 - x_start, 600.0 - y_start))

    return sampled_images, centers

# Input and output folder paths
input_dir = 'CVACT'  # Replace with your dataset folder path
output_dir = 'sampled_CVACT_dataset'
os.makedirs(output_dir, exist_ok=True)

# Get all image file paths
image_paths = glob.glob(os.path.join(input_dir, '*.*'))

# Open file to save center point coordinates
with open(os.path.join(output_dir, 'centers.txt'), 'w') as f:
    # Iterate over all image files
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

        if image.shape != (1200, 1200, 3):
            print(f"Image size does not meet requirements, skip: {image_path}")
            continue

        # Sample images
        sampled_images, centers = sample_image(image)

        # Save the sampled image and center point coordinates
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        for i, (sampled_image, center) in enumerate(zip(sampled_images, centers)):
            output_path = os.path.join(output_dir, f'{base_name}_sampled_{i}.jpg')
            cv2.imwrite(output_path, cv2.cvtColor(sampled_image, cv2.COLOR_RGB2BGR))
            f.write(f'{base_name}_sampled_{i}.jpg: {center}\n')

print("All image sampling is complete and the sampled images and center point coordinates have been saved to the", output_dir)