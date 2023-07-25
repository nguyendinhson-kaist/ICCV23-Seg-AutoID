import cv2
import numpy as np

def apply_gridmask(image):
    
    # This function is used to apply GridMask augmentation to one image. It takes that image as an input and gives the augmented image as an output.

    mask = np.zeros_like(image)
    h, w = image.shape[:2]
    # Create grid-like pattern
    grid_size = 30
    for i in range(0, h, grid_size):
        for j in range(0, w, grid_size):
            mask[i:i+10, j:j+10, :] = 255  # Set a white patch

    # Combine the input image with the gridmask pattern
    augmented_image = cv2.addWeighted(image, 1, mask, 0.5, 0)

    return augmented_image

# image_path = '../../data/new_images/KS-FR-BLOIS_24330_1513710529019_0:1.png'
# image = cv2.imread(image_path)
# augmented_image = apply_gridmask(image)
# cv2.imwrite('testimages/gridmasktest.png', augmented_image)
