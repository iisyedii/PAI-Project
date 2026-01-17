import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
INPUT_DIR = '/home/ubuntu/pneumonia_project/data/raw'
OUTPUT_DIR = '/home/ubuntu/pneumonia_project/data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_image(image_path):
    filename = os.path.basename(image_path)
    print(f"Processing {filename}...")
    
    # 1. Image Loading
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 2. Resizing (Mandatory)
    resized_img = cv2.resize(img, (128, 128))
    
    # 3. Grayscale Conversion (Mandatory)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    
    # 4. Gaussian Blur (Mandatory)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # 5. Image Augmentation - Horizontal Flip (Mandatory)
    flipped_img = cv2.flip(resized_img, 1)
    
    # 6. Binarization/Thresholding (Mandatory)
    _, binary_img = cv2.threshold(blurred_img, 127, 255, cv2.THRESH_BINARY)
    
    # 7. Edge Detection (Mandatory)
    edges = cv2.Canny(blurred_img, 100, 200)
    
    # 8. Feature Extraction - HOG (Mandatory)
    features, hog_image = hog(gray_img, orientations=8, pixels_per_cell=(16, 16),
                             cells_per_block=(1, 1), visualize=True)
    
    # 9. Image Flattening (Mandatory)
    flattened_img = resized_img.flatten()
    
    # Additional from GitHub: Color Inversion
    inverted_img = cv2.bitwise_not(resized_img)
    
    # Additional from GitHub: Tensor Conversion
    transform = transforms.Compose([transforms.ToTensor()])
    tensor_img = transform(Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)))
    
    # Save intermediate steps for report visualization
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"gray_{filename}"), gray_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"blurred_{filename}"), blurred_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"binary_{filename}"), binary_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"edges_{filename}"), edges)
    
    return {
        "filename": filename,
        "features": features,
        "flattened": flattened_img,
        "tensor": tensor_img
    }

# Process all images
all_data = []
labels = []
image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.jpeg')]

for img_file in image_files:
    path = os.path.join(INPUT_DIR, img_file)
    result = preprocess_image(path)
    if result:
        all_data.append(result["features"])
        # 10. Label Encoding (Mandatory)
        label = 1 if "PNEUMONIA" in img_file else 0
        labels.append(label)

# 11. Dataset Splitting (Mandatory)
X_train, X_test, y_train, y_test = train_test_split(all_data, labels, test_size=0.2, random_state=42)

# 12. Batch Creation (Mandatory)
batch_size = 32
print(f"Preprocessing complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Batch size set to: {batch_size}")
