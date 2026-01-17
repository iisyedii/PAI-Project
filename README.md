# PAI-Project
Image Preprocessing Pipeline for Pneumonia Detection

This repository contains an image preprocessing script designed for a pneumonia detection project using chest X-ray images. The script prepares raw images for machine learning and deep learning models by applying multiple mandatory preprocessing and feature extraction techniques.

📌 Project Overview

The goal of this preprocessing pipeline is to:

Standardize chest X-ray images

Extract meaningful features

Prepare datasets for training and testing ML/DL models

The script processes raw .jpeg images, extracts HOG features, encodes labels, and splits the dataset into training and testing sets.

📂 Directory Structure
pneumonia_project/
│
├── data/
│   ├── raw/           # Raw chest X-ray images (.jpeg)
│   └── processed/     # Preprocessed images (auto-generated)
│
├── preprocess.py      # Image preprocessing script
└── README.md

⚙️ Requirements

Make sure the following Python libraries are installed:

pip install opencv-python numpy scikit-image scikit-learn torch torchvision pillow matplotlib

🛠️ Preprocessing Steps

The preprocess.py script performs the following mandatory preprocessing operations:

Image Loading

Image Resizing (128 × 128)

Grayscale Conversion

Gaussian Blurring

Image Augmentation (Horizontal Flip)

Image Binarization / Thresholding

Edge Detection (Canny)

Feature Extraction (HOG – Histogram of Oriented Gradients)

Image Flattening

Label Encoding

1 → Pneumonia image

0 → Normal image

Dataset Splitting

80% Training

20% Testing

Batch Configuration

➕ Additional Enhancements

Color inversion

Conversion of images into PyTorch tensors

Saving intermediate preprocessing results for visualization

🧠 Feature Extraction

HOG Features are extracted from grayscale images

These features are stored as NumPy arrays and used as model inputs

📊 Output

Preprocessed images are saved in the data/processed/ directory:

Grayscale images

Blurred images

Binary images

Edge-detected images

Console output displays:

Training and testing dataset sizes

Batch size configuration

▶️ How to Run

Place all raw chest X-ray images (.jpeg) in:

data/raw/


Run the script:

python preprocess.py


Processed images and extracted features will be generated automatically.

📌 Notes

Image filenames containing the word "PNEUMONIA" are automatically labeled as pneumonia cases.

The script is designed for educational and academic use, especially for AI/ML projects involving medical image analysis.

👤 Author

Syed Shabbir Raza
Capital University of Science and Technology (CUST)
AI / ML Student
