# Stereo-Image-Matching-and-Epipolar-Geometry

An OpenCV-based stereo vision project that detects and matches features between two images, estimates the Fundamental Matrix, visualizes epipolar geometry, and performs interactive stereo correspondence using Zero-mean Normalized Cross-Correlation (ZNCC).

## Overview

This project implements a stereo image matching pipeline using two images captured from slightly different viewpoints.

The system performs the following major operations:

1. Load and preprocess a stereo image pair.
2. Detect local features using SIFT.
3. Match feature descriptors using a Brute-Force matcher.
4. Apply Lowe's ratio test to remove weak feature matches.
5. Estimate the Fundamental Matrix using MAGSAC++ with RANSAC fallbacks.
6. Identify geometrically consistent feature correspondences.
7. Visualize selected SIFT feature matches.
8. Compute and display epipolar lines.
9. Search for corresponding points using ZNCC.
10. Provide an interactive interface where the user can click a point in the left image and find its corresponding point in the right image.

The implementation is designed as a practical demonstration of stereo vision, feature correspondence, epipolar geometry, and image matching.

---

## Features

### SIFT Feature Detection

The project uses the Scale-Invariant Feature Transform (SIFT) to detect distinctive keypoints and generate feature descriptors in both images.

```python
sift_processor = cv2.SIFT_create(
    nfeatures=10000,
    contrastThreshold=0.03,
    edgeThreshold=12
)
```

SIFT provides feature descriptions that are relatively robust to scale, rotation, and changes in image appearance.

---

### Feature Descriptor Matching

SIFT descriptors from the two images are matched using OpenCV's Brute-Force matcher.

A K-nearest-neighbor search is performed with `k=2`, followed by Lowe's ratio test to retain reliable matches.

The project uses a ratio threshold of `0.75`.

---

### Fundamental Matrix Estimation

The Fundamental Matrix is estimated from the matched feature points.

The implementation first attempts to use:

* USAC MAGSAC++
* RANSAC fallback
* 8-point algorithm fallback

This allows the program to continue using alternative estimation methods if the preferred method is unavailable or unsuccessful.

---

### Geometric Outlier Rejection

After estimating the Fundamental Matrix, geometrically consistent feature correspondences are identified as inliers.

These inliers are then ranked according to descriptor matching quality and used for visualization.

---

### Epipolar Geometry

For a selected point in the left image, the corresponding epipolar line is calculated in the right image.

The epipolar line restricts the search area for the corresponding point.

The project uses:

```python
cv2.computeCorrespondEpilines()
```

to calculate the epipolar geometry.

---

### ZNCC Stereo Matching

The project implements Zero-mean Normalized Cross-Correlation (ZNCC) to compare local image patches.

ZNCC is used to determine which region along the epipolar line provides the strongest correspondence.

The best matching point is selected based on the highest correlation score.

---

### Interactive Matching

After initialization, the user can click on a feature or distinctive region in the left image.

The program then:

1. Reads the selected pixel coordinates.
2. Computes the corresponding epipolar line.
3. Searches along the epipolar line.
4. Compares local image patches using ZNCC.
5. Identifies the best matching point.
6. Displays the corresponding point in the right image.

---

## System Pipeline

```text
             Stereo Image Pair
                    │
          ┌─────────┴─────────┐
          │                   │
       Image 1             Image 2
          │                   │
        SIFT                 SIFT
          │                   │
     Descriptors          Descriptors
          │                   │
          └─────────┬─────────┘
                    │
              BF Feature Matching
                    │
              Lowe Ratio Test
                    │
          Fundamental Matrix (F)
                    │
          MAGSAC++ / RANSAC
                    │
           Geometric Inliers
                    │
          ┌─────────┴─────────┐
          │                   │
   Feature Visualization   Interactive
                            Matching
                                │
                         Epipolar Line
                                │
                              ZNCC
                                │
                     Corresponding Point
```

---

## Technologies Used

* Python
* OpenCV
* NumPy
* SIFT
* Brute-Force Feature Matching
* Fundamental Matrix Estimation
* MAGSAC++
* RANSAC
* Epipolar Geometry
* Zero-mean Normalized Cross-Correlation (ZNCC)

---

## Requirements

* Python 3.9 or newer
* OpenCV
* NumPy
* A desktop environment capable of displaying OpenCV GUI windows

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Stereo-Image-Matching-and-Epipolar-Geometry.git
```

### 2. Navigate to the project directory

```bash
cd Stereo-Image-Matching-and-Epipolar-Geometry
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Input Images

The program expects two images in the project root directory:

```text
photo1.jpeg
photo2.jpeg
```

The images should represent the same scene from slightly different camera viewpoints.

Example:

```text
Stereo-Image-Matching-and-Epipolar-Geometry/
│
├── stereo_match.py
├── photo1.jpeg
├── photo2.jpeg
├── requirements.txt
└── README.md
```

The program loads the images using:

```python
img_1 = cv2.imread('photo1.jpeg')
img_2 = cv2.imread('photo2.jpeg')
```

---

## Usage

Run the program with:

```bash
python stereo_match.py
```

### Step 1 — Load Images

The program loads the two input images and resizes the second image to match the first image.

---

### Step 2 — Detect SIFT Features

SIFT keypoints and descriptors are extracted from both images.

The program then performs descriptor matching between the two images.

---

### Step 3 — Filter Feature Matches

Lowe's ratio test is applied to remove ambiguous matches.

The remaining matches are used to estimate the Fundamental Matrix.

---

### Step 4 — Estimate Fundamental Matrix

The program estimates the Fundamental Matrix using MAGSAC++.

If the estimation fails, the program falls back to RANSAC and then the 8-point algorithm.

---

### Step 5 — View Feature Matches

A verification window displays selected SIFT feature correspondences between the two images.

Press any key while the window is active to continue.

---

### Step 6 — Interactive Stereo Matching

The program opens the left and right image windows.

Click on a point in the left image.

The program calculates the corresponding epipolar line in the right image and searches for the best matching point using ZNCC.

---

### Step 7 — Exit

Press:

```text
ESC
```

inside the OpenCV window to terminate the program.

---

## Screenshots

### 1. SIFT Feature Matching

The system detects and matches distinctive features between the two stereo images.

![SIFT Feature Matching](screenshots/feature_matching.png)

---

### 2. Epipolar Geometry

The corresponding epipolar line is calculated in the right image based on a selected point in the left image.

![Epipolar Line](screenshots/epipolar_matching.png)

---

### 3. Interactive Stereo Correspondence

The selected point and its corresponding point are displayed using the interactive matching interface.

![Interactive Stereo Matching](screenshots/interactive_result.png)

---

## Algorithm Details

### ZNCC

For two local image patches, the project removes the mean intensity from each patch and calculates their normalized cross-correlation.

The resulting score is used to measure the similarity between the two image regions.

Higher ZNCC values indicate stronger correspondence.

---

### Epipolar Constraint

Once the Fundamental Matrix is known, a point in one image constrains its corresponding point in the other image to an epipolar line.

The program uses this constraint to reduce the search space for stereo correspondence.

---

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── stereo_match.py
├── photo1.jpeg
├── photo2.jpeg
├── screenshots/
│   ├── feature_matching.png
│   ├── epipolar_matching.png
│   └── interactive_result.png
└── LICENSE
```

---

## Limitations

* The quality of feature matching depends on the visual content of the input images.
* Textureless regions may produce weak ZNCC correspondence scores.
* Very large viewpoint changes can reduce the number of reliable SIFT matches.
* The program requires an environment capable of displaying OpenCV GUI windows.
* The quality of the Fundamental Matrix depends on the quality and geometric distribution of the feature matches.

---

## Future Improvements

Potential improvements include:

* Stereo rectification
* Disparity map generation
* Dense stereo matching
* Depth estimation
* Visualization of epipoles
* Support for additional feature detectors and descriptors
* Improved handling of textureless regions
* GPU-accelerated matching

---

## Camera calibration and sparse 3D reconstruction

`stereo_reconstruction.py` runs a full pipeline on a stereo pair:

1. Calibrates the camera from checkerboard photos (`cv2.calibrateCamera`) and reports the reprojection error.
2. Matches SIFT features between the two images.
3. Estimates the fundamental and essential matrices and recovers the relative camera pose.
4. Triangulates the matches into 3D points, removes outliers and saves a `.ply` point cloud with a plot.

The result is a sparse point cloud built from matched features, not a dense depth map.

```bash
pip install -r requirements.txt
python stereo_reconstruction.py \
    --calib_dir ./calib_images --board_cols 9 --board_rows 6 --square_size 25 \
    --img1 ./stereo/img1.jpg --img2 ./stereo/img2.jpg --output_dir ./results
```

Calibration photos are not included in this repository. Add your own photos of a printed checkerboard (9 x 6 inner corners, 25 mm squares) taken with the same camera as the stereo pair, and adjust `--board_cols`, `--board_rows` and `--square_size` if your board differs.

---
## Academic Context

This project demonstrates concepts commonly used in computer vision and stereo vision, including:

* Local feature detection
* Feature descriptors
* Feature correspondence
* Robust geometric estimation
* Fundamental Matrix estimation
* Epipolar geometry
* Stereo correspondence
* Local image correlation

---

## Author

**Bhavya Patel**

Master of Applied Computing
University of Windsor

---

## License

This project is intended for academic and educational purposes.

