# 🌐 Smart Panorama & Object Recognition System

## 👥 Team

| Name | LinkedIn |
|---|---|
| Ahmad Al-Kordy | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ahmadalkordy) |
| Yassin Yasser | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yassin-yasser1) |
| Nour Hatem | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nour-hatem-) |
| Mahmoud Hossam | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mahmoud-hossam-090958351) |

An end-to-end computer vision and deep learning pipeline designed to perform image preprocessing, feature extraction, panorama stitching, image segmentation, and scene classification. 

This project is meticulously structured into sequential cells, making it highly optimized for execution in **Google Colab** environments (T4 GPU recommended). It serves as a comprehensive demonstration of both classical computer vision algorithms and modern transfer learning techniques.

## ✨ Key Features & Pipeline Steps

1. **Image Preprocessing & Filtering**
   - Implements Gaussian and Median filtering to handle synthetic noise.
   - Evaluated quantitatively using Peak Signal-to-Noise Ratio (PSNR) and Mean Squared Error (MSE).

2. **Image Pyramids**
   - Progressive downsampling and detail extraction using **Gaussian** and **Laplacian** pyramids.
   - Includes accurate image reconstruction directly from Laplacian levels.

3. **Feature Detection & Matching**
   - **Harris Corner Detector:** Custom implementation with corner response heatmaps and nearest-neighbor matching accuracy metrics.
   - **SIFT (Scale-Invariant Feature Transform):** Robust keypoint detection and matching using Flann-based matchers, Lowe's ratio test, and RANSAC for inlier filtering.

4. **Panorama Stitching**
   - Utilizes SIFT keypoints and RANSAC-computed homography matrices to seamlessly warp and stitch multiple overlapping images into a single panorama.

5. **Image Segmentation**
   - Implements multiple segmentation strategies: **SLIC Superpixels**, **Normalized Cut**, and **GrabCut**.
   - Evaluates segmentation boundaries using Intersection over Union (IoU) metrics on generated foreground masks.

6. **Scene Classification (Deep Learning)**
   - Utilizes **EfficientNetB0** via transfer learning for high-accuracy image classification (~90-95%).
   - Integrated `tf.data` pipeline for efficient loading, dynamic data augmentation, and preprocessing.
   - Two-phase training strategy: Head training followed by fine-tuning the final 30 layers.
   - Comprehensive error analysis and confusion matrix visualizations.

## 📊 Dataset

The model is trained and evaluated on the **Intel Image Classification** dataset from Kaggle, encompassing various natural scenes (e.g., mountains, glaciers, seas, forests).

## 🛠️ Setup & Execution

The project is formatted as a cell-based Jupyter Notebook (`Smart_Panorama_FINAL.ipynb`) for straightforward, step-by-step execution. 

### Prerequisites
- Google Colab account (recommended) or a local Jupyter environment.
- Kaggle API token (`kaggle.json`) to download the dataset.
- Hardware: A GPU is highly recommended (e.g., NVIDIA RTX 4050 or Colab's T4 GPU).

### Dependencies
- `tensorflow` / `keras`
- `opencv-python` (`cv2`)
- `scikit-image`
- `scikit-learn`
- `matplotlib`
- `numpy`

### Instructions
1. Open the notebook in Google Colab.
2. Navigate to **Runtime $\rightarrow$ Change runtime type** and select **T4 GPU**.
3. Run **Cell 0** to install dependencies and verify GPU access.
4. Run **Cell 1** and upload your `kaggle.json` when prompted to automatically fetch the dataset.
5. Execute the remaining cells sequentially to observe filtering, feature matching, stitching, segmentation, and finally, model training and evaluation.
6. **Cell 17** allows you to upload and classify your own custom images using the trained EfficientNetB0 model.

## 📈 Results & Metrics Summary
The notebook concludes with a unified quantitative metrics cell tracking all core performance indicators:
- Filter PSNR improvements.
- Pyramid reconstruction fidelity.
- Harris & SIFT matching accuracies and inlier ratios.
- Segmentation map IoU overlap.
- Final EfficientNetB0 Top-1 and Top-3 Classification Accuracies.
