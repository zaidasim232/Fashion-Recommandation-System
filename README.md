# Fashion-Recommandation-System

## Project Goal
- Recommend similar fashion items from a selected product.

## About This Project

This is an end-to-end **Fashion Recommendation System** built from raw dataset preparation to a functional recommendation pipeline.

The project uses the following Kaggle dataset:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

## Approach

The system can be developed using two main methods:

- **Metadata-based recommendation** using `styles.csv`
- **Image-based recommendation** using **ResNet50** feature extraction for visual similarity

## Tech Stack

- **Python** — core programming language
- **Jupyter Notebook** — experimentation and model development
- **Pandas** — data loading and preprocessing
- **NumPy** — numerical computations
- **Matplotlib / Seaborn** — data visualization
- **TensorFlow / Keras** — deep learning and feature extraction
- **ResNet50** — pretrained CNN model for image embeddings
- **Scikit-learn** — similarity computation and recommendation logic
- **Pillow / OpenCV** — image processing

## Project Flow

- Load and clean the dataset
- Process product metadata and/or images
- Extract features
- Compute similarity between items
- Return top recommended fashion products

## Notes

This project is designed to be extended and improved with additional features, better embeddings, or a more advanced deep learning pipeline.

## Result
- The system suggests visually similar fashion products.

## Live Demo
- https://fashion-recommandation-system-bgqkd3t5xrlxxsrnbhbpuv.streamlit.app/
