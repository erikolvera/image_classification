# Image Classification System

A machine learning project comparing Naive Bayes and Perceptron classifiers for handwritten digit recognition and facial image detection.

## Overview

This project implements and compares two fundamental classification algorithms on two distinct computer vision tasks:
- **Digit Recognition**: 10-class classification (digits 0-9)
- **Face Detection**: Binary classification (face vs. non-face)

## Features

- **Custom Data Pipeline**: Parses ASCII-formatted image files and labels
- **Hybrid Feature Extraction**: Combines raw pixel features with statistical count features
- **Scalable Architecture**: Designed to support multiple classification algorithms
- **Ready for ML Integration**: Feature vectors prepared for classifier training

## Dataset Statistics

### Digits Dataset
- Training: 5,000 images
- Validation: 1,000 images  
- Test: 1,000 images
- Image dimensions: 28 lines × 28 characters
- Classes: 10 (digits 0-9)
- Feature vector size: 786 dimensions

### Face Dataset
- Training: 451 images
- Validation: 301 images
- Test: 150 images  
- Image dimensions: 70 lines × 60 characters
- Classes: 2 (face/non-face)
- Feature vector size: 4,202 dimensions

## Feature Engineering

Each image is converted into a numerical feature vector using two methods:

1. **Raw Pixel Features**: Binary encoding of each character position
   - `#` or `+` → 1 (foreground)
   - ` ` (space) → 0 (background)

2. **Count Features**: Statistical aggregates
   - Total whitespace count
   - Total symbol count

These are concatenated to form the final feature vector.

## Current Status

**Phase 1 Complete**: Data preprocessing and feature engineering pipeline fully implemented.

**Next Steps**: Implementing classification algorithms (Naive Bayes and Perceptron) to evaluate performance on the prepared datasets.

## Requirements

```
Python 3.7+
```

No external dependencies required - uses only Python standard library (`math`, `time`, `random`, `statistics`).

## Usage

Currently implemented: Data loading and feature extraction pipeline.

```python
# Data is automatically loaded and processed
# Feature vectors are prepared for classifier training
```

Classification algorithms coming soon.

## Project Structure

```
image_classification/
├── cs4346-data/
│   ├── digitdata/
│   │   ├── traininglabels
│   │   ├── trainingimages
│   │   ├── validationlabels
│   │   ├── validationimages
│   │   ├── testlabels
│   │   └── testimages
│   └── facedata/
│       ├── facedatatrainlabels
│       ├── facedatatrain
│       ├── facedatavalidationlabels
│       ├── facedatavalidation
│       ├── facedatatestlabels
│       └── facedatatest
├── main.py
├── result.txt
└── README.md
```

## Key Insights

- Successfully parsed and processed 6,000+ digit images and 900+ face images from ASCII format
- Created high-dimensional feature spaces (786-dim for digits, 4,202-dim for faces) suitable for machine learning
- Hybrid feature approach combines fine-grained pixel information with aggregate statistics
- Pipeline designed for extensibility - ready to integrate multiple classification algorithms

## Future Improvements

- **Implement core classifiers**: Naive Bayes and Perceptron
- Add additional classifiers (SVM, k-NN, Neural Networks)
- Hyperparameter tuning and optimization
- Feature selection/dimensionality reduction (PCA)
- Confusion matrix visualization
- ROC curves and precision-recall analysis

## Author

Erik Olvera

## Acknowledgments

Dataset provided as part of CS 4346 coursework.
