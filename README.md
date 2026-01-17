# Image Classification System

A machine learning project implementing a **Naive Bayes** classifier for handwritten digit recognition and facial image detection.

## Overview

This project implements a custom Naive Bayes classifier from scratch (identifying features, calculating probabilities, and predicting classes) on two computer vision datasets:

- **Digit Recognition**: 10-class classification (digits 0-9)
- **Face Detection**: Binary classification (face vs. non-face)

## Features

- **Custom Data Pipeline**: Parses ASCII-formatted image files and labels
- **Hybrid Feature Extraction**: Combines raw pixel features with statistical count features
- **Naive Bayes Classifier**: Probabilistic model implemented from scratch with Laplace smoothing
- **Comparable Performance**: Achieves >80% accuracy on both datasets

## Dataset Statistics

### Digits Dataset

- **Training**: 5,000 images
- **Validation**: 1,000 images  
- **Test**: 1,000 images
- **Classes**: 10 (digits 0-9)
- **Feature vector size**: 786 dimensions

### Face Dataset

- **Training**: 451 images
- **Validation**: 301 images
- **Test**: 150 images  
- **Classes**: 2 (face/non-face)
- **Feature vector size**: 4,202 dimensions

## Performance Results

The Naive Bayes classifier achieves the following accuracy on the validation sets:

| Dataset | Accuracy |
| :--- | :--- |
| **Digits** | **82.10%** |
| **Faces** | **87.71%** |

## Usage

### Prerequisites

- Python 3.x
- No external dependencies required (uses standard library: `math`, `time`, `random`)

### Running the Evaluation

To train the model and see accuracy results on both datasets, run:

```bash
python3 test_naive_bayes.py
```

### Running the Main Script

To run the general data loading and a single-image prediction test:

```bash
python3 main.py
```

## Project Structure

```
image_classification/
├── cs4346-data/        # Dataset directory
├── extract_data.py     # Feature extraction and data loading logic
├── naive_bayes.py      # Naive Bayes classifier implementation
├── main.py             # Main entry point for basic testing
├── test_naive_bayes.py # Full evaluation script
└── README.md           # Project documentation
```

## Algorithm Details

**Naive Bayes**:

- Calculates the prior probability of each class.
- Calculates the conditional probability of each feature given the class.
- Uses **Laplace Smoothing** to handle zero-probability features (unseen data).
- Works well with the discrete feature set (pixels encoded as 0/1).

## Author

Erik Olvera
