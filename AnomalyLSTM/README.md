# Anomaly Prediction Using LSTM

## Overview

This project implements an advanced anomaly detection pipeline using deep learning for tabular data. The core algorithm leverages a Bidirectional LSTM neural network, enhanced with SMOTE oversampling and data normalization, to robustly classify anomalies in sequential datasets.

## Features

- **Data Balancing:** Uses SMOTE to address class imbalance.
- **Normalization:** Applies MinMaxScaler to scale features.
- **Deep Learning Model:** Implements a Bidirectional LSTM model with dropout and batch normalization for improved generalization.
- **Model Evaluation:** Reports precision, recall, F1-score, confusion matrix, and per-class accuracy.
- **Automated Threshold Optimization:** Selects the best classification threshold based on F1-score.
- **Submission File Generation:** Outputs predictions in a CSV file ready for submission.

## Dataset

The code expects the following data files:
- `train.csv` — Training data with features and labels.
- `test.csv` — Test data with the same features (and optionally labels for evaluation).

> **Default file paths are set as:**  
> `C:\Users\hp\Downloads\kaggle_out\train.csv`  
> `C:\Users\hp\Downloads\kaggle_out\test.csv`  
> Update these paths in the notebook as needed.

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/krish4210/helicopt.git
   cd helicopt
   ```

2. **Install dependencies:**
   The notebook installs required versions of `numpy` and `pandas`:
   ```bash
   pip install numpy==1.24.4 pandas==1.5.3
   ```
   You should also install:
   ```bash
   pip install scikit-learn imbalanced-learn tensorflow
   ```

## Usage

1. **Edit the notebook:**
   - Set correct paths to your `train.csv` and `test.csv` files.
   - Adjust model parameters (epochs, batch size) if needed.

2. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook "anomaly prediction.ipynb"
   ```

3. **Outputs:**
   - The model prints performance metrics to the console.
   - Predictions are saved to a CSV file (default:  
     `C:\Users\hp\Downloads\new submission\submission2.csv`).

## Model Architecture

- Input Layer
- Bidirectional LSTM (64 units, return sequences)
- BatchNormalization
- Dropout (0.5)
- Bidirectional LSTM (32 units)
- Dropout (0.3)
- Dense (64, relu)
- Dropout (0.2)
- Output Dense (sigmoid)

Optimizer: Adam  
Loss: Binary Crossentropy  
Metrics: Accuracy

## Evaluation

The notebook prints:
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Per-class accuracy

If `test.csv` contains a `label` column, it will also evaluate test performance.

## File Structure

```
.
├── anomaly prediction.ipynb   # Main notebook with all code
├── README.md                  # (This file)
```

## Notes

- The notebook includes pip commands for dependency management.
- Some dependencies (e.g., SMOTE, TensorFlow) require additional installation.
- Paths are hardcoded for Windows—update them for your environment.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

## Contact

For questions or suggestions, open an issue or contact [KrishChaudhari](https://github.com/KrishChaudhari).

---

Feel free to copy and adapt this README for your project! If you need a badge section, contribution guidelines, or anything else, let me know.