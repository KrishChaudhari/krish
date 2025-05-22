# Sun Spot Activity Time Series Prediction

This repository contains a deep learning workflow for time series prediction, specifically applied to sun spot activity. Using TensorFlow, the project demonstrates how to preprocess data, create windowed datasets, build and train a sequence model (LSTM + Conv1D), tune the learning rate, evaluate results, and save the trained model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

The goal of this project is to predict daily minimum temperatures using a neural network, which can be adapted for sun spot or similar time series activity. The workflow includes data parsing, windowing, modeling, hyperparameter tuning, evaluation, and saving the trained model.

## Features

- **Data Preprocessing**: Loads and parses time series CSV data.
- **Windowed Dataset Creation**: Efficiently prepares data for sequence modeling.
- **Model Architecture**: Combines Conv1D and stacked LSTM layers for robust time series learning.
- **Learning Rate Scheduling**: Dynamically adjusts learning rate to optimize training.
- **Training & Evaluation**: Trains model, computes MSE/MAE metrics, and visualizes results.
- **Model Export**: Saves the trained model and compresses it for easy sharing.

## Project Structure

```
.
├── Sun_spot_activity_time_series_prediction.py
├── data/
│   └── daily-min-temperatures.csv
├── saved_model/
│   └── my_model (exported TensorFlow model)
└── saved_model.tar.gz
```

## Requirements

- Python 3.6+
- [TensorFlow](https://www.tensorflow.org/) (tested on 2.x)
- numpy
- matplotlib
- absl-py

You can install the requirements using:

```bash
pip install tensorflow numpy matplotlib absl-py
```

## Getting Started

1. **Clone the Repository**
    ```bash
    git clone https://github.com/krish4210/helicopt.git
    cd helicopt
    ```

2. **Prepare the Data**
    - Ensure `data/daily-min-temperatures.csv` is present.
    - If using your own sun spot data, place it in the `data/` folder and update the script accordingly.

3. **Run the Script**
    ```bash
    python "Sun_spot_activity_time_series_prediction.py"
    ```

## Usage

- The script loads the time series data, prepares it for training, creates and trains a deep learning model, evaluates its performance, and saves the trained model.
- Key steps and outputs are printed to the terminal and visualized with matplotlib.

## Results

- The model’s performance is reported in terms of Mean Squared Error (MSE) and Mean Absolute Error (MAE) on the validation set.
- Plots are generated to visualize the predictions against the actual values.
- The trained model is saved in `saved_model/` and compressed as `saved_model.tar.gz`.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Dataset: [Daily Minimum Temperatures in Melbourne](https://www.kaggle.com/datasets/mahmoudgamal/daily-min-temperatures-in-me)
- TensorFlow documentation and tutorials

---

**Contact:**  
For questions or suggestions, please open an issue or contact [KrishChaudhari](https://github.com/KrishChaudhari).

---

Feel free to modify or extend this README as your project evolves!