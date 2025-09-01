# Suspicious Transactions Model

This project provides a machine learning pipeline for detecting suspicious (potentially fraudulent) financial transactions using transaction data. It includes data preprocessing, model training, prediction, and utilities for handling transaction datasets.

## Project Structure

```
.
├── LICENSE
├── README.md
├── requirements.txt
├── run_prediction.py
├── run_single_test.py
├── run_training.py
├── data/
│   └── transactions.csv
├── fraud_detection/
│   ├── __init__.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
├── models/
│   └── fraud_model.joblib
└── tests/
    ├── test_preprocess.py
    └── test_train.py
```

## Features

- **Data Preprocessing:** Converts raw transaction data into features suitable for machine learning.
- **Rule-based Labeling:** Labels transactions as suspicious or not using basic heuristics.
- **Model Training:** Trains a Random Forest classifier to detect suspicious transactions.
- **Prediction:** Predicts suspicious transactions on new data.
- **Single Transaction Test:** Allows testing a single transaction via command line.
- **Utilities:** Model saving/loading and banner printing.
- **Unit Tests:** Basic tests for preprocessing and training.

## Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd ml_models
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preparation

Place your transaction data CSV in the `data/` directory. The CSV should include columns:
- `Transaction ID`
- `Amount`
- `Sender UPI ID`
- `Receiver UPI ID`
- `State`
- `City`
- `Timestamp` (format: `%d-%m-%Y %H:%M`)

### 2. Training the Model

Run:
```sh
python run_training.py
```
This will preprocess the data, train a Random Forest model, and save it to `models/fraud_model.joblib`.

### 3. Predicting on New Data

To predict suspicious transactions in a CSV:
```sh
python run_prediction.py --input data/transactions.csv --model models/fraud_model.joblib
```
This will output a DataFrame with predictions.

### 4. Testing a Single Transaction

You can test a single transaction using:
```sh
python run_single_test.py
```
Follow the prompts to enter transaction details.

## Code Overview

- **Preprocessing:** `fraud_detection/preprocess.py`
  - `preprocess_data(csv_path)`: Loads and processes transaction data.
  - `suspicious_rule(row)`: Labels transactions using simple rules.

- **Training:** `fraud_detection/train.py`
  - `train_model(csv_path, model_path)`: Trains and saves the model.

- **Prediction:** `fraud_detection/predict.py`
  - `predict_new(csv_path, model_path)`: Loads model and predicts on new data.

- **Utilities:** `fraud_detection/utils.py`
  - `save_model`, `load_model`, `print_banner`

- **Single Transaction Preparation:** `run_single_test.py`
  - `prepare_single_transaction(...)`: Prepares a DataFrame for a single transaction.

## Testing

Run unit tests with:
```sh
pytest tests/
```

## Requirements

- Python 3.10+
- pandas
- scikit-learn
- joblib
- numpy

Install all dependencies using `pip install -r requirements.txt`.

## License

See [LICENSE](LICENSE) for details.