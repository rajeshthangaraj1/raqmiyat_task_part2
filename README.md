# Document Type Classifier

This project trains a simple machine learning model to classify documents (e.g., invoice, resume, report) using traditional ML techniques (no LLMs).

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `sample.env` to `.env` and set your environment variables:
   ```bash
   cp sample.env .env
   ```
   Edit `.env` to set:
   ```
   DATA_PATH=./data
   ```

## Usage
- Run `main.py` to train and evaluate the model.
- The script will automatically generate synthetic training data if none exists.
- For testing, the script will read `test_samples.csv` (with sample document texts) and print predicted types for each—no user input required.
- Final accuracy is printed as a percentage.
- A confusion matrix image (`confusion_matrix.png`) is saved after evaluation.
- The trained model is saved as `model.joblib` in the project directory (if model saving is implemented).

## Approach
- Uses synthetic or open-source text samples.
- Preprocesses text (tokenization, vectorization).
- Trains a simple ML model (e.g., Logistic Regression, SVM, or Naive Bayes).
- Evaluates and reports accuracy and confusion matrix.

> **Note:** Create a `data` directory in the project root if you want to use your own document samples (e.g., `documents.csv`), or let the script generate synthetic data if none exists.

## Test Samples
- Place your test samples in `test_samples.csv` (one text per row, no label needed). The script will predict and print the type for each.

## .gitignore
- Make sure `.env`, `data/`, `test_samples.csv`, `confusion_matrix.png`, and `model.joblib` are listed in `.gitignore` to avoid committing sensitive or large files.
