# 📊 MetricsFromCM: Confusion Matrix Metrics Extractor

Testing with Python the extracting and calculating of metrics from confusion matrices in machine learning classification tasks.

## 🔍 Description

Functions to extract key performance metrics from confusion matrices:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

These fundamental metrics serve as building blocks for calculating more complex evaluation metrics such as accuracy, precision, recall, F1-score, and more.

## ⚙️ Prerequisites

- Python 3.6+
- NumPy
- Pandas
- scikit-learn

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/corticalstack/MetricsFromCM.git
cd MetricsFromCM

# Install dependencies
pip install numpy pandas scikit-learn
```

## 🧩 Functions

### `get_tp_from_cm(cm)`
Calculates the sum of True Positives from the confusion matrix.
- True Positives are the diagonal elements of the confusion matrix.

### `get_tn_from_cm(cm)`
Calculates the sum of True Negatives from the confusion matrix.
- For each class, True Negatives are the sum of all elements except those in the current row and column.

### `get_fp_from_cm(cm)`
Calculates the sum of False Positives from the confusion matrix.
- For each class, False Positives are the sum of the column minus the diagonal element.

### `get_fn_from_cm(cm)`
Calculates the sum of False Negatives from the confusion matrix.
- For each class, False Negatives are the sum of the row minus the diagonal element.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
