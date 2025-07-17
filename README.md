# 💳 Online Payment Fraud Detection using Machine Learning

A machine learning-based fraud detection system trained on a large financial transactions dataset. This project analyzes over 6 million payment records to identify fraudulent behavior using statistical analysis, visual exploration, and neural networks.

---

## 📊 Dataset

- **Name**: Online Payment Fraud Detection
- **Source**: Kaggle
- **Rows**: ~6.36 million transactions
- **Target variable**: `isFraud` (0 = legitimate, 1 = fraudulent)

### Key Features:
- `step`: Time step of transaction
- `type`: Transaction type (TRANSFER, CASH_OUT, etc.)
- `amount`: Transaction amount
- `oldbalanceOrg`, `newbalanceOrig`: Sender's account before/after
- `oldbalanceDest`, `newbalanceDest`: Receiver's account before/after

---

## 🧰 Libraries Used

- `numpy`, `pandas`: Data manipulation
- `matplotlib`, `seaborn`: EDA and visualization
- `scipy`, `tabulate`: Data formatting and stats
- `tensorflow`: Deep learning for fraud prediction

---

## 🔍 Exploratory Data Analysis (EDA)

- Checked for class imbalance (`isFraud` has low % of 1s)
- Visualized transaction amounts by fraud vs. non-fraud
- Explored correlations between features
- Handled skewed data and outliers

---

## 🧠 Model

Used a simple deep learning model built with TensorFlow:

- Dense layers with ReLU activation
- Dropout for regularization
- Binary Cross-Entropy loss
- Adam optimizer

### Performance:
- Trained/validated with 70/30 split
- Evaluated using:
  - Accuracy
  - Confusion Matrix
  - ROC-AUC Curve
  - Precision & Recall

---

## 🛠 How to Run

1. Install required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scipy tensorflow
