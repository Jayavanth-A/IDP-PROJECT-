# 🏥 MediPredict — Intelligent Healthcare Analytics
## 30-Day Hospital Readmission Prediction (ML + Flask)

---

## Project Structure

```
healthcare_readmission/
├── data_loader.py          ← Dataset generation + SMOTE preprocessing
├── train_model.py          ← Model training, evaluation, plots
├── app.py                  ← Flask web application
├── requirements.txt        ← Python dependencies
├── patient_data.csv        ← Generated after running data_loader.py
├── readmission_model.joblib← Saved model bundle (after training)
├── metrics.json            ← Model metrics JSON
├── static/
│   └── plots/              ← ROC, confusion matrix, feature importance PNGs
└── templates/
    └── index.html          ← Full-stack frontend dashboard
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Use Real Kaggle Dataset
```bash
# Install Kaggle CLI
pip install kaggle

# Set up Kaggle API key (~/.kaggle/kaggle.json)
# Then download:
kaggle datasets download -d brandao/diabetes
unzip diabetes.zip

# In data_loader.py, change line in __main__ to:
#   df = load_kaggle_data("diabetic_data.csv")
```

### 3. Generate data + Train model
```bash
python data_loader.py     # creates patient_data.csv
python train_model.py     # trains model, saves .joblib + plots
```

### 4. Run the web app
```bash
python app.py
# → Open http://127.0.0.1:5000
```

---

## Architecture

```
User Browser
     │
     ▼
index.html (fetch API)
     │
     ▼
Flask app.py (/predict endpoint)
     │  loads
     ▼
readmission_model.joblib
  ├── RandomForestClassifier (300 trees)
  ├── StandardScaler
  └── LabelEncoders (Gender, Diagnosis_Code, A1C_Result)
```

---

## ML Pipeline

| Step | Tool | Detail |
|------|------|--------|
| Data source | Synthetic / Kaggle | 2,000 / 101,766 records |
| Missing values | pandas fillna | Median imputation |
| Encoding | LabelEncoder | Gender, Diagnosis, A1C |
| Scaling | StandardScaler | All numerical features |
| Imbalance | SMOTE | Minority class oversampling |
| Model | RandomForestClassifier | 300 trees, balanced weights |
| Split | 80/20 stratified | train_test_split |
| CV | StratifiedKFold (5) | F1 scoring |
| Saved | joblib | Single .joblib bundle |

---

## API Reference

### POST /predict
```json
{
  "age": 67,
  "gender": "Male",
  "diagnosis_code": "428.0",
  "length_of_stay": 7,
  "num_previous_admissions": 3,
  "num_medications": 10,
  "num_lab_procedures": 45,
  "a1c_result": ">8"
}
```

**Response:**
```json
{
  "success": true,
  "label": 1,
  "risk": "High Risk",
  "risk_class": "high",
  "probability": 73.4
}
```

---

## Extending with Kaggle Real Dataset

The **Diabetes 130-US Hospitals** dataset (UCI / Kaggle) contains 101,766 real
inpatient encounters with 50 features. Our `load_kaggle_data()` function maps
the relevant columns to our feature set automatically.

Link: https://www.kaggle.com/datasets/brandao/diabetes

---

## Disclaimer
This tool is for educational and research purposes only.
It is not a certified medical device and should not replace clinical judgment.
