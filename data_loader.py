
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SYNTHETIC DATASET  (2 000 records)
# ─────────────────────────────────────────────────────────────────────────────

DIAGNOSIS_CODES = [
    "250.00",  # Diabetes mellitus
    "428.0",   # Heart failure
    "486",     # Pneumonia
    "414.01",  # Coronary artery disease
    "585.3",   # Chronic kidney disease
    "496",     # COPD
    "038.9",   # Septicemia
    "410.71",  # Myocardial infarction
]


def generate_synthetic_data(n_samples: int = 2000) -> pd.DataFrame:
    
    
    # Age — skewed older (hospital population)
    age = np.random.normal(loc=62, scale=16, size=n_samples).clip(18, 95).astype(int)

    # Gender
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.52, 0.48])

    # Diagnosis code
    diagnosis_code = np.random.choice(DIAGNOSIS_CODES, size=n_samples)

    # Length of stay (days)
    length_of_stay = np.random.lognormal(mean=1.5, sigma=0.6, size=n_samples).clip(1, 30).astype(int)

    # Number of previous admissions
    num_previous = np.random.poisson(lam=1.8, size=n_samples).clip(0, 10)

    # Num medications
    num_medications = np.random.poisson(lam=7, size=n_samples).clip(1, 25)

    # Num lab procedures
    num_lab_procedures = np.random.poisson(lam=43, size=n_samples).clip(1, 80)

    # A1C result (for diabetic patients)
    a1c_result = np.random.choice(
        ["None", ">8", ">7", "Normal"], size=n_samples, p=[0.5, 0.2, 0.15, 0.15]
    )

    # --- Derive readmission probability from features ---
    readmit_prob = (
        0.05
        + 0.003 * (age - 18)                                # older → higher risk
        + 0.06 * num_previous                               # previous admissions strong signal
        + 0.01 * length_of_stay                             # longer stay → higher risk
        + 0.02 * (diagnosis_code == "428.0")                # heart failure high risk
        + 0.02 * (diagnosis_code == "250.00")               # diabetes high risk
        + 0.03 * (a1c_result == ">8")                       # uncontrolled diabetes
        + np.random.normal(0, 0.05, n_samples)              # noise
    ).clip(0.0, 0.95)

    readmission_status = (np.random.random(n_samples) < readmit_prob).astype(int)

    df = pd.DataFrame(
        {
            "Age": age,
            "Gender": gender,
            "Diagnosis_Code": diagnosis_code,
            "Length_of_Stay": length_of_stay,
            "Num_Previous_Admissions": num_previous,
            "Num_Medications": num_medications,
            "Num_Lab_Procedures": num_lab_procedures,
            "A1C_Result": a1c_result,
            "Readmission_Status": readmission_status,
        }
    )

    # Inject ~3 % missing values into Age and Length_of_Stay
    for col in ["Age", "Length_of_Stay"]:
        mask = np.random.random(n_samples) < 0.03
        df.loc[mask, col] = np.nan

    print(f"[data_loader] Synthetic dataset created → {df.shape}")
    print(f"[data_loader] Readmission rate: {df['Readmission_Status'].mean():.2%}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  REAL KAGGLE DATASET LOADER
#     https://www.kaggle.com/datasets/brandao/diabetes
# ─────────────────────────────────────────────────────────────────────────────

def load_kaggle_data(filepath: str = "diabetic_data.csv") -> pd.DataFrame:
    
    df = pd.read_csv(filepath, na_values=["?"])

    # Target: readmitted within 30 days
    df["Readmission_Status"] = (df["readmitted"] == "<30").astype(int)

    # Rename / select relevant columns
    df = df.rename(
        columns={
            "age": "Age_Bracket",
            "gender": "Gender",
            "diag_1": "Diagnosis_Code",
            "time_in_hospital": "Length_of_Stay",
            "number_inpatient": "Num_Previous_Admissions",
            "num_medications": "Num_Medications",
            "num_lab_procedures": "Num_Lab_Procedures",
            "A1Cresult": "A1C_Result",
        }
    )

    # Convert age bracket to midpoint integer
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }
    df["Age"] = df["Age_Bracket"].map(age_map)

    keep = [
        "Age", "Gender", "Diagnosis_Code", "Length_of_Stay",
        "Num_Previous_Admissions", "Num_Medications",
        "Num_Lab_Procedures", "A1C_Result", "Readmission_Status",
    ]
    df = df[keep].copy()

    print(f"[data_loader] Kaggle dataset loaded → {df.shape}")
    print(f"[data_loader] Readmission rate: {df['Readmission_Status'].mean():.2%}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PREPROCESSING + SMOTE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_and_smote(df: pd.DataFrame):

    df = df.copy()

    # --- Missing values (FIXED: no inplace) ---
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Length_of_Stay"] = df["Length_of_Stay"].fillna(df["Length_of_Stay"].median())
    df["Gender"] = df["Gender"].fillna("Unknown")
    df["Diagnosis_Code"] = df["Diagnosis_Code"].fillna("Unknown")
    df["A1C_Result"] = df["A1C_Result"].fillna("None")

    # --- Label Encoding ---
    cat_cols = ["Gender", "Diagnosis_Code", "A1C_Result"]
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [
        "Age", "Gender", "Diagnosis_Code", "Length_of_Stay",
        "Num_Previous_Admissions", "Num_Medications",
        "Num_Lab_Procedures", "A1C_Result",
    ]

    # Convert to DataFrame (IMPORTANT for NaN check)
    X = df[feature_cols]
    y = df["Readmission_Status"]

    # 🔥 FINAL SAFETY FIX (VERY IMPORTANT)
    if X.isnull().sum().sum() > 0:
        print("[FIX] NaNs found → filling with 0")
        X = X.fillna(0)

    # Debug check
    print("NaN count in X:", X.isnull().sum().sum())

    print(f"[data_loader] Before SMOTE  → class 0: {(y==0).sum():,}  |  class 1: {(y==1).sum():,}")

    # --- SMOTE ---
    smote = SMOTE(random_state=SEED)
    X_res, y_res = smote.fit_resample(X, y)

    print(f"[data_loader] After  SMOTE  → class 0: {(y_res==0).sum():,}  |  class 1: {(y_res==1).sum():,}\n")

    return X_res, y_res, encoders, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN — save processed CSV
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Swap to load_kaggle_data("diabetic_data.csv") if you have the real dataset
    df = generate_synthetic_data(n_samples=2000)
    df.to_csv("patient_data.csv", index=False)
    print("[data_loader] Saved → patient_data.csv")

    X, y, encoders, features = preprocess_and_smote(df)
    print(f"[data_loader] Final feature matrix: {X.shape}")
