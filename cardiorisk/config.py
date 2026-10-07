"""Single source of truth for paths, column definitions and valid value ranges.

Both the training code and the API import from here, so the feature list and the
allowed input ranges can never drift apart between training and serving.
This module only uses the standard library so it stays cheap to import.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------- paths
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
RAW_FILE = DATA_RAW_DIR / "processed.cleveland.data"
CLEAN_FILE = DATA_PROCESSED_DIR / "heart_clean.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Tried in order. The first is the classic UCI file, the second is the newer
# UCI zip bundle that contains the same file.
UCI_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data",
    "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
]

# ------------------------------------------------------------- columns
# The raw file has no header. 'num' is 0 (no disease) to 4 (severe disease).
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
RAW_TARGET = "num"
TARGET = "target"  # binary: 1 = any heart disease (num > 0), 0 = none

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Physiologically plausible ranges. Values outside become missing during
# cleaning, and the API rejects them with HTTP 422.
VALID_RANGES = {
    "age": (1, 120),
    "trestbps": (50, 250),
    "chol": (80, 700),
    "thalach": (50, 250),
    "oldpeak": (0, 10),
    "ca": (0, 3),
}
ALLOWED_CODES = {
    "sex": [0, 1],
    "fbs": [0, 1],
    "exang": [0, 1],
    "cp": [1, 2, 3, 4],
    "restecg": [0, 1, 2],
    "slope": [1, 2, 3],
    "thal": [3, 6, 7],
}

# Human-readable labels, used only for plots and documentation.
CODE_LABELS = {
    "sex": {0: "female", 1: "male"},
    "fbs": {0: "fasting sugar <= 120", 1: "fasting sugar > 120"},
    "exang": {0: "no exercise angina", 1: "exercise angina"},
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal", 4: "asymptomatic"},
    "restecg": {0: "normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
}
FEATURE_DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thalassemia test result",
}

# ------------------------------------------------------------ training
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
PRIMARY_METRIC = "roc_auc"
DECISION_THRESHOLD = 0.5
DEFAULT_MIN_TEST_ROC_AUC = 0.70  # practical minimum test ROC-AUC
# Override with the MIN_TEST_ROC_AUC environment variable when needed.

MLFLOW_EXPERIMENT = "heart-disease-risk"
REGISTERED_MODEL_NAME = "heart-disease-risk-classifier"
MODEL_VERSION = "1.0.0"
