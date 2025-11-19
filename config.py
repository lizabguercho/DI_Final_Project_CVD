from pathlib import Path
import os
import numpy as np

DATASET_NAME = Path(__file__).parent / "data" / "cardio_train.csv"
STANDARD_BIO_INDICATOR_ORDER = ["Normal","Above Normal","Well Above Normal"]
BMI_BINS = [-np.inf, 18.5, 25, 30, 35, 40, np.inf]
BMI_LABELS = ["Underweight", "Normal", "Overweight",
              "Obesity I", "Obesity II", "Obesity III"]

GRADIENT_COLORS = [ "#AFF8CC","#4CAF50",  "#FFC107",  "#FF9800",  "#F44336", "#9C27B0"]
BP_ORDER = ["Normal","Elevated","Hypertension Stage 1","Hypertension Stage 2","Hypertensive Crisis"]

AGE_BINS  = [18, 40, 50, 60, 90]
AGE_NAMES = ["18-39", "40-49", "50-59", "60-89"]
NUM_VARS = ['age_years', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo']
CAT_VARS = ["bp_category","bmi_bin","cholesterol","gluc"]

