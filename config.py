from pathlib import Path
import os

DATASET_NAME = Path(__file__).parent / "data" / "cardio_train.csv"
STANDARD_BIO_INDICATOR_ORDER = ["Normal","Above Normal","Well Above Normal"]