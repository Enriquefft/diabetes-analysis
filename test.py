import os
import kagglehub
from pathlib import Path
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')
path = Path(Path(kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")) / "diabetes_012_health_indicators_BRFSS2015.csv")
