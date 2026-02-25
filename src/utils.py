# ================================================
# utils.py
# ================================================
import joblib
from pathlib import Path

def save_model(model, outpath):
    outpath = Path(outpath)
    outpath.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, outpath)
    print(f"Model saved to {outpath}")