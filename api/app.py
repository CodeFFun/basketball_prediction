from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Load models
with open("log_model.pkl", "rb") as f:
    log_model = pickle.load(f)
with open("dtree_model.pkl", "rb") as f:
    dtree_model = pickle.load(f)
with open("hbm_model.pkl", "rb") as f:
    hbm_model = pickle.load(f)
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# One shared scaler for all models
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# PCA only for XGB and HBM
with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

FEATURE_COLUMNS = [
    "CLOSE_DEF_DIST", "SHOT_NUMBER", "PERIOD", "GAME_CLOCK",
    "SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME", "SHOT_DIST"
]
DEFAULTS = {
    "SHOT_NUMBER": 6.0,
    "DRIBBLES": 2.0,
    "TOUCH_TIME": 3.0,
}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

class ShotInput(BaseModel):
    SHOT_DIST: float
    CLOSE_DEF_DIST: float
    SHOT_CLOCK: float
    GAME_CLOCK: float
    PERIOD: float

@app.post("/predict")
def predict(shot: ShotInput):
    input_data = {**DEFAULTS, **shot.dict()}
    input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]

    input_scaled = scaler.transform(input_df)

    log_prob   = float(log_model.predict_proba(input_scaled)[0][1])
    dtree_prob = float(dtree_model.predict_proba(input_scaled)[0][1])

    input_pca  = pca.transform(input_scaled)
    xgb_prob   = float(xgb_model.predict_proba(input_pca)[0][1])
    hbm_prob   = float(hbm_model.predict_proba(input_pca)[0][1])

    return {
        "logistic_regression": {
            "prob_made":  round(log_prob, 4),
            "prob_miss":  round(1 - log_prob, 4),
            "prediction": int(log_prob >= 0.5)
        },
        "decision_tree": {
            "prob_made":  round(dtree_prob, 4),
            "prob_miss":  round(1 - dtree_prob, 4),
            "prediction": int(dtree_prob >= 0.5)
        },
        "xgboost": {
            "prob_made":  round(xgb_prob, 4),
            "prob_miss":  round(1 - xgb_prob, 4),
            "prediction": int(xgb_prob >= 0.5)
        },
        "hist_gradient_boosting": {
            "prob_made":  round(hbm_prob, 4),
            "prob_miss":  round(1 - hbm_prob, 4),
            "prediction": int(hbm_prob >= 0.5)
        }
    }
