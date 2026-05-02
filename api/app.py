from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Load model and scaler
with open("log_model.pkl", "rb") as f:
    log_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

FEATURE_COLUMNS = ["CLOSE_DEF_DIST", "SHOT_NUMBER", "PERIOD", "GAME_CLOCK", "SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME","SHOT_DIST"]

DEFAULTS = {
    "SHOT_NUMBER": 6.0,
    "DRIBBLES": 2.0,
    "TOUCH_TIME": 3.0,
}

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at root
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# Prediction endpoint
class ShotInput(BaseModel):
    SHOT_DIST: float
    CLOSE_DEF_DIST: float
    SHOT_CLOCK:float
    GAME_CLOCK:float
    PERIOD: float

@app.post("/predict")
def predict(shot: ShotInput):
    input_data = {**DEFAULTS, **shot.dict()}
    input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]

    input_scaled = scaler.transform(input_df)
    prob = log_model.predict_proba(input_scaled)[0][1]

    return {
        "prob_made": round(prob, 4),
        "prob_miss": round(1 - prob, 4),
        "prediction": int(prob >= 0.5)
    }