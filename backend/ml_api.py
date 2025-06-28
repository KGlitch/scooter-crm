# ml_api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Optional

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = joblib.load("xgb_scooter_model.joblib")

feature_columns = [
    'Größe', 'Stunde_sin', 'Stunde_cos',
    'Stadtteil_Friedrichshain', 'Stadtteil_Kreuzberg', 'Stadtteil_Lichtenberg',
    'Stadtteil_Marzahn', 'Stadtteil_Mitte', 'Stadtteil_Neukölln',
    'Stadtteil_Prenzlauer Berg', 'Stadtteil_Reinickendorf', 'Stadtteil_Spandau',
    'Stadtteil_Treptow', 'Wochentag_Friday', 'Wochentag_Monday',
    'Wochentag_Saturday', 'Wochentag_Sunday', 'Wochentag_Thursday',
    'Wochentag_Tuesday', 'Wochentag_Wednesday', 'Jahreszeit_Frühling',
    'Jahreszeit_Herbst', 'Jahreszeit_Sommer', 'Jahreszeit_Winter',
    'Wetter_Bewölkt', 'Wetter_Regen', 'Wetter_Sonnig', 'Wetter_Windig',
    'Event Art_Kultur', 'Event Art_Musik', 'Event Art_Politik',
    'Event Art_Sport', 'Feiertag_False', 'Feiertag_True'
]

districts = [
    "Friedrichshain", "Kreuzberg", "Lichtenberg", "Marzahn", "Mitte",
    "Neukölln", "Prenzlauer Berg", "Reinickendorf", "Spandau", "Treptow"
]

class Event(BaseModel):
    district: str
    name: str
    participants: int
    category: str

class DemandRequest(BaseModel):
    wochentag: str
    jahreszeit: str
    wetter: str
    event_art: str
    feiertag: bool
    stunde: int
    größe: float = 300.0
    events: Optional[List[Event]] = None  # <--- Events akzeptieren

@app.post("/predict")
def predict_demand(data: DemandRequest):
    from math import sin, cos, pi
    hour_angle = 2 * pi * data.stunde / 24
    sin_hour = round(sin(hour_angle), 4)
    cos_hour = round(cos(hour_angle), 4)

    predictions = {}

    # Events pro Stadtteil aufsummieren
    event_counts = {d: 0 for d in districts}
    if data.events:
        for ev in data.events:
            if ev.district in event_counts:
                event_counts[ev.district] += ev.participants

    for stadtteil in districts:
        features = dict.fromkeys(feature_columns, 0.0)
        features["Größe"] = data.größe
        features["Stunde_sin"] = sin_hour
        features["Stunde_cos"] = cos_hour

        features[f"Stadtteil_{stadtteil}"] = 1
        features[f"Wochentag_{data.wochentag}"] = 1
        features[f"Jahreszeit_{data.jahreszeit}"] = 1
        features[f"Wetter_{data.wetter}"] = 1
        features[f"Event Art_{data.event_art}"] = 1
        features[f"Feiertag_{data.feiertag}"] = 1

        # Beispiel: Teilnehmerzahl als Einfluss auf Größe
        features["Größe"] += event_counts[stadtteil]

        X = pd.DataFrame([features])
        y_pred = model.predict(X)[0]
        predictions[stadtteil] = round(float(y_pred), 2)

    return predictions
