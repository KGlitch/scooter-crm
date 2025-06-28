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

model = joblib.load('/Users/matthiaswesthoff/Documents/studium/dhbw/neue konzepte2/scooter-crm/backend/xgb_scooter_model.joblib')

feature_columns = [
    'Größe', 'Stunde_sin', 'Stunde_cos', 'Stunden_bis_Event_Start',
    'Stunden_bis_Event_Ende', 'Stadtteil_Friedrichshain',
    'Stadtteil_Kreuzberg', 'Stadtteil_Lichtenberg', 'Stadtteil_Marzahn',
    'Stadtteil_Mitte', 'Stadtteil_Neukölln', 'Stadtteil_Prenzlauer Berg',
    'Stadtteil_Reinickendorf', 'Stadtteil_Spandau', 'Stadtteil_Treptow',
    'Wochentag_Friday', 'Wochentag_Monday', 'Wochentag_Saturday',
    'Wochentag_Sunday', 'Wochentag_Thursday', 'Wochentag_Tuesday',
    'Wochentag_Wednesday', 'Jahreszeit_Frühling', 'Jahreszeit_Herbst',
    'Jahreszeit_Sommer', 'Jahreszeit_Winter', 'Wetter_Bewölkt',
    'Wetter_Regen', 'Wetter_Sonnig', 'Wetter_Windig', 'Event Art_Kultur',
    'Event Art_Musik', 'Event Art_Politik', 'Event Art_Sport',
    'Feiertag_False', 'Feiertag_True'
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
    startTime: int
    endTime: int

class DemandRequest(BaseModel):
    wochentag: str
    jahreszeit: str
    wetter: str
    feiertag: bool
    stunde: int
    events: Optional[dict[str, Event]] = None  # <--- Events akzeptieren

@app.post("/predict")
def predict_demand(data: DemandRequest):
    from math import sin, cos, pi
    hour_angle = 2 * pi * data.stunde / 24
    sin_hour = round(sin(hour_angle), 4)
    cos_hour = round(cos(hour_angle), 4)

    predictions = {}

    for stadtteil in districts:
        features = dict.fromkeys(feature_columns, 0.0)
        features["Stunde_sin"] = sin_hour
        features["Stunde_cos"] = cos_hour
        if stadtteil in list(data.events.keys()):
            print("Event is Happening!")
            features["Größe"] = data.events[stadtteil].participants
            features["Stunden_bis_Event_Start"] = data.events[stadtteil].startTime - data.stunde
            features["Stunden_bis_Event_Ende"] = data.events[stadtteil].endTime - data.stunde
            features[f"Event Art_{data.events[stadtteil].category}"] = 1
        else:
            features["Stunden_bis_Event_Start"] = 9999
            features["Stunden_bis_Event_Ende"] = 9999

        features[f"Stadtteil_{stadtteil}"] = 1
        features[f"Wochentag_{data.wochentag}"] = 1
        features[f"Jahreszeit_{data.jahreszeit}"] = 1
        features[f"Wetter_{data.wetter}"] = 1

        features[f"Feiertag_{data.feiertag}"] = 1

        X = pd.DataFrame([features])
        y_pred = model.predict(X)[0]
        predictions[stadtteil] = round(float(y_pred), 2)

    return predictions
