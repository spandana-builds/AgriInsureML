"""
app.py  –  Agricultural Insurance Claim Prediction  (Flask)
"""
import os, joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
BASE = os.path.dirname(__file__)

# Load artefacts once at startup
scaler   = joblib.load(os.path.join(BASE, "models/scaler.pkl"))
features = joblib.load(os.path.join(BASE, "models/features.pkl"))
best     = joblib.load(os.path.join(BASE, "models/best_model.pkl"))

model      = best["model"]
use_scaler = best["scaled"]
model_name = best["name"]

CROPS   = ["Rice","Wheat","Maize","Cotton","Sugarcane",
           "Soybean","Groundnut","Jowar","Bajra","Pulses"]
SEASONS = ["Kharif","Rabi","Zaid"]
SOILS   = {"Alluvial":5,"Black":4,"Red":3,"Laterite":2,"Sandy":1}
STATES  = ["Uttar Pradesh","Maharashtra","Punjab","Haryana","Madhya Pradesh",
           "Rajasthan","Andhra Pradesh","Karnataka","Tamil Nadu","West Bengal"]

CROP_ENC   = {c:i for i,c in enumerate(sorted(CROPS))}
SEASON_ENC = {s:i for i,s in enumerate(sorted(SEASONS))}
SOIL_ENC   = {s:i for i,s in enumerate(sorted(SOILS))}
STATE_ENC  = {s:i for i,s in enumerate(sorted(STATES))}

def build_features(form):
    rain_dev   = float(form["rain_dev"])
    soil_type  = form["soil_type"]
    crop       = form["crop"]
    season     = form["season"]
    state      = form["state"]
    area       = float(form["area"])
    production = float(form["production"])
    temp       = float(form["temperature"])
    fertiliser = float(form["fertiliser"])
    irrigation = int(form["irrigation"])

    yield_val  = production / max(area, 0.01)
    area_log   = np.log1p(area)
    prod_log   = np.log1p(production)
    drought    = int(rain_dev < -20)
    flood      = int(rain_dev >  30)
    low_yield  = int(yield_val < 2.0)
    soil_q     = SOILS.get(soil_type, 3)

    row = [rain_dev, soil_q, yield_val, area_log, prod_log,
           temp, fertiliser, irrigation,
           drought, flood, low_yield,
           CROP_ENC.get(crop,0), SEASON_ENC.get(season,0),
           SOIL_ENC.get(soil_type,0), STATE_ENC.get(state,0)]
    return np.array(row).reshape(1,-1)

@app.route("/")
def index():
    return render_template("index.html",
                           crops=sorted(CROPS),
                           seasons=sorted(SEASONS),
                           soils=sorted(SOILS),
                           states=sorted(STATES),
                           model_name=model_name)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        X = build_features(request.form)
        Xi = scaler.transform(X) if use_scaler else X
        pred  = int(model.predict(Xi)[0])
        proba = float(model.predict_proba(Xi)[0][1])
        result = "CLAIM" if pred == 1 else "NO CLAIM"
        color  = "danger" if pred == 1 else "success"
        icon   = "⚠️" if pred == 1 else "✅"
        advice = ("High risk detected. Consider filing a claim promptly."
                  if pred == 1 else
                  "Low risk. Your crop looks stable this season.")
        return render_template("result.html",
                               result=result, color=color,
                               icon=icon, advice=advice,
                               confidence=round(proba*100,1),
                               model_name=model_name)
    except Exception as e:
        return render_template("result.html", result="ERROR",
                               color="warning", icon="❌",
                               advice=str(e), confidence=0,
                               model_name=model_name)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    X  = build_features(data)
    Xi = scaler.transform(X) if use_scaler else X
    pred  = int(model.predict(Xi)[0])
    proba = float(model.predict_proba(Xi)[0][1])
    return jsonify({"prediction": pred,
                    "label": "Claim" if pred==1 else "No Claim",
                    "confidence": round(proba,4)})

if __name__ == "__main__":
    app.run()