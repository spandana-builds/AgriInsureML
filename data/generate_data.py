"""
Synthetic dataset generator mimicking the Kaggle
'All Agriculture Data of India' structure.
Run once to produce agriculture_data.csv in this folder.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

STATES  = ["Uttar Pradesh","Maharashtra","Punjab","Haryana","Madhya Pradesh",
           "Rajasthan","Andhra Pradesh","Karnataka","Tamil Nadu","West Bengal"]
CROPS   = ["Rice","Wheat","Maize","Cotton","Sugarcane",
           "Soybean","Groundnut","Jowar","Bajra","Pulses"]
SEASONS = ["Kharif","Rabi","Zaid"]
SOILS   = ["Alluvial","Black","Red","Laterite","Sandy"]

state   = np.random.choice(STATES,  N)
crop    = np.random.choice(CROPS,   N)
season  = np.random.choice(SEASONS, N)
soil    = np.random.choice(SOILS,   N)

area        = np.random.uniform(0.5, 50,  N)          # hectares
production  = area * np.random.uniform(1, 6, N)       # tonnes
yield_      = production / area
rainfall_mm = np.random.normal(800, 250, N).clip(100, 2500)
normal_rain = np.random.normal(850, 150, N).clip(400, 2000)
rain_dev    = ((rainfall_mm - normal_rain) / normal_rain * 100).round(2)

soil_quality_map = {"Alluvial":5,"Black":4,"Red":3,"Laterite":2,"Sandy":1}
soil_q = np.array([soil_quality_map[s] for s in soil])
temperature = np.random.normal(26, 6, N).clip(10, 45)
fertiliser  = np.random.uniform(50, 400, N)
irrigation  = np.random.choice([0, 1], N, p=[0.4, 0.6])

# Claim logic (interpretable rules → realistic ~38% claim rate)
claim_prob = (
    (rain_dev < -20).astype(float) * 0.55 +
    (yield_ < 2.0).astype(float)   * 0.40 +
    (soil_q <= 2).astype(float)    * 0.25 +
    (irrigation == 0).astype(float)* 0.15 +
    np.random.uniform(0, 0.15, N)
)
claim_prob = claim_prob.clip(0, 1)
insurance_claim = (claim_prob > 0.55).astype(int)

df = pd.DataFrame({
    "State": state, "Crop": crop, "Season": season,
    "Soil_Type": soil, "Soil_Quality_Score": soil_q,
    "Area_Hectares": area.round(2),
    "Production_Tonnes": production.round(2),
    "Yield_Tonnes_Per_Ha": yield_.round(3),
    "Rainfall_mm": rainfall_mm.round(1),
    "Normal_Rainfall_mm": normal_rain.round(1),
    "Rainfall_Deviation_Pct": rain_dev,
    "Temperature_C": temperature.round(1),
    "Fertiliser_Kg_Ha": fertiliser.round(1),
    "Irrigation": irrigation,
    "Insurance_Claim": insurance_claim
})

df.to_csv("agriculture_data.csv", index=False)
print(f"Dataset saved — {N} rows, claim rate: {insurance_claim.mean():.1%}")
