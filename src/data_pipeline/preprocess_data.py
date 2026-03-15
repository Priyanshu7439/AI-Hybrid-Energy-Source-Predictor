import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw")

gen = pd.read_csv(RAW_PATH / "Plant_1_Generation_Data.csv")
weather = pd.read_csv(RAW_PATH / "Plant_1_Weather_Sensor_Data.csv")

gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"])
weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"])

df = pd.merge(gen, weather, on="DATE_TIME")

print(df.shape)
print(df.head())

df = df[[
    "DATE_TIME",
    "AC_POWER",
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE"
]]

print(df.head())
print(df.shape)

df.to_csv("data/processed/solar_cleaned.csv", index=False)
print("Clean dataset saved")