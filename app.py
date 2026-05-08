import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Title
st.title("🌍 Temperature Prediction App")

# Inputs
avg_temp = st.number_input("Enter the Land Average Temperature")
max_temp = st.number_input("Enter the Land Maximum Temperature")
min_temp = st.number_input("Enter the Land Minimum Temperature")

# Load model (retrain simple version)
@st.cache_resource
def load_model():
    df = pd.read_csv("C:\\Users\\LOQ\\Desktop\\datasets\\weather_data.csv")

    def clean_data(df):
        df = df.copy()
        df = df.drop(columns=[
            "LandAverageTemperatureUncertainty",
            "LandMaxTemperatureUncertainty",
            "LandMinTemperatureUncertainty",
            "LandAndOceanAverageTemperatureUncertainty"
        ])
        df["dt"] = pd.to_datetime(df["dt"])
        df["year"] = df["dt"].dt.year
        df = df.drop(columns=["dt"])
        df = df.dropna()
        df = df.set_index("year")
        return df

    df = clean_data(df)

    X = df[["LandAverageTemperature", "LandMaxTemperature", "LandMinTemperature"]]
    y = df["LandAndOceanAverageTemperature"]

    model = RandomForestRegressor(n_estimators=100,
                                   max_depth=50, 
                                   random_state=77)
    model.fit(X, y)

    return model

model = load_model()

# Predict button
if st.button("Predict Temperature"):
    input_data = np.array([[avg_temp, max_temp, min_temp]])
    prediction = model.predict(input_data)

    st.success(f"🌡 Predicted Temperature: {prediction[0]:.2f}")