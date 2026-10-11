import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗")

# ------------------------------------------------------------------
# Load the saved model (cached, so it loads only once)
# ------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "car_price_model.pkl")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        data = pickle.load(file)
    return data["model"], data["columns"], data["top_cars"], data["brand_model_map"]


model, model_columns, top_cars, brand_model_map = load_model()

# same mapping that was used while training
owner_map = {
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
    "Test Drive Car": 0,
}

# ------------------------------------------------------------------
# User interface
# ------------------------------------------------------------------
st.title("🚗 Used Car Price Predictor")
st.caption("Machine learning model (XGBoost) trained on historical used-car listings")

col1, col2 = st.columns(2)

with col1:
    brand_list = sorted(brand_model_map.keys())
    brand = st.selectbox("Car Brand", options=brand_list)

    # the model list depends on the selected brand
    model_list = brand_model_map[brand]
    car_model_name = st.selectbox("Car Model", options=model_list)

    year = st.slider("Manufacturing Year", min_value=2005, max_value=2026, value=2018)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=300000, value=50000, step=1000)

with col2:
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner_str = st.selectbox("Owner", list(owner_map.keys()))

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if st.button("Predict Price"):
    # 1. same feature engineering as in the notebook
    car_age = 2026 - year
    owner_val = owner_map[owner_str]

    brand_model_combined = f"{brand} {car_model_name}"
    if brand_model_combined not in top_cars:
        brand_model_combined = "Other"

    # 2. one-row table with the same columns as the training data
    input_df = pd.DataFrame({
        "km_driven": [km_driven],
        "owner": [owner_val],
        "car_age": [car_age],
        "brand_model": [brand_model_combined],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
    })

    # 3. one-hot encode, then force the exact training columns (missing ones become 0)
    input_encoded = pd.get_dummies(input_df, columns=["brand_model", "fuel", "seller_type", "transmission"])
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

    # 4. predict (log scale) and convert back to rupees
    predicted_log_price = model.predict(input_final)[0]
    prediction = np.expm1(predicted_log_price)

    st.success(f"### Estimated Selling Price: Rs {prediction:,.0f}")
    st.caption("This is an estimate based on historical listings, not an exact market price.")
