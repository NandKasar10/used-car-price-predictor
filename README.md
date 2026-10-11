# Used Car Price Predictor

A machine learning project that predicts the selling price of a used car and shows it in a simple Streamlit web app.

## Project structure

```
used-car-price-predictor/
|-- dataset/
|   `-- car_data.csv                     (put the dataset here)
|-- notebooks/
|   `-- car_price_prediction.ipynb       (data cleaning, features, model training, saving the model)
|-- models/
|   `-- car_price_model.pkl              (created by the notebook)
|-- app.py                               (Streamlit web app)
|-- requirements.txt
`-- README.md
```

## How to run

1. Install the libraries: `pip install -r requirements.txt`
2. Open `notebooks/car_price_prediction.ipynb` and run all cells. This creates `models/car_price_model.pkl`.
3. From the project root, start the app: `streamlit run app.py`

## Workflow

1. **Data cleaning**: check missing values, remove duplicate rows.
2. **Feature engineering**: `name` -> `brand_model` (for example "Maruti Swift"); `year` -> `car_age`; cars with fewer than 15 rows are grouped as "Other".
3. **Outlier handling**: prices clipped to the 1st-99th percentile, cars with 300,000 km or more removed.
4. **Encoding**: `owner` is ordinal (1, 2, 3, 4); brand_model, fuel, seller_type and transmission are one-hot encoded.
5. **Target transformation**: `log1p(price)` because prices are right-skewed; predictions are converted back with `expm1`.
6. **Model comparison** on the same 80/20 split: Linear Regression (baseline), Random Forest, XGBoost.
7. **Final model**: XGBoost. It is saved to a pickle file together with the column order, the list of popular cars and the brand-to-model map used by the dropdowns.

## How the app works

The app loads the pickle once, shows brand and model dropdowns (the model list depends on the brand), repeats the same preprocessing on the user's input (car age, owner number, one-hot encoding, same column order as training), predicts the log price and converts it back to rupees.

## Limitations

- The data are historical asking prices, so the result is an estimate, not an exact market price.
- Details such as the exact variant, condition and accident history are not in the data.
- Very new cars, very high kilometres and very rare models are outside what the model learned from.
