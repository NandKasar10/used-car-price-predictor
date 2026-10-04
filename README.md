# 🚘 Used Car Price Predictor

Ek machine learning pipeline aur interactive web application jo historical data ke basis par used cars ki estimated selling price predict karti hai. Ye project minor submission ke liye design kiya gaya hai jisme solid data science practices (jaise feature engineering aur dynamic UI) use ki gayi hain.

## ✨ Key Features
* **Dynamic Cascading Dropdowns:** Streamlit UI me car models selected brand ke hisaab se dynamically update hote hain (e.g., selecting 'Hyundai' only shows Hyundai models) jisse user invalid inputs nahi de sakta.
* **Log Transformation:** Target variable (Selling Price) par `np.log1p` apply karke right-skewed data ko normalize kiya gaya hai taaki algorithm outliers se bias na ho.
* **Robust Preprocessing:** Categorical features (Fuel, Transmission, Seller Type) ko handle karne ke liye One-Hot Encoding aur proper column reindexing implement ki gayi hai taaki model strictly training structure ko follow kare.
* **Interactive Web App:** Streamlit ka use karke ek clean, real-time prediction frontend banaya gaya hai.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn (Random Forest Regressor)
* **Frontend:** Streamlit
* **Serialization:** Pickle

## 📂 Project Structure
```text
Minor_Project/
├── dataset/
│   └── car_data.csv            # Original Kaggle dataset
├── models/
│   └── car_price_model.pkl     # Trained ML model, column names, aur brand-model mapping
├── car_price_prediction.ipynb  # Step-by-step EDA, cleaning aur training logic
├── app.py                      # Streamlit frontend application
└── requirements.txt            # Project dependencies