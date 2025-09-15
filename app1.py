import pandas as pd
import numpy as np
import joblib
import streamlit as st


Kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('Scaler.pkl')

st.title('Customer segmentation App')
st.write ('Enter customer details to predict the segment.')

income = st.number_input('Income',min_value=0,max_value=200000,value=50000)
age = st.number_input('Age',min_value=18,max_value=100,value=35)
recency = st.number_input('Recency (days since last purchases) ',min_value=0,max_value=365,value=30)
total_spend = st.number_input('Total Spending (sum of purchases)',min_value=0,max_value=50,value=3)
num_web_visit_monts =st.number_input('Number of Web visits montly',min_value=0,max_value=100,value=10)
num_web_purchases = st.number_input('Number of Web purchases ',min_value=0,max_value=100,value=10)
num_store_purchases = st.number_input('Number of Store purchases ',min_value=0,max_value=100,value=10)
tenure = st.number_input("Tenure (days)", min_value=0, max_value=10000, value=2000)


input_data = pd.DataFrame({
    "Income": [income],
    "Age": [age],
    "Recency": [recency],
    "Tenure_days": [tenure],
    "Total_Spend": [total_spend],
    "NumWebVisitsMonth": [num_web_visit_monts],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases]
})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    
    cluster = Kmeans.predict(input_scaled)[0]

    st.success(f"Predicted Segment : Cluster{cluster}")

    st.write("""
            Cluster 0 - Earn a lot, spend a lot. Give them premium offers.

            Cluster 1 - Steady spenders. Keep them happy with easy offers.

            Cluster 2 - Look a lot, buy little. Attract them with discounts.

            Cluster 3 - Spend little but active now. Push with limited-time deals.

            Cluster 4 - Rich and older, prefer stores. Give VIP or premium service.

            Cluster 5 - Moderate income, not buying lately.

            """)
