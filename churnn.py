


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Set up the Streamlit app
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Prediction Web App")

# Load and process data
def load_data():
    url = 'https://raw.githubusercontent.com/shreyasnimkhedkar/churnn/refs/heads/master/customer_churn%20(1).csv'
    df = pd.read_csv(url)
    df['Onboard_date'] = pd.to_datetime(df['Onboard_date'])
    df['Onboard_timestamp'] = df['Onboard_date'].astype('int64') // 10**9
    return df

data = load_data()

# Select specific columns
x = data[['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites']]
y = data[['Churn']]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=2)

# Standardize the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the model
model = LogisticRegression()
model.fit(x_train_scaled, y_train)

# Accuracy
train_pred = model.predict(x_train_scaled)
train_accuracy = accuracy_score(y_train, train_pred)

# Display model performance
st.subheader("Model Performance")
st.success(f"Training Accuracy: {train_accuracy:.2f}")

# --- User Input for Prediction ---
st.subheader("Enter Customer Information to Predict Churn")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
total_purchase = st.number_input("Total Purchase Amount", min_value=0.0, value=10000.0)
account_manager = st.selectbox("Has Account Manager?", [0, 1])
years = st.number_input("Years with Company", min_value=0.0, value=5.0)
num_sites = st.number_input("Number of Sites", min_value=1, max_value=20, value=10)

# Create input array
input_data = np.array([[age, total_purchase, account_manager, years, num_sites]])
input_scaled = scaler.transform(input_data)

# Predict and show results
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    prob_churn = model.predict_proba(input_scaled)[0][1]
    prob_stay = 1 - prob_churn

    # Pie chart
    fig = go.Figure(data=[go.Pie(
        labels=["Churn", "Stay"],
        values=[prob_churn, prob_stay],
        marker=dict(colors=["red", "green"]),
        hole=0.4,
        textinfo='label+percent'
    )])
    fig.update_layout(title="Churn Prediction Meter")

    st.plotly_chart(fig)

    if prediction[0] == 1:
        st.error(f"Customer is likely to CHURN (Churn Probability: {prob_churn:.2%})")
    else:
        st.success(f"Customer is likely to STAY (Churn Probability: {prob_churn:.2%})")