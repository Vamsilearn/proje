import streamlit as st
import numpy as np
import joblib

# Load the LSTM model
model = joblib.load('lstm_model.pkl')  # Make sure the .h5 file is in the same directory

st.title('Hotel Cancelation Rates Based on Economic Triggers')
st.write('Predictions utilizing stacking ensemble and LSTM')

Days_til_booking = st.number_input('Enter number of days between booking and check out')
Month_of_arrival = st.number_input('Enter the month of arrival (0-12)')
Gross_domestic_product = st.number_input('Enter GDP')
Interest_rate = st.number_input('Enter Interest Rate')
Inflation_chg = st.number_input('Enter Inflation Change')
Inflation = st.number_input('Enter Inflation')
CPI_avg = st.number_input('Enter CPI Average')
CPI_hotels = st.number_input('Enter CPI Hotels')
Fuel_prc = st.number_input('Enter average Fuel Price')
Unemployment_rate = st.number_input('Enter Unemployment Rate')

if st.button('Predict Cancelation'):
    # Order: time steps first, then features
    # Shape = (1, 3, 4) = batch size, time steps, features
    sequence = np.array([
        [gdp[0], interest_rate[0], inflation_chg[0], inflation[0]],
        [gdp[1], interest_rate[1], inflation_chg[1], inflation[1]],
        [gdp[2], interest_rate[2], inflation_chg[2], inflation[2]]
    ]).reshape(1, 3, 4)

    prediction = model.predict(sequence)[0][0]
    st.write(f'LSTM Prediction (probability of cancellation): {prediction:.2f}')
