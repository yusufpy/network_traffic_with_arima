import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import datetime

st.title("Network Traffic Analysis and Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Ensure timestamp is a datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Summarize bytes transferred by timestamp
    df_summary = df.groupby('timestamp').sum()['bytes_transferred'].reset_index()

    # Set timestamp as index
    df_summary.set_index('timestamp', inplace=True)

    # Display raw data
    st.write("### Raw Data")
    st.dataframe(df)

    # Display summarized data
    st.write("### Summarized Data by Timestamp")
    st.dataframe(df_summary)

    # Plot the data
    st.write("### Bytes Transferred Over Time")
    plt.figure(figsize=(10, 6))
    plt.plot(df_summary.index, df_summary['bytes_transferred'], marker='o')
    plt.title("Bytes Transferred Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Bytes Transferred")
    st.pyplot(plt)

    # Train ARIMA model
    model = ARIMA(df_summary['bytes_transferred'], order=(1, 1, 1))
    model_fit = model.fit()

    # Make forecast
    forecast = model_fit.forecast(steps=5)
    forecast_index = [df_summary.index[-1] + datetime.timedelta(minutes=i) for i in range(1, 6)]

    # Plot forecast
    st.write("### ARIMA Forecast")
    plt.figure(figsize=(10, 6))
    plt.plot(df_summary.index, df_summary['bytes_transferred'], marker='o', label='Historical Data')
    plt.plot(forecast_index, forecast, marker='x', linestyle='--', color='red', label='Forecast')
    plt.title("ARIMA Forecast of Bytes Transferred")
    plt.xlabel("Timestamp")
    plt.ylabel("Bytes Transferred")
    plt.legend()
    st.pyplot(plt)

    # Display forecast values
    st.write("### Forecasted Values")
    forecast_df = pd.DataFrame({'timestamp': forecast_index, 'forecasted_bytes': forecast})
    st.dataframe(forecast_df)
else:
    st.write("Please upload a CSV file to proceed.")
