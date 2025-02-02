import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_data(symbol: str, period: str) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance
    """
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        if df.empty:
            st.error(f"No data found for symbol {symbol}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None
