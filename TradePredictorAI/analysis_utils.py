import pandas as pd
import numpy as np
from typing import Dict

def calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate technical indicators for the given stock data
    """
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    return {
        'RSI': rsi.iloc[-1],
        'MACD': macd.iloc[-1],
        'Signal': signal.iloc[-1]
    }

def calculate_trend(df: pd.DataFrame) -> str:
    """
    Determine the current trend
    """
    sma20 = df['Close'].rolling(window=20).mean()
    sma50 = df['Close'].rolling(window=50).mean()
    
    if sma20.iloc[-1] > sma50.iloc[-1]:
        return "Uptrend"
    elif sma20.iloc[-1] < sma50.iloc[-1]:
        return "Downtrend"
    else:
        return "Sideways"
