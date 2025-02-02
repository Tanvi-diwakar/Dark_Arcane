import pandas as pd
import numpy as np
from typing import Dict

def calculate_risk_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate risk metrics and probability of profit/loss
    """
    # Calculate daily returns
    returns = df['Close'].pct_change()
    
    # Calculate volatility (annualized)
    volatility = returns.std() * np.sqrt(252) * 100
    
    # Calculate probability metrics using historical distribution
    positive_days = (returns > 0).sum() / len(returns) * 100
    negative_days = (returns < 0).sum() / len(returns) * 100
    
    # Calculate risk-adjusted metrics
    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    return {
        'volatility': volatility,
        'profit_prob': positive_days,
        'loss_prob': negative_days,
        'sharpe_ratio': sharpe_ratio
    }

def calculate_position_size(df: pd.DataFrame, risk_percentage: float) -> float:
    """
    Calculate recommended position size based on risk tolerance
    """
    atr = calculate_atr(df)
    account_size = 100000  # Example account size
    risk_amount = account_size * (risk_percentage / 100)
    
    return risk_amount / atr

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average True Range
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    return true_range.rolling(period).mean().iloc[-1]
