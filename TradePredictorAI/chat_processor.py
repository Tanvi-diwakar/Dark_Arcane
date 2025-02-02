import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st
from typing import Dict, Any

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def process_user_query(query: str, 
                      df: pd.DataFrame, 
                      risk_metrics: Dict[str, float], 
                      indicators: Dict[str, float]) -> str:
    """
    Process user queries and return relevant responses
    """
    query = query.lower()
    tokens = word_tokenize(query)
    
    # Basic query matching
    if any(word in tokens for word in ['trend', 'direction']):
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        trend = "upward" if current_price > prev_price else "downward"
        return f"The current trend is {trend}. The RSI is {indicators['RSI']:.2f}, suggesting {'overbought' if indicators['RSI'] > 70 else 'oversold' if indicators['RSI'] < 30 else 'neutral'} conditions."
    
    elif any(word in tokens for word in ['risk', 'safe']):
        return f"Based on historical data, there is a {risk_metrics['profit_prob']:.1f}% probability of profit and a {risk_metrics['loss_prob']:.1f}% probability of loss. The volatility is {risk_metrics['volatility']:.2f}%."
    
    elif any(word in tokens for word in ['recommend', 'suggestion', 'advice']):
        if indicators['RSI'] > 70:
            return "The market appears overbought. Consider waiting for a pullback before entering long positions."
        elif indicators['RSI'] < 30:
            return "The market appears oversold. This might present buying opportunities, but manage your risk carefully."
        else:
            return "The market is in a neutral state. Focus on your trading plan and risk management rules."
    
    elif any(word in tokens for word in ['volume', 'liquidity']):
        avg_volume = df['Volume'].mean()
        current_volume = df['Volume'].iloc[-1]
        return f"The current volume is {current_volume:,.0f} compared to the average volume of {avg_volume:,.0f}."
    
    else:
        return "I can help you analyze trends, assess risks, and provide trading suggestions. Please ask specific questions about these topics."
