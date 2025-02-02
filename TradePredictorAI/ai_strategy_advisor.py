import os
from typing import Dict, Any
from openai import OpenAI
import pandas as pd
import json

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_trading_strategy(
    df: pd.DataFrame,
    technical_indicators: Dict[str, float],
    risk_metrics: Dict[str, float],
    symbol: str
) -> Dict[str, Any]:
    """
    Generate AI-powered trading strategy suggestions based on market data and analysis
    """
    # Prepare the context for the AI
    latest_price = df['Close'].iloc[-1]
    price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100
    volume_change = ((df['Volume'].iloc[-1] - df['Volume'].iloc[-5]) / df['Volume'].iloc[-5]) * 100
    
    prompt = f"""
    Analyze the following market data for {symbol} and provide a trading strategy recommendation:
    
    Current Price: ${latest_price:.2f}
    5-day Price Change: {price_change:.2f}%
    5-day Volume Change: {volume_change:.2f}%
    
    Technical Indicators:
    - RSI: {technical_indicators['RSI']:.2f}
    - MACD: {technical_indicators['MACD']:.2f}
    - Signal: {technical_indicators['Signal']:.2f}
    
    Risk Metrics:
    - Volatility: {risk_metrics['volatility']:.2f}%
    - Profit Probability: {risk_metrics['profit_prob']:.2f}%
    - Loss Probability: {risk_metrics['loss_prob']:.2f}%
    
    Please provide:
    1. A trading recommendation (BUY, SELL, or HOLD)
    2. Confidence level (0-1)
    3. Detailed rationale
    4. Risk considerations
    5. Suggested entry/exit points
    
    Respond in JSON format with these keys: recommendation, confidence, rationale, risks, entry_point, exit_point
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        strategy = json.loads(response.choices[0].message.content)
        return {
            "recommendation": strategy["recommendation"],
            "confidence": strategy["confidence"],
            "rationale": strategy["rationale"],
            "risks": strategy["risks"],
            "entry_point": strategy["entry_point"],
            "exit_point": strategy["exit_point"],
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {
            "error": f"Failed to generate strategy: {str(e)}",
            "recommendation": "HOLD",
            "confidence": 0.0,
            "rationale": "Error in strategy generation",
            "risks": "Unable to assess risks",
            "entry_point": None,
            "exit_point": None,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
