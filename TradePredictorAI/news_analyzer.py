import os
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def fetch_company_news(symbol: str, days: int = 7) -> List[Dict[str, Any]]:
    """
    Fetch recent news for a company using Yahoo Finance
    """
    try:
        stock = yf.Ticker(symbol)
        news = stock.news
        
        # Filter and format news
        recent_news = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for item in news:
            if datetime.fromtimestamp(item['providerPublishTime']) > cutoff_date:
                recent_news.append({
                    'title': item['title'],
                    'publisher': item['publisher'],
                    'link': item['link'],
                    'published_at': datetime.fromtimestamp(item['providerPublishTime']),
                    'type': item.get('type', 'general')
                })
        
        return recent_news
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []

def analyze_news_credibility(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze news credibility using AI
    """
    analyzed_news = []
    
    for item in news_items:
        try:
            prompt = f"""
            Analyze the credibility and impact of this financial news:
            Title: {item['title']}
            Publisher: {item['publisher']}
            Type: {item['type']}
            
            Please evaluate:
            1. Source credibility
            2. Potential market impact
            3. Correlation with market data
            4. Risk of misinformation
            5. Trading implications
            
            Respond in JSON format with these keys: credibility_score, impact_score, verification_status, key_points, trading_implications
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            analysis = response.choices[0].message.content
            item.update({
                'analysis': analysis,
                'analyzed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
        except Exception as e:
            item.update({
                'analysis_error': str(e),
                'analyzed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
        analyzed_news.append(item)
    
    return analyzed_news

def generate_market_impact_report(symbol: str, analyzed_news: List[Dict[str, Any]], stock_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive market impact report
    """
    try:
        # Prepare context for the AI
        price_change = ((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-5]) / stock_data['Close'].iloc[-5]) * 100
        avg_volume = stock_data['Volume'].mean()
        current_volume = stock_data['Volume'].iloc[-1]
        
        news_summary = "\n".join([f"- {item['title']} ({item['publisher']})" for item in analyzed_news[:5]])
        
        prompt = f"""
        Generate a comprehensive market impact report for {symbol} based on:

        Recent Price Action:
        - 5-day price change: {price_change:.2f}%
        - Current volume vs average: {current_volume/avg_volume:.2f}x

        Recent News:
        {news_summary}

        Please provide:
        1. Overall market sentiment
        2. News credibility assessment
        3. Potential market impact
        4. Risk factors
        5. Recommendations for traders

        Respond in JSON format with these keys: market_sentiment, credibility_assessment, potential_impact, risk_factors, trading_recommendations
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        report = response.choices[0].message.content
        return {
            "report": report,
            "news_count": len(analyzed_news),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "news_count": len(analyzed_news),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol
        }
