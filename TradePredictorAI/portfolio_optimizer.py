import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Tuple
from scipy.optimize import minimize

def get_portfolio_data(symbols: List[str], period: str = '1y') -> pd.DataFrame:
    """
    Fetch historical data for multiple stocks
    """
    data = pd.DataFrame()
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)['Close']
        data[symbol] = hist
    return data

def calculate_portfolio_metrics(data: pd.DataFrame, weights: np.ndarray) -> Tuple[float, float]:
    """
    Calculate expected return and risk for a given portfolio
    """
    returns = data.pct_change()
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_risk

def optimize_portfolio(data: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict:
    """
    Optimize portfolio weights for maximum Sharpe ratio
    """
    num_assets = len(data.columns)
    returns = data.pct_change()
    
    def objective(weights):
        portfolio_return, portfolio_risk = calculate_portfolio_metrics(data, weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
        return -sharpe_ratio  # Minimize negative Sharpe ratio
    
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
    )
    bounds = tuple((0, 1) for _ in range(num_assets))  # Weights between 0 and 1
    
    initial_weights = np.array([1/num_assets] * num_assets)
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    optimized_weights = result.x
    opt_return, opt_risk = calculate_portfolio_metrics(data, optimized_weights)
    
    return {
        'weights': dict(zip(data.columns, optimized_weights)),
        'expected_return': opt_return * 100,  # Convert to percentage
        'risk': opt_risk * 100,  # Convert to percentage
        'sharpe_ratio': -result.fun  # Convert back to positive
    }

def generate_efficient_frontier(data: pd.DataFrame, num_portfolios: int = 100) -> Dict[str, List]:
    """
    Generate efficient frontier points
    """
    returns = []
    risks = []
    weights_list = []
    
    num_assets = len(data.columns)
    
    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        portfolio_return, portfolio_risk = calculate_portfolio_metrics(data, weights)
        
        returns.append(portfolio_return * 100)
        risks.append(portfolio_risk * 100)
        weights_list.append(weights)
    
    return {
        'returns': returns,
        'risks': risks,
        'weights': weights_list
    }
