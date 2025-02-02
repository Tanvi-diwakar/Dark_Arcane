import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from market_data import get_stock_data
from analysis_utils import calculate_technical_indicators
from risk_calculator import calculate_risk_metrics
from chat_processor import process_user_query
from portfolio_optimizer import get_portfolio_data, optimize_portfolio, generate_efficient_frontier
from ai_strategy_advisor import generate_trading_strategy
from news_analyzer import fetch_company_news, analyze_news_credibility, generate_market_impact_report
from voice_processor import voice_processor

# Page configuration
st.set_page_config(
    page_title="Trading Analysis Bot",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'portfolio_symbols' not in st.session_state:
    st.session_state.portfolio_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

def main():
    st.title("📊 Trading Analysis Bot")

    # Tabs for different views
    tab1, tab2 = st.tabs(["Single Stock Analysis", "Portfolio Optimization"])

    # Sidebar for user inputs
    st.sidebar.header("Analysis Parameters")

    with tab1:
        symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
        period = st.sidebar.selectbox(
            "Time Period",
            options=["1mo", "3mo", "6mo", "1y", "2y"],
            index=2
        )

        # Add chart type selector
        chart_type = st.sidebar.selectbox(
            "Chart Type",
            options=["Candlestick", "Line", "OHLC"],
            index=0
        )

        # Add technical indicator selector
        indicators_select = st.sidebar.multiselect(
            "Technical Indicators",
            options=["RSI", "MACD", "Bollinger Bands", "Moving Averages"],
            default=["RSI", "MACD"]
        )

        df = get_stock_data(symbol, period)

        if df is not None:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Enhanced price chart with multiple subplots
                fig = go.Figure()

                # Main price chart
                if chart_type == "Candlestick":
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name="Price"
                    ))
                elif chart_type == "Line":
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Close'],
                        mode='lines',
                        name="Price"
                    ))
                elif chart_type == "OHLC":
                    fig.add_trace(go.Ohlc(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name="Price"
                    ))

                # Add technical indicators based on selection
                if "Moving Averages" in indicators_select:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Close'].rolling(window=20).mean(),
                        mode='lines',
                        name="20 SMA",
                        line=dict(color='orange')
                    ))
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Close'].rolling(window=50).mean(),
                        mode='lines',
                        name="50 SMA",
                        line=dict(color='blue')
                    ))

                if "Bollinger Bands" in indicators_select:
                    sma = df['Close'].rolling(window=20).mean()
                    std = df['Close'].rolling(window=20).std()
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=sma + (2 * std),
                        mode='lines',
                        name="Upper BB",
                        line=dict(dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=sma - (2 * std),
                        mode='lines',
                        name="Lower BB",
                        line=dict(dash='dash')
                    ))

                # Update layout with better styling
                fig.update_layout(
                    title=f"{symbol} Price Chart",
                    yaxis_title="Price",
                    xaxis_title="Date",
                    height=600,
                    template="plotly_white",
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    margin=dict(l=0, r=0, t=30, b=0)
                )

                # Add range slider
                fig.update_xaxes(rangeslider_visible=True)

                st.plotly_chart(fig, use_container_width=True)

                # Technical indicators
                indicators = calculate_technical_indicators(df)
                st.subheader("Technical Analysis")

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("RSI", f"{indicators['RSI']:.2f}")
                with metric_col2:
                    st.metric("MACD", f"{indicators['MACD']:.2f}")
                with metric_col3:
                    st.metric("Signal", f"{indicators['Signal']:.2f}")

            with col2:
                # Risk metrics
                st.subheader("Risk Analysis")
                risk_metrics = calculate_risk_metrics(df)

                st.metric("Profit Probability", f"{risk_metrics['profit_prob']:.1f}%")
                st.metric("Loss Probability", f"{risk_metrics['loss_prob']:.1f}%")
                st.metric("Volatility", f"{risk_metrics['volatility']:.2f}%")

                # Add AI Strategy Recommendations
                st.subheader("🤖 AI Strategy Advisor")
                if st.button("Generate Trading Strategy"):
                    with st.spinner("Analyzing market data..."):
                        strategy = generate_trading_strategy(df, indicators, risk_metrics, symbol)

                        # Display recommendation
                        rec_color = {
                            "BUY": "green",
                            "SELL": "red",
                            "HOLD": "orange"
                        }.get(strategy["recommendation"], "gray")

                        st.markdown(f"**Recommendation:** ::{rec_color}[{strategy['recommendation']}]")
                        st.markdown(f"**Confidence:** {strategy['confidence']:.2%}")

                        # Create expandable sections for detailed analysis
                        with st.expander("View Detailed Analysis"):
                            st.markdown("### Rationale")
                            st.write(strategy["rationale"])

                            st.markdown("### Risk Considerations")
                            st.write(strategy["risks"])

                            st.markdown("### Entry/Exit Points")
                            st.write(f"📈 Entry Point: ${strategy['entry_point']}")
                            st.write(f"📉 Exit Point: ${strategy['exit_point']}")

                            st.markdown("### Generated at")
                            st.write(strategy["timestamp"])

                # Add News Analysis Section
                st.subheader("📰 Verified News Analysis")
                with st.spinner("Fetching and analyzing news..."):
                    news_items = fetch_company_news(symbol)

                    if news_items:
                        analyzed_news = analyze_news_credibility(news_items)
                        impact_report = generate_market_impact_report(symbol, analyzed_news, df)

                        # Display market impact summary
                        if "error" not in impact_report:
                            report = impact_report["report"]
                            st.markdown("### Market Impact Summary")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Market Sentiment", report["market_sentiment"])
                                st.metric("Credibility Score", f"{float(report['credibility_assessment']):.1f}/10")

                            with col2:
                                st.metric("News Count", impact_report["news_count"])
                                st.metric("Last Updated", impact_report["generated_at"])

                            # Display detailed news analysis
                            with st.expander("View Detailed News Analysis"):
                                st.markdown("### Key Findings")
                                st.write(report["potential_impact"])

                                st.markdown("### Risk Factors")
                                st.write(report["risk_factors"])

                                st.markdown("### Trading Recommendations")
                                st.write(report["trading_recommendations"])

                        # Display individual news items
                        st.markdown("### Recent News Items")
                        for item in analyzed_news:
                            with st.expander(f"{item['title']} ({item['published_at'].strftime('%Y-%m-%d')})"):
                                st.write(f"**Publisher:** {item['publisher']}")
                                st.write(f"**Published:** {item['published_at'].strftime('%Y-%m-%d %H:%M:%S')}")

                                if 'analysis' in item:
                                    analysis = item['analysis']
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Credibility Score", f"{float(analysis['credibility_score']):.1f}/10")
                                    with col2:
                                        st.metric("Market Impact", f"{float(analysis['impact_score']):.1f}/10")

                                    st.markdown("#### Key Points")
                                    st.write(analysis["key_points"])

                                    st.markdown("#### Trading Implications")
                                    st.write(analysis["trading_implications"])

                                st.markdown(f"[Read More]({item['link']})")
                    else:
                        st.warning("No recent news found for this company.")
                # Chatbot interface
                st.subheader("Analysis Assistant")

                # Add voice input option
                col1, col2 = st.columns([3, 1])
                with col1:
                    user_input = st.text_input("Ask me about the analysis:", key="user_input")
                with col2:
                    if st.button("🎤 Voice Input"):
                        user_input = voice_processor.listen_for_speech()
                        if user_input:
                            st.session_state.user_input = user_input

                if user_input:
                    response = process_user_query(user_input, df, risk_metrics, indicators)
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("Bot", response))

                    # Add voice response
                    if st.button("🔊 Listen to Response"):
                        voice_processor.text_to_speech(response)

                # Display chat history
                st.subheader("Chat History")
                for role, message in st.session_state.chat_history[-6:]:
                    if role == "You":
                        st.write(f"🧑 **You:** {message}")
                    else:
                        st.write(f"🤖 **Bot:** {message}")

    with tab2:
        st.subheader("Portfolio Optimization")

        # Portfolio symbols input
        portfolio_symbols = st.text_input(
            "Enter stock symbols (comma-separated)",
            value=",".join(st.session_state.portfolio_symbols)
        ).upper()
        st.session_state.portfolio_symbols = [s.strip() for s in portfolio_symbols.split(",")]

        optimize_button = st.button("Optimize Portfolio")

        if optimize_button:
            with st.spinner("Optimizing portfolio..."):
                # Get portfolio data
                portfolio_data = get_portfolio_data(st.session_state.portfolio_symbols)

                if not portfolio_data.empty:
                    # Optimize portfolio
                    optimization_result = optimize_portfolio(portfolio_data)

                    # Display optimization results
                    st.subheader("Optimal Portfolio Allocation")

                    # Create pie chart for weights
                    weights_fig = go.Figure(data=[go.Pie(
                        labels=list(optimization_result['weights'].keys()),
                        values=list(optimization_result['weights'].values()),
                        hole=0.3
                    )])
                    weights_fig.update_layout(title="Optimal Portfolio Weights")
                    st.plotly_chart(weights_fig, use_container_width=True)

                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Expected Return", f"{optimization_result['expected_return']:.2f}%")
                    with col2:
                        st.metric("Portfolio Risk", f"{optimization_result['risk']:.2f}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{optimization_result['sharpe_ratio']:.2f}")

                    # Generate and display efficient frontier
                    st.subheader("Efficient Frontier")
                    frontier_data = generate_efficient_frontier(portfolio_data)

                    frontier_fig = go.Figure()
                    frontier_fig.add_scatter(
                        x=frontier_data['risks'],
                        y=frontier_data['returns'],
                        mode='markers',
                        name='Possible Portfolios',
                        marker=dict(
                            size=5,
                            color='blue',
                            opacity=0.5
                        )
                    )
                    frontier_fig.add_scatter(
                        x=[optimization_result['risk']],
                        y=[optimization_result['expected_return']],
                        mode='markers',
                        name='Optimal Portfolio',
                        marker=dict(
                            size=15,
                            color='red',
                            symbol='star'
                        )
                    )
                    frontier_fig.update_layout(
                        title="Efficient Frontier",
                        xaxis_title="Portfolio Risk (%)",
                        yaxis_title="Expected Return (%)",
                        height=500
                    )
                    st.plotly_chart(frontier_fig, use_container_width=True)
                else:
                    st.error("Unable to fetch data for one or more symbols. Please check the symbols and try again.")

if __name__ == "__main__":
    main()