# File: dashboard.py
"""
Polymarket Trading Dashboard
Simple GUI to control trading bot and monitor activity
CSV-based storage version
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from pathlib import Path
from polymarket_trader import PolymarketTrader

# Page config
st.set_page_config(
    page_title="Polymarket Trading Bot",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'bot' not in st.session_state:
    config = {
        'large_order_threshold': 1000,
        'max_position_size': 5000,
        'markets_to_watch': []
    }
    st.session_state.bot = PolymarketTrader(config)
    st.session_state.bot_running = False

# Define data files
data_dir = Path('data')
large_orders_file = data_dir / 'large_orders.csv'
trades_file = data_dir / 'trades.csv'

# Title
st.title("📈 Polymarket Trading Bot")
st.markdown("---")

# Sidebar - Controls
with st.sidebar:
    st.header("⚙️ Bot Controls")
    
    # Start/Stop button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🟢 START", use_container_width=True, type="primary"):
            if not st.session_state.bot_running:
                st.session_state.bot.start()
                st.session_state.bot_running = True
                st.success("Bot started!")
                st.rerun()
    
    with col2:
        if st.button("🔴 STOP", use_container_width=True):
            if st.session_state.bot_running:
                st.session_state.bot.stop()
                st.session_state.bot_running = False
                st.warning("Bot stopped!")
                st.rerun()
    
    # Status indicator
    if st.session_state.bot_running:
        st.success("✅ Bot is RUNNING")
    else:
        st.error("⏹️ Bot is STOPPED")
    
    st.markdown("---")
    
    # Configuration
    st.header("🔧 Configuration")
    
    large_order_threshold = st.number_input(
        "Large Order Threshold ($)",
        min_value=100,
        max_value=100000,
        value=1000,
        step=100,
        help="Minimum order size to track"
    )
    
    max_position = st.number_input(
        "Max Position Size ($)",
        min_value=100,
        max_value=50000,
        value=5000,
        step=500,
        help="Maximum USDC per position"
    )
    
    if st.button("💾 Save Config"):
        st.session_state.bot.config['large_order_threshold'] = large_order_threshold
        st.session_state.bot.config['max_position_size'] = max_position
        st.success("Config saved!")
    
    st.markdown("---")
    
    # Refresh rate
    refresh_rate = st.slider(
        "Auto-refresh (seconds)",
        min_value=5,
        max_value=60,
        value=10,
        help="Dashboard update frequency"
    )
    
    if st.button("🔄 Refresh Now"):
        st.rerun()

# Main dashboard
stats = st.session_state.bot.get_stats()

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Large Orders Detected",
        stats['total_large_orders'],
        delta=None
    )

with col2:
    st.metric(
        "Trades Executed",
        stats['total_trades'],
        delta=None
    )

with col3:
    st.metric(
        "Bot Status",
        "RUNNING" if stats['is_running'] else "STOPPED",
        delta=None
    )

with col4:
    st.metric(
        "Signals Generated",
        len(stats.get('recent_signals', [])),
        delta=None
    )

st.markdown("---")

# Two columns for tables
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔔 Recent Large Orders")
    
    # Read from CSV
    try:
        if large_orders_file.exists():
            df_orders = pd.read_csv(large_orders_file)
            
            if not df_orders.empty:
                # Get last 20 orders
                df_orders = df_orders.tail(20).sort_values('timestamp', ascending=False)
                
                # Format columns for display - include question and outcome
                df_display = df_orders[['timestamp', 'question', 'outcome', 'side', 'price', 'size', 'total_value']].copy()
                df_display.columns = ['Time', 'Market Question', 'Outcome', 'Side', 'Price', 'Size', 'Value']

                # Format timestamp
                df_display['Time'] = pd.to_datetime(df_display['Time']).dt.strftime('%H:%M:%S')
                df_display['Market Question'] = df_display['Market Question'].apply(lambda x: str(x)[:30] + '...' if len(str(x)) > 30 else str(x))
                df_display['Outcome'] = df_display['Outcome'].apply(lambda x: str(x)[:15] if pd.notna(x) else 'N/A')
                df_display['Price'] = df_display['Price'].apply(lambda x: f"${x:.4f}")
                df_display['Size'] = df_display['Size'].apply(lambda x: f"{x:.0f}")
                df_display['Value'] = df_display['Value'].apply(lambda x: f"${x:,.0f}")
                
                st.dataframe(
                    df_display,
                    hide_index=True,
                    width='stretch'
                )
            else:
                st.info("No large orders detected yet")
        else:
            st.info("No large orders detected yet")
    except Exception as e:
        st.error(f"Error loading orders: {e}")

with col2:
    st.subheader("💼 Recent Trades")
    
    # Read from CSV
    try:
        if trades_file.exists():
            df_trades = pd.read_csv(trades_file)
            
            if not df_trades.empty:
                # Get last 20 trades
                df_trades = df_trades.tail(20).sort_values('timestamp', ascending=False)
                
                # Format columns for display
                df_display = df_trades[['timestamp', 'side', 'price', 'size', 'status']].copy()
                df_display.columns = ['Time', 'Side', 'Price', 'Size', 'Status']
                
                # Format timestamp
                df_display['Time'] = pd.to_datetime(df_display['Time']).dt.strftime('%H:%M:%S')
                df_display['Price'] = df_display['Price'].apply(lambda x: f"${x:.4f}")
                df_display['Size'] = df_display['Size'].apply(lambda x: f"{x:.0f}")
                
                st.dataframe(
                    df_display,
                    hide_index=True,
                    width='stretch'
                )
            else:
                st.info("No trades executed yet")
        else:
            st.info("No trades executed yet")
    except Exception as e:
        st.error(f"Error loading trades: {e}")

st.markdown("---")

# Signals section
st.subheader("🚨 Trading Signals")

try:
    recent_signals = stats.get('recent_signals', [])

    if recent_signals:
        # Convert to DataFrame for better display
        df_signals = pd.DataFrame(recent_signals)

        if not df_signals.empty:
            # Format the dataframe
            df_display = df_signals[['timestamp', 'type', 'strength', 'outcome', 'price', 'acceleration', 'velocity', 'confidence']].copy()
            df_display.columns = ['Time', 'Signal Type', 'Strength', 'Outcome', 'Price', 'Acceleration', 'Velocity', 'Confidence']

            # Format timestamp
            df_display['Time'] = pd.to_datetime(df_display['Time']).dt.strftime('%H:%M:%S')
            df_display['Price'] = df_display['Price'].apply(lambda x: f"${x:.4f}")
            df_display['Acceleration'] = df_display['Acceleration'].apply(lambda x: f"{x:.6f}")
            df_display['Velocity'] = df_display['Velocity'].apply(lambda x: f"{x:.6f}")

            # Add color coding for signal types
            def color_signal_type(val):
                if val == 'STRONG_BUY':
                    return 'background-color: #d4edda; color: #155724'
                elif val == 'STRONG_SELL':
                    return 'background-color: #f8d7da; color: #721c24'
                elif val == 'REVERSAL_WARNING':
                    return 'background-color: #fff3cd; color: #856404'
                elif val == 'ACCELERATION_SPIKE':
                    return 'background-color: #f8f9fa; color: #495057; font-weight: bold'
                return ''

            styled_df = df_display.style.applymap(color_signal_type, subset=['Signal Type'])

            st.dataframe(
                styled_df,
                hide_index=True,
                width='stretch'
            )

            # Show signal details in expandable sections
            with st.expander("📋 Signal Details", expanded=False):
                for i, signal in enumerate(recent_signals[-5:]):  # Show last 5 signals
                    st.markdown(f"**Signal {i+1}: {signal['type']} - {signal['outcome']}**")
                    st.write(f"- Price: ${signal['price']:.4f}")
                    st.write(f"- Acceleration: {signal['acceleration']:.6f}")
                    st.write(f"- Velocity: {signal['velocity']:.6f}")
                    st.write(f"- Confidence: {signal['confidence']}")
                    st.write(f"- Reason: {signal['reason']}")
                    st.write(f"- Question: {signal['question'][:100]}...")
                    if i < len(recent_signals[-5:]) - 1:
                        st.markdown("---")
        else:
            st.info("No signals generated yet")
    else:
        st.info("No signals generated yet")

except Exception as e:
    st.error(f"Error loading signals: {e}")

st.markdown("---")

# Charts
st.subheader("📊 Analytics")

col1, col2 = st.columns(2)

with col1:
    # Order volume chart
    try:
        if large_orders_file.exists():
            df_orders = pd.read_csv(large_orders_file)
            
            if not df_orders.empty:
                # Convert timestamp to date
                df_orders['date'] = pd.to_datetime(df_orders['timestamp']).dt.date
                
                # Group by date
                df_chart = df_orders.groupby('date').agg({
                    'timestamp': 'count',
                    'total_value': 'sum'
                }).reset_index()
                df_chart.columns = ['date', 'count', 'total_value']
                
                # Sort and limit to last 30 days
                df_chart = df_chart.sort_values('date', ascending=False).head(30)
                
                fig = px.bar(
                    df_chart,
                    x='date',
                    y='count',
                    title='Large Orders per Day',
                    labels={'count': 'Number of Orders', 'date': 'Date'}
                )
                st.plotly_chart(fig)
            else:
                st.info("Not enough data for chart")
        else:
            st.info("Not enough data for chart")
    except Exception as e:
        st.error(f"Error creating chart: {e}")

with col2:
    # Value chart
    try:
        if large_orders_file.exists():
            df_orders = pd.read_csv(large_orders_file)
            
            if not df_orders.empty:
                # Convert timestamp to date
                df_orders['date'] = pd.to_datetime(df_orders['timestamp']).dt.date
                
                # Group by date
                df_chart = df_orders.groupby('date').agg({
                    'total_value': 'sum'
                }).reset_index()
                df_chart.columns = ['date', 'total_value']
                
                # Sort and limit to last 30 days
                df_chart = df_chart.sort_values('date', ascending=False).head(30)
                
                fig = px.line(
                    df_chart,
                    x='date',
                    y='total_value',
                    title='Total Order Value per Day',
                    labels={'total_value': 'Value ($)', 'date': 'Date'}
                )
                st.plotly_chart(fig)
            else:
                st.info("Not enough data for chart")
        else:
            st.info("Not enough data for chart")
    except Exception as e:
        st.error(f"Error creating chart: {e}")

# Logs section
st.markdown("---")
st.subheader("📝 Recent Logs")

# Read last 50 lines from log file
try:
    log_file = Path('trading_bot.log')
    if log_file.exists():
        with open(log_file, 'r') as f:
            logs = f.readlines()[-50:]
        
        log_text = ''.join(logs)
        st.text_area(
            "Log Output",
            log_text,
            height=200,
            disabled=True
        )
    else:
        st.info("No logs available yet")
except Exception as e:
    st.error(f"Error loading logs: {e}")

# Auto-refresh
if st.session_state.bot_running:
    time.sleep(refresh_rate)
    st.rerun()

# Footer
st.markdown("---")
st.caption("Polymarket Trading Bot v1.0 | Built with Streamlit | CSV Storage")
