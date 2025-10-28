# File: polymarket_trader.py
"""
Polymarket Trading Bot - CSV Storage Version
Simple CSV-based data persistence
"""

import requests
import time
from datetime import datetime
from typing import Dict, List, Optional
import threading
import logging
import pandas as pd
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class PolymarketTrader:
    def __init__(self, config: Dict = None):
        """Initialize trading bot with CSV storage"""
        
        # Default config
        if config is None:
            config = {
                'large_order_threshold': 1000,
                'max_position_size': 5000,
                'markets_to_watch': []
            }
        
        self.config = config
        self.gamma_api = "https://gamma-api.polymarket.com"
        self.clob_api = "https://clob.polymarket.com"
        self.is_running = False
        self.positions = {}
        self.logs = []  # In-memory logs
        
        # Initialize CSV storage
        self.data_dir = Path('data')
        self.large_orders_file = self.data_dir / 'large_orders.csv'
        self.trades_file = self.data_dir / 'trades.csv'
        
        self.init_csv_storage()
        
        logging.info("Trading bot initialized with CSV storage")
    
    def init_csv_storage(self):
        """Initialize CSV files and directory"""
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize large_orders.csv if it doesn't exist
        if not self.large_orders_file.exists():
            df = pd.DataFrame(columns=[
                'timestamp', 'market_id', 'token_id', 'side', 
                'price', 'size', 'total_value', 'outcome'
            ])
            df.to_csv(self.large_orders_file, index=False)
            logging.info("✅ Created large_orders.csv")
        
        # Initialize trades.csv if it doesn't exist
        if not self.trades_file.exists():
            df = pd.DataFrame(columns=[
                'timestamp', 'market_id', 'token_id', 'side',
                'price', 'size', 'status'
            ])
            df.to_csv(self.trades_file, index=False)
            logging.info("✅ Created trades.csv")
        
        logging.info("📁 CSV storage initialized")
    
    def get_markets(self) -> List[Dict]:
        """Fetch markets to watch"""
        url = f"{self.gamma_api}/markets"
        params = {"limit": 20, "active": True}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching markets: {e}")
            return []
    
    def get_order_book(self, token_id: str) -> Optional[Dict]:
        """Get order book for a token"""
        url = f"{self.clob_api}/book"
        params = {"token_id": token_id}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching order book: {e}")
            return None
    
    def detect_large_orders(self, token_id: str, market_id: str, outcome: str):
        """Detect and log large orders"""
        book = self.get_order_book(token_id)
        
        if not book:
            return
        
        threshold = self.config.get('large_order_threshold', 1000)
        
        # Check bids and asks
        for side, orders in [('BUY', book.get('bids', [])), ('SELL', book.get('asks', []))]:
            for order in orders:
                price = float(order.get('price', 0))
                size = float(order.get('size', 0))
                value = price * size
                
                if value >= threshold:
                    self.log_large_order(
                        market_id, token_id, side, 
                        price, size, value, outcome
                    )
                    log_msg = f"🔔 Large {side} order: {outcome} - {size:.0f} @ ${price:.4f} (${value:,.0f})"
                    logging.info(log_msg)
                    self.logs.append(log_msg)
    
    def log_large_order(self, market_id, token_id, side, price, size, value, outcome):
        """Log large order to CSV"""
        order_data = {
            'timestamp': datetime.now().isoformat(),
            'market_id': market_id,
            'token_id': token_id,
            'side': side,
            'price': price,
            'size': size,
            'total_value': value,
            'outcome': outcome
        }
        
        try:
            # Append to CSV
            df = pd.DataFrame([order_data])
            df.to_csv(self.large_orders_file, mode='a', header=False, index=False)
        except Exception as e:
            logging.error(f"Error logging large order to CSV: {e}")
    
    def log_trade(self, market_id, token_id, side, price, size, status='completed'):
        """Log trade to CSV"""
        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'market_id': market_id,
            'token_id': token_id,
            'side': side,
            'price': price,
            'size': size,
            'status': status
        }
        
        try:
            # Append to CSV
            df = pd.DataFrame([trade_data])
            df.to_csv(self.trades_file, mode='a', header=False, index=False)
        except Exception as e:
            logging.error(f"Error logging trade to CSV: {e}")
    
    def get_current_price(self, token_id: str) -> Optional[float]:
        """Get current midpoint price"""
        url = f"{self.clob_api}/midpoint"
        params = {"token_id": token_id}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return float(data.get('mid', 0))
        except Exception as e:
            logging.error(f"Error fetching price: {e}")
            return None
    
    def monitoring_loop(self):
        """Main monitoring loop"""
        logging.info("🚀 Starting monitoring loop...")
        
        while self.is_running:
            try:
                markets = self.get_markets()
                
                for market in markets[:5]:  # Monitor top 5
                    if 'clobTokenIds' not in market:
                        continue
                    
                    market_id = market.get('id')
                    token_ids = market['clobTokenIds']
                    outcomes = market.get('outcomes', [])
                    question = market.get('question', 'N/A')
                    
                    for token_id, outcome in zip(token_ids, outcomes):
                        self.detect_large_orders(token_id, market_id, outcome)
                
                time.sleep(10)
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(30)
    
    def start(self):
        """Start the trading bot"""
        if self.is_running:
            logging.warning("Bot is already running")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logging.info("✅ Trading bot started")
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        logging.info("⏹️ Trading bot stopped")
    
    def get_stats(self) -> Dict:
        """Get trading statistics from CSV files"""
        try:
            # Read CSV files
            df_orders = pd.read_csv(self.large_orders_file)
            df_trades = pd.read_csv(self.trades_file)
            
            # Get counts
            total_orders = len(df_orders)
            total_trades = len(df_trades)
            
            # Get recent orders (last 20)
            recent_orders = df_orders.tail(20).to_dict('records') if not df_orders.empty else []
            
        except Exception as e:
            logging.error(f"Error reading stats from CSV: {e}")
            total_orders = 0
            total_trades = 0
            recent_orders = []
        
        return {
            'total_large_orders': total_orders,
            'total_trades': total_trades,
            'recent_orders': recent_orders,
            'is_running': self.is_running,
            'logs': self.logs[-50:]  # Last 50 logs
        }
