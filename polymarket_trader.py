# File: polymarket_trader.py
"""
Polymarket Trading Bot - CSV Storage Version
Simple CSV-based data persistence with fixed outcome parsing
"""

import requests
import time
from datetime import datetime
from typing import Dict, List, Optional
import threading
import logging
import pandas as pd
import os
import json
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
                'markets_to_watch': [],
                'min_liquidity': 1000  # Minimum liquidity to consider
            }
        
        self.config = config
        self.gamma_api = "https://gamma-api.polymarket.com"
        self.clob_api = "https://clob.polymarket.com"
        self.is_running = False
        self.positions = {}
        self.logs = []  # In-memory logs
        self.debug_mode = True  # Enable debug logging
        
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
                'price', 'size', 'total_value', 'outcome', 'question'
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
        """Fetch markets to watch - focusing on high liquidity, active markets"""
        url = f"{self.gamma_api}/markets"
        
        # Try to get markets sorted by liquidity/volume
        params = {
            "limit": 100,  # Get more markets to filter from
            "active": "true",
            "closed": "false"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            markets = response.json()
            
            # Debug: Print first market's structure (only outcomes field)
            if self.debug_mode and markets and len(markets) > 0:
                sample_market = markets[0]
                logging.info(f"📊 Sample outcomes field: {sample_market.get('outcomes')}")
                logging.info(f"📊 Sample outcomes type: {type(sample_market.get('outcomes'))}")
                self.debug_mode = False  # Only log once
            
            # Filter for markets with liquidity
            min_liquidity = self.config.get('min_liquidity', 1000)
            active_markets = []
            
            for market in markets:
                try:
                    liquidity = float(market.get('liquidity', 0))
                    
                    # Check if market has clobTokenIds
                    if liquidity > min_liquidity and 'clobTokenIds' in market:
                        active_markets.append(market)
                        
                        # Log first few active markets
                        if len(active_markets) <= 3:
                            logging.info(f"✅ Active market: {market.get('question', 'N/A')[:60]}... (Liquidity: ${liquidity:,.0f})")
                
                except (ValueError, TypeError):
                    continue
            
            if not active_markets:
                logging.warning(f"⚠️ No active markets with liquidity > ${min_liquidity} found")
            else:
                logging.info(f"📈 Found {len(active_markets)} active markets with sufficient liquidity")
            
            return active_markets
            
        except Exception as e:
            logging.error(f"Error fetching markets: {e}")
            return []
    
    def get_order_book(self, token_id: str) -> Optional[Dict]:
        """Get order book for a token"""
        # Validate token_id
        if not token_id or not isinstance(token_id, str) or len(token_id) < 10:
            logging.warning(f"⚠️ Invalid token_id: {token_id}")
            return None
        
        url = f"{self.clob_api}/book"
        params = {"token_id": token_id}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logging.debug(f"Order book not found for token: {token_id[:20]}...")
            else:
                logging.error(f"HTTP error fetching order book: {e}")
            return None
        except Exception as e:
            logging.error(f"Error fetching order book: {e}")
            return None
    
    def extract_token_ids(self, market: Dict) -> List[str]:
        """
        Extract valid token IDs from market data.
        Handles different API response formats.
        """
        token_ids = []
        
        # Try different possible field names and formats
        possible_fields = ['clobTokenIds', 'tokens', 'tokenIds']
        
        for field in possible_fields:
            if field in market:
                data = market[field]
                
                # If it's a string, try to parse as JSON
                if isinstance(data, str):
                    try:
                        parsed = json.loads(data)
                        if isinstance(parsed, list):
                            token_ids = parsed
                            break
                    except json.JSONDecodeError:
                        continue
                
                # If it's already a list
                elif isinstance(data, list):
                    token_ids = data
                    break
        
        # Validate token IDs (should be alphanumeric strings > 10 chars)
        valid_token_ids = []
        for tid in token_ids:
            if isinstance(tid, str) and len(tid) > 10 and tid.replace('-', '').replace('_', '').isalnum():
                valid_token_ids.append(tid)
            else:
                logging.debug(f"Skipping invalid token_id: {tid}")
        
        return valid_token_ids
    
    def extract_outcomes(self, market: Dict) -> List[str]:
        """
        Extract outcomes from market data, handling JSON strings.
        """
        outcomes_data = market.get('outcomes', [])
        
        # If it's a string, try to parse as JSON
        if isinstance(outcomes_data, str):
            try:
                parsed = json.loads(outcomes_data)
                if isinstance(parsed, list):
                    return [str(o) for o in parsed]
            except json.JSONDecodeError:
                return ['Unknown']
        
        # If it's already a list
        elif isinstance(outcomes_data, list):
            return [str(o) for o in outcomes_data]
        
        return ['Unknown']
    
    def detect_large_orders(self, token_id: str, market_id: str, outcome: str, question: str):
        """Detect and log large orders"""
        book = self.get_order_book(token_id)
        
        if not book:
            return
        
        threshold = self.config.get('large_order_threshold', 1000)
        
        # Check bids and asks
        for side, orders in [('BUY', book.get('bids', [])), ('SELL', book.get('asks', []))]:
            for order in orders:
                try:
                    price = float(order.get('price', 0))
                    size = float(order.get('size', 0))
                    value = price * size
                    
                    if value >= threshold:
                        self.log_large_order(
                            market_id, token_id, side, 
                            price, size, value, outcome, question
                        )
                        log_msg = f"🔔 Large {side} order: {outcome} - {size:.0f} @ ${price:.4f} (${value:,.0f}) [{question[:40]}...]"
                        logging.info(log_msg)
                        self.logs.append(log_msg)
                except (ValueError, TypeError) as e:
                    logging.warning(f"Error parsing order: {e}")
                    continue
    
    def log_large_order(self, market_id, token_id, side, price, size, value, outcome, question):
        """Log large order to CSV"""
        order_data = {
            'timestamp': datetime.now().isoformat(),
            'market_id': market_id,
            'token_id': token_id,
            'side': side,
            'price': price,
            'size': size,
            'total_value': value,
            'outcome': outcome,
            'question': question
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
        
        loop_count = 0
        
        while self.is_running:
            try:
                markets = self.get_markets()
                
                if not markets:
                    logging.warning("⚠️ No active markets found, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                
                markets_processed = 0
                orders_found = 0
                
                for market in markets[:10]:  # Monitor top 10 active markets
                    # Extract token IDs and outcomes
                    token_ids = self.extract_token_ids(market)
                    outcomes = self.extract_outcomes(market)
                    
                    if not token_ids:
                        continue
                    
                    market_id = market.get('id', 'unknown')
                    question = market.get('question', 'N/A')
                    
                    # Ensure we have matching outcomes for token_ids
                    if len(outcomes) < len(token_ids):
                        outcomes.extend(['Unknown'] * (len(token_ids) - len(outcomes)))
                    
                    # Process each token
                    for token_id, outcome in zip(token_ids, outcomes):
                        orders_before = len(self.logs)
                        self.detect_large_orders(token_id, market_id, outcome, question)
                        orders_after = len(self.logs)
                        
                        if orders_after > orders_before:
                            orders_found += (orders_after - orders_before)
                        
                        markets_processed += 1
                
                loop_count += 1
                if loop_count % 6 == 0:  # Log every 60 seconds (6 loops * 10 sec)
                    logging.info(f"✅ Monitoring active - processed {markets_processed} tokens, found {orders_found} large orders this cycle")
                
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