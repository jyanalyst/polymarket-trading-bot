# File: polymarket_trader.py
"""
Polymarket Trading Bot - CSV Storage Version
Simple CSV-based data persistence with fixed outcome parsing
"""

import time
from datetime import datetime
from typing import Dict, List, Optional
import threading
import logging
import pandas as pd
import os
import json
import requests
from pathlib import Path
from signal_generator import SignalGenerator, Signal
from py_clob_client.client import ClobClient

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
                'large_order_threshold': 100000,  # Minimum $100,000 for large orders
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

        # Initialize official Polymarket client
        self.client = ClobClient(self.clob_api)
        
        # Initialize CSV storage
        self.data_dir = Path('data')
        self.large_orders_file = self.data_dir / 'large_orders.csv'
        self.trades_file = self.data_dir / 'trades.csv'
        
        self.init_csv_storage()

        # Initialize signal generator
        signal_config = {
            'min_history_points': 10,
            'acceleration_threshold': 1000000,  # Much higher threshold for large acceleration values
            'velocity_threshold': 1000,         # Higher velocity threshold
            'liquidity_min': self.config.get('min_liquidity', 1000),
            'liquidity_medium': 5000,
            'price_extreme_threshold': 0.85,
            'alert_cooldown_minutes': 5,
        }
        self.signal_generator = SignalGenerator(signal_config)
        self.signals = []  # Store recent signals

        logging.info("Trading bot initialized with CSV storage and signal generation")
    
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
            logging.info("Created large_orders.csv")
        
        # Initialize trades.csv if it doesn't exist
        if not self.trades_file.exists():
            df = pd.DataFrame(columns=[
                'timestamp', 'market_id', 'token_id', 'side',
                'price', 'size', 'status'
            ])
            df.to_csv(self.trades_file, index=False)
            logging.info("Created trades.csv")
        
            logging.info("CSV storage initialized")
    
    def get_markets(self) -> List[Dict]:
        """Fetch markets to watch - focusing on CLOB-tradable markets"""
        try:
            # Use full Gamma Markets API for complete market data
            url = f"{self.gamma_api}/markets"
            params = {
                "closed": "false",  # Only get open markets
                "limit": 100
            }

            response = requests.get(url, params=params, timeout=10, verify=False)
            response.raise_for_status()
            markets = response.json()

            # Debug: Print first market's structure
            if self.debug_mode and markets and len(markets) > 0:
                sample_market = markets[0]
                logging.info(f"Sample market structure: {list(sample_market.keys())}")
                logging.info(f"Sample market data: {sample_market}")
                self.debug_mode = False  # Only log once

            # Filter for CLOB-tradable markets
            tradable_markets = []

            for market in markets:
                try:
                    # Key criteria for CLOB trading (per official docs)
                    has_order_book = market.get('enableOrderBook', False)
                    is_active = market.get('active', False)
                    is_open = not market.get('closed', True)
                    has_tokens = 'clobTokenIds' in market and market['clobTokenIds']

                    if has_order_book and is_active and is_open and has_tokens:
                        tradable_markets.append(market)

                        # Log first few tradable markets
                        if len(tradable_markets) <= 3:
                            market_id = market.get('id', 'unknown')[:8]
                            question = market.get('question', 'N/A')[:40]
                            logging.info(f"Tradable market: {market_id}... '{question}...'")

                except (ValueError, TypeError):
                    continue

            if not tradable_markets:
                logging.warning("No CLOB-tradable markets found")
            else:
                logging.info(f"Found {len(tradable_markets)} CLOB-tradable markets")

            return tradable_markets

        except Exception as e:
            logging.error(f"Error fetching markets: {e}")
            return []
    
    def get_order_book(self, token_id: str) -> Optional[Dict]:
        """Get order book for a token"""
        # Validate token_id
        if not token_id or not isinstance(token_id, str) or len(token_id) < 10:
            logging.warning(f"Invalid token_id: {token_id}")
            return None

        try:
            # Use official client method
            return self.client.get_order_book(token_id)
        except Exception as e:
            logging.error(f"Error fetching order book: {e}")
            return None
    
    def extract_token_ids(self, market: Dict) -> List[str]:
        """
        Extract valid token IDs from market data.
        Handles official py-clob-client format: tokens as list of dicts.
        """
        token_ids = []

        # Check for tokens field (official client format)
        if 'tokens' in market and isinstance(market['tokens'], list):
            for token_dict in market['tokens']:
                if isinstance(token_dict, dict) and 'token_id' in token_dict:
                    token_id = token_dict['token_id']
                    if isinstance(token_id, str) and len(token_id) > 10:
                        token_ids.append(token_id)

        # Fallback: Try legacy formats
        if not token_ids:
            possible_fields = ['clobTokenIds', 'tokenIds']

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
        Extract outcomes from market data.
        Handles official py-clob-client format: outcomes embedded in tokens.
        """
        outcomes = []

        # Check for tokens field (official client format)
        if 'tokens' in market and isinstance(market['tokens'], list):
            for token_dict in market['tokens']:
                if isinstance(token_dict, dict) and 'outcome' in token_dict:
                    outcome = token_dict['outcome']
                    if outcome:
                        outcomes.append(str(outcome))

        # Fallback: Try legacy outcomes field
        if not outcomes:
            outcomes_data = market.get('outcomes', [])

            # If it's a string, try to parse as JSON
            if isinstance(outcomes_data, str):
                try:
                    parsed = json.loads(outcomes_data)
                    if isinstance(parsed, list):
                        return [str(o) for o in parsed]
                except json.JSONDecodeError:
                    pass

            # If it's already a list
            elif isinstance(outcomes_data, list) and outcomes_data:
                return [str(o) for o in outcomes_data]

        # Final fallback: Generate outcome names based on token count
        if not outcomes:
            token_ids = self.extract_token_ids(market)
            if token_ids:
                outcomes = [f"Outcome {i+1}" for i in range(len(token_ids))]

        return outcomes if outcomes else ['Unknown']
    
    def detect_large_orders(self, token_id: str, market_id: str, outcome: str, question: str):
        """Detect and log large orders"""
        book = self.get_order_book(token_id)
        
        if not book:
            return
        
        threshold = self.config.get('large_order_threshold', 1000)
        
        # Check bids and asks (OrderBookSummary object has .bids and .asks attributes)
        for side, orders in [('BUY', book.bids or []), ('SELL', book.asks or [])]:
            for order in orders:
                try:
                    # OrderSummary objects have .price and .size attributes
                    price = float(order.price)
                    size = float(order.size)
                    value = price * size

                    if value >= threshold:
                        self.log_large_order(
                            market_id, token_id, side,
                            price, size, value, outcome, question
                        )
                        log_msg = f"Large {side} order: {outcome} - {size:.0f} @ ${price:.4f} (${value:,.0f}) [{question[:40]}...]"
                        logging.info(log_msg)
                        self.logs.append(log_msg)
                except (ValueError, TypeError, AttributeError) as e:
                    logging.warning(f"Error parsing order: {e}")
                    continue
    
    def log_large_order(self, market_id, token_id, side, price, size, value, outcome, question):
        """Log large order to CSV and update volume tracking"""
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

            # Update volume tracking for signal generation
            self.signal_generator.update_volume_data(token_id, value)

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
        try:
            # Use official client method
            return self.client.get_midpoint(token_id)
        except Exception as e:
            logging.error(f"Error fetching price: {e}")
            return None

    def get_liquidity(self, token_id: str) -> float:
        """Calculate total liquidity from order book"""
        book = self.get_order_book(token_id)
        if not book:
            return 0.0

        total_liquidity = 0.0

        # Sum bid liquidity (OrderBookSummary object has .bids and .asks attributes)
        for bid in (book.bids or []):
            try:
                # OrderSummary objects have .price and .size attributes
                price = float(bid.price)
                size = float(bid.size)
                total_liquidity += price * size
            except (ValueError, TypeError, AttributeError):
                continue

        # Sum ask liquidity
        for ask in (book.asks or []):
            try:
                # OrderSummary objects have .price and .size attributes
                price = float(ask.price)
                size = float(ask.size)
                total_liquidity += price * size
            except (ValueError, TypeError, AttributeError):
                continue

        return total_liquidity

    def collect_price_data(self, token_id: str, market_id: str, outcome: str, question: str):
        """Collect price and liquidity data for signal generation"""
        price = self.get_current_price(token_id)
        liquidity = self.get_liquidity(token_id)

        if price is not None and liquidity > 0:
            # Update signal generator with new data
            self.signal_generator.update_price_data(token_id, price, liquidity)

            # Try to generate signals
            signals = self.signal_generator.generate_signals(token_id, market_id, outcome, question)

            # Process any signals generated
            for signal in signals:
                self.process_signal(signal)

    def process_signal(self, signal: Signal):
        """Process a generated signal (log and alert)"""
        # Add to recent signals list
        self.signals.append(signal.to_dict())
        if len(self.signals) > 100:  # Keep last 100 signals
            self.signals = self.signals[-100:]

        # Create alert message
        alert_msg = self.format_signal_alert(signal)
        logging.info(f"SIGNAL: {alert_msg}")
        self.logs.append(f"SIGNAL: {alert_msg}")

        # Here you would add Telegram/email alerts
        # self.send_telegram_alert(signal)
        # self.send_email_alert(signal)

    def format_signal_alert(self, signal: Signal) -> str:
        """Format signal for alert display"""
        return (f"{signal.type.value} - {signal.outcome} "
                f"@ ${signal.price:.4f} | Acc: {signal.acceleration:.6f} | "
                f"Vel: {signal.velocity:.6f} | Conf: {signal.confidence} | "
                f"{signal.question[:50]}...")

    def get_recent_signals(self, limit: int = 20) -> List[Dict]:
        """Get recent signals for dashboard"""
        return self.signals[-limit:] if self.signals else []
    
    def monitoring_loop(self):
        """Main monitoring loop"""
        logging.info("Starting monitoring loop...")
        
        loop_count = 0
        
        while self.is_running:
            try:
                markets = self.get_markets()
                
                if not markets:
                    logging.warning("No active markets found, waiting 60 seconds...")
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
                    
                    market_id = market.get('condition_id', 'unknown')
                    question = market.get('question', f'Market {market_id[:8]}...')
                    
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

                        # Collect price data for signal generation
                        self.collect_price_data(token_id, market_id, outcome, question)

                        markets_processed += 1
                
                loop_count += 1
                if loop_count % 6 == 0:  # Log every 60 seconds (6 loops * 10 sec)
                    logging.info(f"Monitoring active - processed {markets_processed} tokens, found {orders_found} large orders this cycle")
                
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
        
        logging.info("Trading bot started")
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        logging.info("Trading bot stopped")
    
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
            'recent_signals': self.get_recent_signals(20),
            'is_running': self.is_running,
            'logs': self.logs[-50:]  # Last 50 logs
        }
