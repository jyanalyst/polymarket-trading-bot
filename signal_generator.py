"""
Signal Generator for Polymarket Trading Bot
Price-based acceleration signals with liquidity filtering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

class SignalStrength(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"
    REVERSAL_WARNING = "REVERSAL_WARNING"
    ACCELERATION_SPIKE = "ACCELERATION_SPIKE"
    MOMENTUM_FADE = "MOMENTUM_FADE"
    VOLUME_SPIKE = "VOLUME_SPIKE"

@dataclass
class Signal:
    """Trading signal data structure"""
    type: SignalType
    strength: SignalStrength
    market_id: str
    token_id: str
    outcome: str
    question: str
    price: float
    velocity: float
    acceleration: float
    liquidity: float
    timestamp: datetime
    reason: str
    confidence: str

    def to_dict(self) -> Dict:
        return {
            'type': self.type.value,
            'strength': self.strength.value,
            'market_id': self.market_id,
            'token_id': self.token_id,
            'outcome': self.outcome,
            'question': self.question,
            'price': self.price,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'liquidity': self.liquidity,
            'timestamp': self.timestamp.isoformat(),
            'reason': self.reason,
            'confidence': self.confidence
        }

class PriceDataCollector:
    """Collects and stores price data for signal calculation"""

    def __init__(self, max_history: int = 100):
        """
        Initialize price data collector

        Args:
            max_history: Maximum number of price points to keep per token
        """
        self.max_history = max_history
        self.price_history = {}  # token_id -> list of (timestamp, price, liquidity)
        self.logger = logging.getLogger(__name__)

    def add_price_point(self, token_id: str, price: float, liquidity: float):
        """Add a new price point for a token"""
        if token_id not in self.price_history:
            self.price_history[token_id] = []

        timestamp = datetime.now()
        self.price_history[token_id].append((timestamp, price, liquidity))

        # Keep only recent history
        if len(self.price_history[token_id]) > self.max_history:
            self.price_history[token_id] = self.price_history[token_id][-self.max_history:]

    def get_price_series(self, token_id: str, min_points: int = 10) -> Optional[List[Tuple[datetime, float, float]]]:
        """Get price series for a token (timestamp, price, liquidity)"""
        if token_id not in self.price_history:
            return None

        series = self.price_history[token_id]
        if len(series) < min_points:
            return None

        return series

    def get_current_data(self, token_id: str) -> Optional[Tuple[float, float]]:
        """Get current price and liquidity for a token"""
        series = self.get_price_series(token_id, min_points=1)
        if not series:
            return None
        return series[-1][1], series[-1][2]  # price, liquidity

class VolumeTracker:
    """Tracks volume data for relative volume signal generation"""

    def __init__(self, window_minutes: int = 10, baseline_hours: int = 24, spike_threshold: int = 100):
        """
        Initialize volume tracker

        Args:
            window_minutes: Size of current activity window in minutes
            baseline_hours: Hours of historical data to use for baseline
            spike_threshold: Multiplier for volume spike detection
        """
        self.window_minutes = window_minutes
        self.baseline_hours = baseline_hours
        self.spike_threshold = spike_threshold
        self.volume_windows = {}  # token_id -> list of {'start': datetime, 'volume': float}
        self.logger = logging.getLogger(__name__)

    def add_volume_point(self, token_id: str, volume: float):
        """Add a volume point to the current window"""
        now = datetime.now()

        if token_id not in self.volume_windows:
            self.volume_windows[token_id] = []

        # Find or create current window
        current_window = None
        for window in self.volume_windows[token_id]:
            if window['start'] <= now < window['start'] + timedelta(minutes=self.window_minutes):
                current_window = window
                break

        if current_window:
            current_window['volume'] += volume
        else:
            # Create new window
            window_start = now.replace(second=0, microsecond=0)  # Round to minute
            # Align to window boundary
            minutes_since_epoch = int(window_start.timestamp() // 60)
            aligned_minutes = (minutes_since_epoch // self.window_minutes) * self.window_minutes
            window_start = datetime.fromtimestamp(aligned_minutes * 60)

            current_window = {'start': window_start, 'volume': volume}
            self.volume_windows[token_id].append(current_window)

        # Clean up old windows (keep only last 24 hours worth)
        cutoff = now - timedelta(hours=self.baseline_hours)
        self.volume_windows[token_id] = [
            w for w in self.volume_windows[token_id]
            if w['start'] > cutoff
        ]

    def get_relative_volume(self, token_id: str) -> Optional[Tuple[float, float, float]]:
        """
        Calculate relative volume for a token

        Returns:
            Tuple of (current_volume, baseline_average, relative_volume)
            Returns None if insufficient data
        """
        if token_id not in self.volume_windows:
            return None

        windows = self.volume_windows[token_id]
        if len(windows) < 2:  # Need at least current + some history
            return None

        # Get current window (most recent)
        current_window = max(windows, key=lambda w: w['start'])
        current_volume = current_window['volume']

        # Calculate baseline (average of all windows except current)
        historical_windows = [w for w in windows if w != current_window]
        if not historical_windows:
            return None

        baseline_average = sum(w['volume'] for w in historical_windows) / len(historical_windows)

        # Avoid division by zero
        if baseline_average <= 0:
            return None

        relative_volume = current_volume / baseline_average

        return current_volume, baseline_average, relative_volume

    def check_volume_spike(self, token_id: str, min_baseline_volume: float = 100) -> Optional[Dict]:
        """
        Check if token has a volume spike

        Args:
            min_baseline_volume: Minimum baseline volume to consider (avoids noise)

        Returns:
            Dict with spike info if detected, None otherwise
        """
        result = self.get_relative_volume(token_id)
        if not result:
            return None

        current_volume, baseline_average, relative_volume = result

        # Check conditions
        if (baseline_average >= min_baseline_volume and
            relative_volume >= self.spike_threshold):

            return {
                'current_volume': current_volume,
                'baseline_average': baseline_average,
                'relative_volume': relative_volume,
                'threshold': self.spike_threshold
            }

        return None

class SignalGenerator:
    """Generates trading signals based on price acceleration"""

    def __init__(self, config: Dict = None):
        """
        Initialize signal generator

        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            'min_history_points': 10,
            'acceleration_threshold': 0.001,  # Minimum acceleration for signal
            'velocity_threshold': 0.0005,     # Minimum velocity for trend
            'liquidity_min': 1000,            # Minimum liquidity for HIGH confidence
            'liquidity_medium': 5000,         # Minimum liquidity for MEDIUM confidence
            'price_extreme_threshold': 0.85,  # Price level to avoid (overbought)
            'smoothing_window': 3,            # EMA smoothing window
            'alert_cooldown_minutes': 5,      # Minimum time between alerts per token
        }

        self.price_collector = PriceDataCollector(max_history=100)
        self.volume_tracker = VolumeTracker(
            window_minutes=10,
            baseline_hours=24,
            spike_threshold=100
        )
        self.last_alert_times = {}  # token_id -> last_alert_timestamp
        self.logger = logging.getLogger(__name__)

    def calculate_velocity(self, prices: List[float], time_deltas: List[float]) -> List[float]:
        """Calculate first derivative (velocity)"""
        if len(prices) < 2:
            return []

        velocities = []
        for i in range(1, len(prices)):
            dt = time_deltas[i-1] if i-1 < len(time_deltas) else 1.0
            v = (prices[i] - prices[i-1]) / dt if dt > 0 else 0
            velocities.append(v)

        return velocities

    def calculate_acceleration(self, prices: List[float], time_deltas: List[float]) -> List[float]:
        """Calculate second derivative (acceleration) in price units per second squared"""
        if len(prices) < 3:
            return []

        accelerations = []
        for i in range(2, len(prices)):
            dt1 = time_deltas[i-2] if i-2 < len(time_deltas) else 1.0
            dt2 = time_deltas[i-1] if i-1 < len(time_deltas) else 1.0
            dt = (dt1 + dt2) / 2  # Average time delta in seconds

            if dt > 0:
                # Central difference approximation for acceleration
                # a = d²p/dt² = (p[i] - 2*p[i-1] + p[i-2]) / dt²
                a = (prices[i] - 2 * prices[i-1] + prices[i-2]) / (dt ** 2)
            else:
                a = 0

            accelerations.append(a)

        return accelerations

    def smooth_data(self, data: List[float], alpha: float = 0.3) -> List[float]:
        """Apply exponential moving average smoothing"""
        if not data:
            return []

        smoothed = [data[0]]
        for value in data[1:]:
            smoothed_value = alpha * value + (1 - alpha) * smoothed[-1]
            smoothed.append(smoothed_value)

        return smoothed

    def calculate_confidence(self, liquidity: float) -> str:
        """Calculate signal confidence based on liquidity"""
        if liquidity >= self.config['liquidity_medium']:
            return "HIGH"
        elif liquidity >= self.config['liquidity_min']:
            return "MEDIUM"
        else:
            return "LOW"

    def detect_acceleration_spike(self, acceleration: float, recent_accelerations: List[float]) -> bool:
        """Detect if current acceleration is a significant spike"""
        if len(recent_accelerations) < 5:
            return False

        # Calculate mean and std of recent accelerations
        recent = recent_accelerations[-10:] if len(recent_accelerations) >= 10 else recent_accelerations
        mean_a = np.mean(recent)
        std_a = np.std(recent)

        # Spike if acceleration is 3+ standard deviations from mean
        threshold = mean_a + 3 * std_a
        return abs(acceleration) > abs(threshold) and abs(acceleration) > abs(mean_a) * 2

    def should_skip_alert(self, token_id: str) -> bool:
        """Check if we should skip alert due to cooldown"""
        if token_id not in self.last_alert_times:
            return False

        cooldown_minutes = self.config['alert_cooldown_minutes']
        time_since_last = datetime.now() - self.last_alert_times[token_id]

        return time_since_last < timedelta(minutes=cooldown_minutes)

    def generate_signals(self, token_id: str, market_id: str, outcome: str, question: str) -> List[Signal]:
        """
        Generate signals for a token based on price history

        Returns:
            List of Signal objects
        """
        signals = []

        # Get price series
        series = self.price_collector.get_price_series(token_id, self.config['min_history_points'])
        if not series:
            return signals

        # Extract data
        timestamps = [s[0] for s in series]
        prices = [s[1] for s in series]
        liquidities = [s[2] for s in series]

        # Calculate time deltas in seconds
        time_deltas = []
        for i in range(1, len(timestamps)):
            try:
                # Ensure timestamps are datetime objects
                ts1 = timestamps[i-1]
                ts2 = timestamps[i]

                # Convert to datetime if they're strings (defensive programming)
                if isinstance(ts1, str):
                    ts1 = datetime.fromisoformat(ts1)
                if isinstance(ts2, str):
                    ts2 = datetime.fromisoformat(ts2)

                dt = (ts2 - ts1).total_seconds()
                time_deltas.append(dt)
            except (TypeError, ValueError, AttributeError) as e:
                self.logger.warning(f"Error calculating time delta: {e}, ts1={ts1}, ts2={ts2}")
                time_deltas.append(1.0)  # Default 1 second

        # Calculate raw velocity and acceleration
        velocities = self.calculate_velocity(prices, time_deltas)
        accelerations = self.calculate_acceleration(prices, time_deltas)

        if not accelerations:
            return signals

        # Smooth the data
        smoothed_prices = self.smooth_data(prices)
        smoothed_velocities = self.smooth_data(velocities) if velocities else []
        smoothed_accelerations = self.smooth_data(accelerations)

        # Get current values
        current_price = prices[-1]
        current_liquidity = liquidities[-1]
        current_velocity = smoothed_velocities[-1] if smoothed_velocities else velocities[-1] if velocities else 0
        current_acceleration = smoothed_accelerations[-1]

        # Skip if cooldown active
        if self.should_skip_alert(token_id):
            return signals

        # Calculate confidence
        confidence = self.calculate_confidence(current_liquidity)

        # Signal 1: Strong Buy Momentum
        if (current_acceleration > self.config['acceleration_threshold'] and
            current_velocity > self.config['velocity_threshold'] and
            current_price < self.config['price_extreme_threshold']):

            signal = Signal(
                type=SignalType.STRONG_BUY,
                strength=SignalStrength.HIGH if confidence == "HIGH" else SignalStrength.MEDIUM,
                market_id=market_id,
                token_id=token_id,
                outcome=outcome,
                question=question,
                price=current_price,
                velocity=current_velocity,
                acceleration=current_acceleration,
                liquidity=current_liquidity,
                timestamp=datetime.now(),
                reason=f"Positive acceleration ({current_acceleration:.6f}) + uptrend velocity ({current_velocity:.6f})",
                confidence=confidence
            )
            signals.append(signal)

        # Signal 2: Strong Sell Momentum
        elif (current_acceleration < -self.config['acceleration_threshold'] and
              current_velocity < -self.config['velocity_threshold'] and
              current_price > (1 - self.config['price_extreme_threshold'])):

            signal = Signal(
                type=SignalType.STRONG_SELL,
                strength=SignalStrength.HIGH if confidence == "HIGH" else SignalStrength.MEDIUM,
                market_id=market_id,
                token_id=token_id,
                outcome=outcome,
                question=question,
                price=current_price,
                velocity=current_velocity,
                acceleration=current_acceleration,
                liquidity=current_liquidity,
                timestamp=datetime.now(),
                reason=f"Negative acceleration ({current_acceleration:.6f}) + downtrend velocity ({current_velocity:.6f})",
                confidence=confidence
            )
            signals.append(signal)

        # Signal 3: Reversal Warning (Deceleration in trend)
        elif (len(smoothed_accelerations) >= 3 and
              current_acceleration < 0 and
              smoothed_accelerations[-2] > 0 and  # Previously accelerating
              current_velocity > 0 and  # Still moving up
              current_price > 0.75):  # Approaching overbought

            signal = Signal(
                type=SignalType.REVERSAL_WARNING,
                strength=SignalStrength.MEDIUM,
                market_id=market_id,
                token_id=token_id,
                outcome=outcome,
                question=question,
                price=current_price,
                velocity=current_velocity,
                acceleration=current_acceleration,
                liquidity=current_liquidity,
                timestamp=datetime.now(),
                reason=f"Deceleration in uptrend - potential reversal (prev_a: {smoothed_accelerations[-2]:.6f}, curr_a: {current_acceleration:.6f})",
                confidence=confidence
            )
            signals.append(signal)

        # Signal 4: Acceleration Spike (Event-driven)
        elif self.detect_acceleration_spike(current_acceleration, smoothed_accelerations):

            signal = Signal(
                type=SignalType.ACCELERATION_SPIKE,
                strength=SignalStrength.CRITICAL,
                market_id=market_id,
                token_id=token_id,
                outcome=outcome,
                question=question,
                price=current_price,
                velocity=current_velocity,
                acceleration=current_acceleration,
                liquidity=current_liquidity,
                timestamp=datetime.now(),
                reason=f"Extreme acceleration spike detected ({current_acceleration:.6f}) - possible breaking news",
                confidence=confidence
            )
            signals.append(signal)

        # Signal 5: Volume Spike (Abnormal activity)
        volume_spike = self.volume_tracker.check_volume_spike(token_id, min_baseline_volume=100)
        if volume_spike:
            signal = Signal(
                type=SignalType.VOLUME_SPIKE,
                strength=SignalStrength.CRITICAL,
                market_id=market_id,
                token_id=token_id,
                outcome=outcome,
                question=question,
                price=current_price,
                velocity=current_velocity,
                acceleration=current_acceleration,
                liquidity=current_liquidity,
                timestamp=datetime.now(),
                reason=f"Volume spike: {volume_spike['relative_volume']:.0f}x normal (${volume_spike['current_volume']:,.0f} vs ${volume_spike['baseline_average']:,.0f} baseline)",
                confidence="HIGH"  # Volume spikes are highly significant
            )
            signals.append(signal)

        # Update last alert time if signals generated
        if signals:
            self.last_alert_times[token_id] = datetime.now()

        return signals

    def update_price_data(self, token_id: str, price: float, liquidity: float):
        """Update price data for signal calculation"""
        self.price_collector.add_price_point(token_id, price, liquidity)

    def update_volume_data(self, token_id: str, volume: float):
        """Update volume data for relative volume signal calculation"""
        self.volume_tracker.add_volume_point(token_id, volume)

    def get_signal_history(self, token_id: str) -> Optional[Dict]:
        """Get signal calculation data for debugging"""
        series = self.price_collector.get_price_series(token_id, 1)
        if not series:
            return None

        timestamps = [s[0] for s in series]
        prices = [s[1] for s in series]
        liquidities = [s[2] for s in series]

        time_deltas = []
        for i in range(1, len(timestamps)):
            dt = (timestamps[i] - timestamps[i-1]).total_seconds()
            time_deltas.append(dt)

        velocities = self.calculate_velocity(prices, time_deltas)
        accelerations = self.calculate_acceleration(prices, time_deltas)

        return {
            'timestamps': [t.isoformat() for t in timestamps],
            'prices': prices,
            'liquidities': liquidities,
            'velocities': velocities,
            'accelerations': accelerations,
            'smoothed_prices': self.smooth_data(prices),
            'smoothed_velocities': self.smooth_data(velocities) if velocities else [],
            'smoothed_accelerations': self.smooth_data(accelerations) if accelerations else []
        }
