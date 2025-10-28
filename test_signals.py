"""
Test script for signal generation
"""

import numpy as np
from datetime import datetime, timedelta
from signal_generator import SignalGenerator, SignalType, SignalStrength

def test_signal_generation():
    """Test the signal generation logic with synthetic data"""

    # Create signal generator
    config = {
        'min_history_points': 10,
        'acceleration_threshold': 1000000,  # Higher threshold for large acceleration values
        'velocity_threshold': 1000,         # Higher velocity threshold
        'liquidity_min': 1000,
        'liquidity_medium': 5000,
        'price_extreme_threshold': 0.85,
        'alert_cooldown_minutes': 5,
    }

    generator = SignalGenerator(config)

    # Simulate price data for a token
    token_id = "test_token_123"
    market_id = "test_market"
    outcome = "YES"
    question = "Will this test pass?"

    # Generate synthetic price data with increasing acceleration
    base_price = 0.5
    prices = []
    timestamps = []

    # Start with stable prices
    for i in range(10):
        price = base_price + np.random.normal(0, 0.001)  # Small random variation
        timestamp = datetime.now() - timedelta(seconds=(20-i)*10)
        prices.append(price)
        timestamps.append(timestamp)

        # Add to generator
        generator.update_price_data(token_id, price, 6000)  # Good liquidity

    print("Added initial stable prices...")

    # Now add prices with increasing acceleration (simulating momentum)
    for i in range(10):
        # Create acceleration: price increases with increasing velocity
        acceleration_factor = i * 0.002  # Increasing acceleration
        price = base_price + (i * 0.01) + (acceleration_factor * i * 0.1) + np.random.normal(0, 0.001)
        timestamp = datetime.now() - timedelta(seconds=(10-i)*10)
        prices.append(price)
        timestamps.append(timestamp)

        # Add to generator
        generator.update_price_data(token_id, price, 6000)  # Good liquidity

    print("Added accelerating prices...")

    # Debug: Check the actual acceleration values
    history = generator.get_signal_history(token_id)
    if history:
        accelerations = history.get('accelerations', [])
        velocities = history.get('velocities', [])
        prices = history.get('prices', [])

        print(f"\nDebug Info:")
        print(f"  Final price: ${prices[-1]:.4f}")
        print(f"  Final velocity: {velocities[-1]:.6f}" if velocities else "  No velocities")
        print(f"  Final acceleration: {accelerations[-1]:.6f}" if accelerations else "  No accelerations")
        print(f"  Acceleration threshold: {config['acceleration_threshold']}")
        print(f"  Velocity threshold: {config['velocity_threshold']}")

    # Try to generate signals
    signals = generator.generate_signals(token_id, market_id, outcome, question)

    print(f"\nGenerated {len(signals)} signals:")
    for i, signal in enumerate(signals, 1):
        print(f"\nSignal {i}:")
        print(f"  Type: {signal.type.value}")
        print(f"  Strength: {signal.strength.value}")
        print(f"  Price: ${signal.price:.4f}")
        print(f"  Acceleration: {signal.acceleration:.6f}")
        print(f"  Velocity: {signal.velocity:.6f}")
        print(f"  Confidence: {signal.confidence}")
        print(f"  Reason: {signal.reason}")

    # Test signal history
    history = generator.get_signal_history(token_id)
    if history:
        print("\nSignal History Available:")
        print(f"  Data points: {len(history.get('prices', []))}")
        print(f"  Velocities: {len(history.get('velocities', []))}")
        print(f"  Accelerations: {len(history.get('accelerations', []))}")

    return signals

if __name__ == "__main__":
    print("Testing Signal Generation...")
    signals = test_signal_generation()

    if signals:
        print(f"\n✅ Test PASSED: Generated {len(signals)} signals")
    else:
        print("\n❌ Test FAILED: No signals generated")
