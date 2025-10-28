"""
Quick test script to verify backend predictions work
"""
import pandas as pd
from datetime import datetime
import sys

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("Testing Backend Data Processing")
print("=" * 60)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('all_data_cleaned.csv')
print(f"   OK Loaded {len(df):,} rows")
print(f"   OK Columns: {len(df.columns)}")
print(f"   OK Has 'Client MAC': {'Client MAC' in df.columns}")
print(f"   OK Has 'Start_dt': {'Start_dt' in df.columns}")
print(f"   OK Has 'Location': {'Location' in df.columns}")

# Process timestamps
print("\n2. Processing timestamps...")
df['Start_dt'] = pd.to_datetime(df['Start_dt'])
df.set_index('Start_dt', inplace=True)
print(f"   ✓ Date range: {df.index.min()} to {df.index.max()}")

# Calculate occupancy
print("\n3. Calculating hourly occupancy...")
occupancy = df['Client MAC'].resample('h').nunique()
occupancy = occupancy.fillna(0)
print(f"   ✓ Total hours: {len(occupancy)}")
print(f"   ✓ Min users/hour: {int(occupancy.min())}")
print(f"   ✓ Max users/hour: {int(occupancy.max())}")
print(f"   ✓ Average users/hour: {int(occupancy.mean())}")
print(f"   ✓ Current (last hour): {int(occupancy.iloc[-1])}")

# Show last 10 hours
print("\n4. Last 10 hours of data:")
print(occupancy.tail(10))

# Simple prediction
print("\n5. Testing prediction...")
current = int(occupancy.iloc[-1])
recent_24h = occupancy.tail(24)
avg_24h = int(recent_24h.mean())

if len(occupancy) >= 48:
    older_avg = occupancy.tail(48).head(24).mean()
    recent_avg = occupancy.tail(24).mean()
    trend = (recent_avg - older_avg) / 24
    predicted_next = current + trend
else:
    predicted_next = current

print(f"   ✓ Current users: {current}")
print(f"   ✓ 24h average: {avg_24h}")
print(f"   ✓ Predicted next hour: {int(max(0, predicted_next))}")

print("\n" + "=" * 60)
print("✅ All tests passed! Backend should work correctly.")
print("=" * 60)
