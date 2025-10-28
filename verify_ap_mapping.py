"""
Verify AP MAC addresses exist in the data
"""
import pandas as pd
import sys
import io

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("VERIFYING AP MAC ADDRESSES")
print("=" * 70)

# Load data
df = pd.read_csv('all_data_cleaned.csv')
print(f"\nTotal records: {len(df):,}")
print(f"Unique AP MACs in data: {df['AP MAC'].nunique()}")

# APs to check
ap_mapping = {
    'Miguel Pro': [
        '10:F0:68:29:66:70',  # AP-202209-000005 (confirmed)
        '10:F0:68:28:3C:D0',  # AP-202209-000006 (confirmed)
        '10:F0:68:38:BD:40',  # AP-202209-000007 (confirmed)
        '80:BC:37:20:8A:20',  # Additional
        '10:F0:68:28:3A:80',  # Additional
    ],
    'American Corner': [
        '34:15:93:01:25:40',  # AP-202209-000062 (confirmed)
        'C0:C7:0A:32:62:10',  # Additional
        'C0:C7:0A:29:C0:70',  # Additional
    ],
    'Gisbert 2nd Floor': [
        '10:F0:68:29:68:20',  # AP-202209-000009 (confirmed)
        '5C:DF:89:07:69:30',  # Additional
        '5C:DF:89:07:62:A0',  # Additional
    ],
    'Gisbert 3rd Floor': [
        '5C:DF:89:07:3A:80',  # AP 89 (confirmed)
        '5C:DF:89:07:58:D0',  # Additional
        '00:33:58:29:52:F0',  # Additional
    ],
    'Gisbert 4th Floor': [
        '10:F0:68:29:21:60',  # AP-202209-000010 (confirmed)
        '80:BC:37:18:95:60',  # Additional
        '00:33:58:11:D3:D0',  # Additional
    ],
    'Gisbert 5th Floor': [
        '10:F0:68:28:76:50',  # AP-202209-000011 (confirmed)
    ],
}

print("\n" + "=" * 70)
print("VERIFICATION RESULTS")
print("=" * 70)

total_mapped = 0
for library, aps in ap_mapping.items():
    print(f"\n{library}:")
    lib_count = 0
    for ap in aps:
        count = (df['AP MAC'] == ap).sum()
        lib_count += count
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {ap}: {count:,} records")
    total_mapped += lib_count
    print(f"  → Total for {library}: {lib_count:,} records")

print("\n" + "=" * 70)
print(f"Total records mapped: {total_mapped:,} ({(total_mapped/len(df)*100):.1f}%)")
print(f"Unmapped records: {len(df) - total_mapped:,} ({((len(df)-total_mapped)/len(df)*100):.1f}%)")
print("=" * 70)

# Show top unmapped APs
print("\nTop 10 UNMAPPED AP MACs (for reference):")
mapped_aps = [ap for aps in ap_mapping.values() for ap in aps]
unmapped = df[~df['AP MAC'].isin(mapped_aps)]
if len(unmapped) > 0:
    print(unmapped['AP MAC'].value_counts().head(10))
else:
    print("  All APs are mapped!")
