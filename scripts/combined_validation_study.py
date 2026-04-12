"""
Combined Validation Study: WiFi vs RFID vs Student Survey
Correlates three data sources to determine device-to-student ratios
and validate detection accuracy.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
from datetime import datetime, timedelta

# Mock data generation for demonstration if files don't exist
def generate_mock_data():
    """Generate mock RFID and Survey data for validation demonstration"""
    print("Generating mock RFID and Survey data for validation...")
    
    # Dates aligned with WiFi logs (July 2025)
    base_date = datetime(2025, 7, 28)
    dates = [(base_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    
    # 1. Mock RFID data
    rfid_records = []
    locations = ['miguel_pro', 'american_corner', 'gisbert_3']
    
    for date in dates:
        for loc in locations:
            # Generate 20-50 students per hour during open hours (8am-8pm)
            for hour in range(8, 20):
                num_students = np.random.randint(20, 100)
                for s in range(num_students):
                    stu_id = f"STU_{np.random.randint(1000, 9999)}"
                    # Entry
                    rfid_records.append({
                        'Timestamp': f"{date} {hour:02d}:{np.random.randint(0, 30):02d}:00",
                        'StudentID': stu_id,
                        'Location': loc,
                        'Action': 'Entry'
                    })
                    # Exit
                    rfid_records.append({
                        'Timestamp': f"{date} {hour:02d}:{np.random.randint(31, 59):02d}:00",
                        'StudentID': stu_id,
                        'Location': loc,
                        'Action': 'Exit'
                    })
    
    pd.DataFrame(rfid_records).to_csv('rfid_logs_mock.csv', index=False)
    
    # 2. Mock Survey data
    survey_records = []
    for i in range(200):
        # Most students have 1-2 devices
        num_devices = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        survey_records.append({
            'ResponseID': f"RESP_{i:03d}",
            'Date': np.random.choice(dates),
            'VisitedLibrary': 'Yes',
            'LibraryLocation': np.random.choice(['Miguel Pro Library', 'Gisbert 3rd Floor']),
            'ConnectedToWiFi': 'Yes',
            'NumberOfDevices': num_devices,
            'DeviceTypes': 'Laptop' if num_devices == 1 else 'Laptop, Phone'
        })
    
    pd.DataFrame(survey_records).to_csv('survey_responses_mock.csv', index=False)
    print("✓ Created rfid_logs_mock.csv and survey_responses_mock.csv")

class CombinedValidator:
    def __init__(self, wifi_file='all_data_cleaned.csv', rfid_file='rfid_logs_mock.csv', survey_file='survey_responses_mock.csv'):
        self.wifi_file = wifi_file
        self.rfid_file = rfid_file
        self.survey_file = survey_file
        self.results_dir = 'combined_validation_results'
        os.makedirs(self.results_dir, exist_ok=True)

    def load_data(self):
        print("Loading data sources...")
        
        # Load WiFi (if exists)
        if os.path.exists(self.wifi_file):
            from ap_location_mapping import get_location_from_ap
            self.wifi_df = pd.read_csv(self.wifi_file)
            self.wifi_df['Start_dt'] = pd.to_datetime(self.wifi_df['Start_dt'])
            self.wifi_df['Location'] = self.wifi_df['AP MAC'].apply(get_location_from_ap)
        else:
            print("⚠ WiFi data not found. Using empty DF.")
            self.wifi_df = pd.DataFrame(columns=['Start_dt', 'Client MAC', 'Location'])

        # Load RFID
        if not os.path.exists(self.rfid_file):
            generate_mock_data()
        self.rfid_df = pd.read_csv(self.rfid_file)
        self.rfid_df['Timestamp'] = pd.to_datetime(self.rfid_df['Timestamp'])

        # Load Survey
        if not os.path.exists(self.survey_file):
            generate_mock_data()
        self.survey_df = pd.read_csv(self.survey_file)

    def process_occupancy(self):
        print("Processing hourly occupancy...")
        
        # 1. WiFi Hourly
        self.wifi_hourly = self.wifi_df.groupby([
            pd.Grouper(key='Start_dt', freq='h'), 'Location'
        ])['Client MAC'].nunique().reset_index()
        self.wifi_hourly.columns = ['DateTime', 'Location', 'WiFi_MAC_Count']

        # 2. RFID Hourly (Snapshot approach)
        # Simplified for mock: Max entries per hour
        self.rfid_hourly = self.rfid_df[self.rfid_df['Action'] == 'Entry'].groupby([
            pd.Grouper(key='Timestamp', freq='h'), 'Location'
        ])['StudentID'].nunique().reset_index()
        self.rfid_hourly.columns = ['DateTime', 'Location', 'RFID_Student_Count']

        # Merge WiFi and RFID
        self.merged = pd.merge(self.wifi_hourly, self.rfid_hourly, on=['DateTime', 'Location'], how='inner')

    def analyze_survey(self):
        print("Analyzing survey device patterns...")
        avg_devices = self.survey_df['NumberOfDevices'].mean()
        device_dist = self.survey_df['NumberOfDevices'].value_counts(normalize=True).sort_index()
        
        print(f"Average Devices per Student (Survey): {avg_devices:.2f}")
        return avg_devices, device_dist

    def perform_validation(self):
        self.load_data()
        self.process_occupancy()
        avg_survey_devices, device_dist = self.analyze_survey()

        if len(self.merged) == 0:
            print("⚠ No overlapping WiFi and RFID data to validate.")
            return

        # Calculate implied device count from WiFi/RFID ratio
        self.merged['Implied_Devices_per_Student'] = self.merged['WiFi_MAC_Count'] / self.merged['RFID_Student_Count'].replace(0, 1)
        
        avg_implied_devices = self.merged['Implied_Devices_per_Student'].mean()
        
        print(f"Average Implied Devices (WiFi/RFID): {avg_implied_devices:.2f}")
        print(f"Comparison: Survey ({avg_survey_devices:.2f}) vs WiFi/RFID ({avg_implied_devices:.2f})")
        
        # Validation Metrics
        correlation = self.merged['WiFi_MAC_Count'].corr(self.merged['RFID_Student_Count'])
        
        # Generate Correction Factor
        # Use average of Survey and WiFi/RFID for robustness
        combined_factor = 1 / ((avg_survey_devices + avg_implied_devices) / 2)
        
        report = f"""
============================================================
COMBINED VALIDATION REPORT
============================================================

1. DATA SOURCE SUMMARY
- WiFi Records processed: {len(self.wifi_df)}
- RFID Records processed: {len(self.rfid_df)}
- Survey Responses: {len(self.survey_df)}

2. MULTI-DEVICE ANALYSIS
- Survey-reported devices/student: {avg_survey_devices:.2f}
- Implied devices/student (WiFi/RFID): {avg_implied_devices:.2f}
- Observed Bias: {'Over-counting' if avg_implied_devices > 1.0 else 'Under-counting'}

3. CORRELATION RESULTS
- WiFi MAC Count vs RFID Student Count: {correlation:.4f}

4. RECOMMENDATION
- Calculated Correction Factor: {combined_factor:.4f}
- Formula: Adjusted Occupancy = WiFi_MAC_Count * {combined_factor:.4f}

============================================================
"""
        with open(os.path.join(self.results_dir, 'validation_report.txt'), 'w') as f:
            f.write(report)
        print(report)
        
        # Visualizations
        plt.figure(figsize=(10, 6))
        sns.regplot(data=self.merged, x='RFID_Student_Count', y='WiFi_MAC_Count')
        plt.title('WiFi MAC Count vs RFID Ground Truth')
        plt.savefig(os.path.join(self.results_dir, 'wifi_rfid_correlation.png'))
        
        plt.figure(figsize=(8, 5))
        device_dist.plot(kind='bar', color='skyblue')
        plt.title('Device Count Distribution (Survey)')
        plt.xlabel('Number of Devices')
        plt.ylabel('Percentage of Students')
        plt.savefig(os.path.join(self.results_dir, 'survey_device_distribution.png'))

if __name__ == "__main__":
    validator = CombinedValidator()
    validator.perform_validation()
