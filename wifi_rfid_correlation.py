"""
WiFi vs RFID Correlation Analysis
Validates WiFi-based occupancy detection against RFID entry/exit logs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os
from datetime import datetime

class WiFiRFIDCorrelation:
    """
    System for correlating WiFi detection with RFID access logs
    to validate occupancy counting accuracy
    """

    def __init__(self):
        self.results_dir = 'wifi_rfid_correlation_results'
        os.makedirs(self.results_dir, exist_ok=True)

    def load_wifi_data(self, filepath, location_column='Location', datetime_column='Start_dt'):
        """
        Load and process WiFi logs

        Expected columns: Start_dt, Client MAC, Location (or AP MAC)
        """
        print(f"Loading WiFi data from {filepath}...")
        df = pd.read_csv(filepath)

        # Ensure datetime column
        df[datetime_column] = pd.to_datetime(df[datetime_column])

        # Calculate hourly occupancy per location
        df_grouped = df.groupby([
            pd.Grouper(key=datetime_column, freq='h'),
            location_column
        ])['Client MAC'].nunique().reset_index()

        df_grouped.columns = ['DateTime', 'Location', 'WiFi_Occupancy']

        print(f"✓ Loaded {len(df)} WiFi records")
        print(f"✓ Aggregated to {len(df_grouped)} hourly records")

        return df_grouped

    def load_rfid_data(self, filepath, location_column='Location', datetime_column='Timestamp'):
        """
        Load and process RFID entry/exit logs

        Expected columns: Timestamp, StudentID, Location, Action (Entry/Exit)
        """
        print(f"\nLoading RFID data from {filepath}...")
        df = pd.read_csv(filepath)

        # Ensure datetime column
        df[datetime_column] = pd.to_datetime(df[datetime_column])

        # Calculate occupancy by tracking entries and exits
        df_sorted = df.sort_values([location_column, datetime_column])

        occupancy_records = []

        for location in df[location_column].unique():
            location_data = df_sorted[df_sorted[location_column] == location].copy()

            # Track current occupancy
            current_occupancy = {}
            hourly_occupancy = {}

            for idx, row in location_data.iterrows():
                student_id = row['StudentID']
                action = row.get('Action', 'Entry').lower()
                timestamp = row[datetime_column]
                hour = timestamp.floor('h')

                if action == 'entry':
                    current_occupancy[student_id] = timestamp
                elif action == 'exit':
                    if student_id in current_occupancy:
                        del current_occupancy[student_id]

                # Record hourly snapshot
                if hour not in hourly_occupancy:
                    hourly_occupancy[hour] = len(current_occupancy)
                else:
                    hourly_occupancy[hour] = max(hourly_occupancy[hour], len(current_occupancy))

            # Convert to records
            for hour, count in hourly_occupancy.items():
                occupancy_records.append({
                    'DateTime': hour,
                    'Location': location,
                    'RFID_Occupancy': count
                })

        rfid_df = pd.DataFrame(occupancy_records)

        print(f"✓ Loaded {len(df)} RFID records")
        print(f"✓ Calculated {len(rfid_df)} hourly occupancy snapshots")

        return rfid_df

    def create_rfid_template(self):
        """
        Create a template CSV for RFID data collection

        This helps institutions that have RFID systems to format their data
        """
        template_data = {
            'Timestamp': [
                '2024-03-15 09:00:00',
                '2024-03-15 09:05:00',
                '2024-03-15 09:10:00',
                '2024-03-15 09:15:00',
                '2024-03-15 11:00:00',
                '2024-03-15 11:05:00'
            ],
            'StudentID': [
                'RFID_001',
                'RFID_002',
                'RFID_003',
                'RFID_001',
                'RFID_002',
                'RFID_004'
            ],
            'Location': [
                'miguel_pro',
                'miguel_pro',
                'gisbert_3',
                'miguel_pro',
                'miguel_pro',
                'miguel_pro'
            ],
            'Action': [
                'Entry',
                'Entry',
                'Entry',
                'Exit',
                'Exit',
                'Entry'
            ],
            'Notes': [
                '',
                '',
                '',
                '',
                '',
                'New entry after previous exit'
            ]
        }

        template_df = pd.DataFrame(template_data)

        template_path = os.path.join(self.results_dir, 'rfid_data_template.csv')
        template_df.to_csv(template_path, index=False)

        print(f"\n✓ RFID data template created: {template_path}")
        print("\nRFID Data Format:")
        print("  - Timestamp: Date and time of entry/exit (YYYY-MM-DD HH:MM:SS)")
        print("  - StudentID: Anonymous student/RFID card identifier")
        print("  - Location: Library location code")
        print("  - Action: 'Entry' or 'Exit'")
        print("  - Notes: Optional notes about the record")

        return template_path

    def correlate_data(self, wifi_df, rfid_df):
        """
        Correlate WiFi and RFID data by location and time
        """
        print("\n" + "=" * 80)
        print("CORRELATING WiFi AND RFID DATA")
        print("=" * 80)

        # Merge datasets
        merged_df = pd.merge(
            wifi_df,
            rfid_df,
            on=['DateTime', 'Location'],
            how='inner'
        )

        if len(merged_df) == 0:
            print("⚠ No overlapping data found between WiFi and RFID logs")
            print("  Check that:")
            print("  - Date ranges overlap")
            print("  - Location names match")
            print("  - Time zones are consistent")
            return None

        print(f"\n✓ Found {len(merged_df)} overlapping hourly records")

        # Calculate correlation metrics
        correlation = merged_df['WiFi_Occupancy'].corr(merged_df['RFID_Occupancy'])
        r2 = r2_score(merged_df['RFID_Occupancy'], merged_df['WiFi_Occupancy'])
        mae = mean_absolute_error(merged_df['RFID_Occupancy'], merged_df['WiFi_Occupancy'])
        rmse = np.sqrt(mean_squared_error(merged_df['RFID_Occupancy'], merged_df['WiFi_Occupancy']))

        # Perform statistical tests
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            merged_df['RFID_Occupancy'],
            merged_df['WiFi_Occupancy']
        )

        print(f"\nCorrelation Metrics:")
        print(f"  Pearson Correlation: {correlation:.4f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.2f} users")
        print(f"  RMSE: {rmse:.2f} users")
        print(f"\nLinear Regression:")
        print(f"  WiFi = {slope:.4f} × RFID + {intercept:.2f}")
        print(f"  p-value: {p_value:.6f}")

        # Calculate over/under detection rates
        merged_df['Difference'] = merged_df['WiFi_Occupancy'] - merged_df['RFID_Occupancy']
        merged_df['Abs_Difference'] = merged_df['Difference'].abs()
        merged_df['Percent_Difference'] = (merged_df['Difference'] / merged_df['RFID_Occupancy'].replace(0, 1)) * 100

        over_detection = (merged_df['Difference'] > 0).sum()
        under_detection = (merged_df['Difference'] < 0).sum()
        exact_match = (merged_df['Difference'] == 0).sum()

        print(f"\nDetection Analysis:")
        print(f"  Over-detection (WiFi > RFID): {over_detection} ({over_detection/len(merged_df)*100:.1f}%)")
        print(f"  Under-detection (WiFi < RFID): {under_detection} ({under_detection/len(merged_df)*100:.1f}%)")
        print(f"  Exact matches: {exact_match} ({exact_match/len(merged_df)*100:.1f}%)")
        print(f"  Average difference: {merged_df['Difference'].mean():.2f} users")
        print(f"  Average absolute difference: {merged_df['Abs_Difference'].mean():.2f} users")

        results = {
            'total_records': len(merged_df),
            'correlation': float(correlation),
            'r2_score': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'slope': float(slope),
            'intercept': float(intercept),
            'p_value': float(p_value),
            'over_detection_rate': float(over_detection / len(merged_df)),
            'under_detection_rate': float(under_detection / len(merged_df)),
            'average_difference': float(merged_df['Difference'].mean()),
            'average_absolute_difference': float(merged_df['Abs_Difference'].mean())
        }

        return merged_df, results

    def visualize_correlation(self, merged_df, results):
        """
        Create comprehensive visualizations of WiFi vs RFID correlation
        """
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Scatter plot with regression line
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.scatter(merged_df['RFID_Occupancy'], merged_df['WiFi_Occupancy'],
                   alpha=0.5, s=50, color='#3498db', edgecolor='black', linewidth=0.5)

        # Add regression line
        x_range = np.linspace(merged_df['RFID_Occupancy'].min(), merged_df['RFID_Occupancy'].max(), 100)
        y_pred = results['slope'] * x_range + results['intercept']
        ax1.plot(x_range, y_pred, 'r--', linewidth=2, label=f'y = {results["slope"]:.2f}x + {results["intercept"]:.2f}')

        # Add perfect correlation line
        max_val = max(merged_df['RFID_Occupancy'].max(), merged_df['WiFi_Occupancy'].max())
        ax1.plot([0, max_val], [0, max_val], 'g-', linewidth=2, alpha=0.5, label='Perfect Correlation (y=x)')

        ax1.set_xlabel('RFID Occupancy (Ground Truth)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('WiFi Occupancy (Detected)', fontsize=12, fontweight='bold')
        ax1.set_title(f'WiFi vs RFID Occupancy Correlation\nR² = {results["r2_score"]:.4f}, r = {results["correlation"]:.4f}',
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. Correlation metrics summary
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')

        metrics_text = f"""
CORRELATION METRICS

Pearson r: {results['correlation']:.4f}
R² Score: {results['r2_score']:.4f}
p-value: {results['p_value']:.6f}

ERROR METRICS

MAE: {results['mae']:.2f} users
RMSE: {results['rmse']:.2f} users

DETECTION RATES

Over-detection: {results['over_detection_rate']*100:.1f}%
Under-detection: {results['under_detection_rate']*100:.1f}%

Avg Difference: {results['average_difference']:.2f}
Avg |Difference|: {results['average_absolute_difference']:.2f}
"""

        ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes,
                fontsize=10, fontfamily='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # 3. Difference distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(merged_df['Difference'], bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='green', linestyle='--', linewidth=2, label='Perfect Match')
        ax3.axvline(merged_df['Difference'].mean(), color='blue', linestyle='--', linewidth=2, label='Mean Difference')
        ax3.set_xlabel('Difference (WiFi - RFID)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title('Detection Difference Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # 4. Absolute difference distribution
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(merged_df['Abs_Difference'], bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax4.axvline(merged_df['Abs_Difference'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {merged_df["Abs_Difference"].mean():.2f}')
        ax4.set_xlabel('Absolute Difference |WiFi - RFID|', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax4.set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        # 5. Box plot comparison
        ax5 = fig.add_subplot(gs[1, 2])
        bp = ax5.boxplot([merged_df['RFID_Occupancy'], merged_df['WiFi_Occupancy']],
                         labels=['RFID\n(Ground Truth)', 'WiFi\n(Detected)'],
                         patch_artist=True,
                         widths=0.6)

        colors = ['#2ecc71', '#3498db']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax5.set_ylabel('Occupancy Count', fontsize=11, fontweight='bold')
        ax5.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)

        # 6. Time series comparison (sample)
        ax6 = fig.add_subplot(gs[2, :])

        # Plot sample period (first 168 hours = 1 week)
        sample_df = merged_df.head(min(168, len(merged_df))).copy()
        sample_df = sample_df.sort_values('DateTime')

        ax6.plot(sample_df['DateTime'], sample_df['RFID_Occupancy'],
                label='RFID (Ground Truth)', color='#2ecc71', linewidth=2, marker='o', markersize=3)
        ax6.plot(sample_df['DateTime'], sample_df['WiFi_Occupancy'],
                label='WiFi (Detected)', color='#3498db', linewidth=2, marker='s', markersize=3, alpha=0.7)

        ax6.set_xlabel('Date/Time', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Occupancy', fontsize=11, fontweight='bold')
        ax6.set_title('Time Series Comparison (Sample Period)', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(alpha=0.3)
        ax6.tick_params(axis='x', rotation=45)

        plt.suptitle('WiFi vs RFID Occupancy Correlation Analysis', fontsize=14, fontweight='bold', y=0.995)

        plot_path = os.path.join(self.results_dir, 'wifi_rfid_correlation_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved: {plot_path}")
        plt.close()

        return plot_path

    def generate_report(self, merged_df, results):
        """
        Generate comprehensive text report
        """
        report_lines = [
            "=" * 80,
            "WiFi vs RFID CORRELATION ANALYSIS REPORT",
            "=" * 80,
            "",
            "OBJECTIVE:",
            "Validate WiFi-based occupancy detection against RFID entry/exit logs",
            "to assess accuracy and identify systematic biases.",
            "",
            "=" * 80,
            "DATA SUMMARY",
            "=" * 80,
            "",
            f"Total overlapping records: {results['total_records']:,}",
            f"Date range: {merged_df['DateTime'].min()} to {merged_df['DateTime'].max()}",
            f"Locations: {merged_df['Location'].nunique()} unique locations",
            "",
            "=" * 80,
            "CORRELATION ANALYSIS",
            "=" * 80,
            "",
            f"Pearson Correlation Coefficient: {results['correlation']:.4f}",
            f"R² Score: {results['r2_score']:.4f}",
            f"p-value: {results['p_value']:.6f}",
            "",
            "Interpretation:",
            f"  - Correlation strength: {'Very Strong' if abs(results['correlation']) > 0.9 else 'Strong' if abs(results['correlation']) > 0.7 else 'Moderate' if abs(results['correlation']) > 0.5 else 'Weak'}",
            f"  - Statistical significance: {'Significant' if results['p_value'] < 0.05 else 'Not significant'} (α = 0.05)",
            "",
            "=" * 80,
            "ACCURACY METRICS",
            "=" * 80,
            "",
            f"Mean Absolute Error (MAE): {results['mae']:.2f} users",
            f"Root Mean Square Error (RMSE): {results['rmse']:.2f} users",
            f"Average Difference (WiFi - RFID): {results['average_difference']:.2f} users",
            "",
            "=" * 80,
            "DETECTION BIAS ANALYSIS",
            "=" * 80,
            "",
            f"Over-detection rate: {results['over_detection_rate']*100:.1f}%",
            f"  (WiFi counts MORE users than RFID)",
            "",
            f"Under-detection rate: {results['under_detection_rate']*100:.1f}%",
            f"  (WiFi counts FEWER users than RFID)",
            "",
            "Likely causes of over-detection:",
            "  - Students connecting multiple devices",
            "  - Device MAC randomization creating multiple IDs",
            "  - WiFi signal bleed from adjacent areas",
            "",
            "Likely causes of under-detection:",
            "  - Students not connecting to WiFi",
            "  - Devices in airplane mode",
            "  - Weak signal areas",
            "",
            "=" * 80,
            "LINEAR REGRESSION MODEL",
            "=" * 80,
            "",
            f"WiFi Occupancy = {results['slope']:.4f} × RFID Occupancy + {results['intercept']:.2f}",
            "",
            f"Slope: {results['slope']:.4f}",
            f"  - Value > 1: WiFi tends to over-count",
            f"  - Value < 1: WiFi tends to under-count",
            f"  - Value ≈ 1: WiFi accurately tracks RFID",
            "",
            f"Intercept: {results['intercept']:.2f}",
            f"  - Baseline difference when RFID = 0",
            "",
            "=" * 80,
            "RECOMMENDATIONS",
            "=" * 80,
            "",
        ]

        # Add specific recommendations based on results
        if results['correlation'] > 0.9:
            report_lines.append("✓ EXCELLENT correlation between WiFi and RFID")
            report_lines.append("  WiFi-based occupancy is highly reliable for this environment")
        elif results['correlation'] > 0.7:
            report_lines.append("✓ GOOD correlation between WiFi and RFID")
            report_lines.append("  WiFi-based occupancy is reasonably reliable")
        else:
            report_lines.append("⚠ MODERATE correlation between WiFi and RFID")
            report_lines.append("  Consider calibration or hybrid approach")

        report_lines.append("")

        if abs(results['average_difference']) > 5:
            report_lines.append("⚠ Significant systematic bias detected")
            if results['average_difference'] > 0:
                report_lines.append(f"  WiFi over-counts by ~{results['average_difference']:.1f} users on average")
                report_lines.append("  Apply correction factor or survey validation")
            else:
                report_lines.append(f"  WiFi under-counts by ~{abs(results['average_difference']):.1f} users on average")
                report_lines.append("  Investigate WiFi coverage issues")
        else:
            report_lines.append("✓ No significant systematic bias")
            report_lines.append("  WiFi counts are well-calibrated")

        report_lines.extend([
            "",
            "=" * 80,
            "GENERATED FILES",
            "=" * 80,
            "",
            f"Visualization: {self.results_dir}/wifi_rfid_correlation_analysis.png",
            f"Report: {self.results_dir}/correlation_report.txt",
            f"Data: {self.results_dir}/merged_correlation_data.csv",
            "",
            "=" * 80,
        ])

        report_text = "\n".join(report_lines)

        report_path = os.path.join(self.results_dir, 'correlation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n✓ Report saved: {report_path}")

        return report_text


def main():
    """Demo and setup"""
    print("=" * 80)
    print("WiFi vs RFID CORRELATION ANALYSIS SYSTEM")
    print("=" * 80)

    analyzer = WiFiRFIDCorrelation()

    # Create RFID template
    print("\n1. Creating RFID data template...")
    template_path = analyzer.create_rfid_template()

    print("\n" + "=" * 80)
    print("SETUP COMPLETE")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. Obtain RFID entry/exit logs from your institution")
    print("  2. Format RFID data using the template:")
    print(f"     {template_path}")
    print("\n  3. Run correlation analysis:")
    print("     from wifi_rfid_correlation import WiFiRFIDCorrelation")
    print("     analyzer = WiFiRFIDCorrelation()")
    print("     wifi_df = analyzer.load_wifi_data('all_data_cleaned.csv')")
    print("     rfid_df = analyzer.load_rfid_data('rfid_logs.csv')")
    print("     merged, results = analyzer.correlate_data(wifi_df, rfid_df)")
    print("     analyzer.visualize_correlation(merged, results)")
    print("=" * 80)


if __name__ == "__main__":
    main()
