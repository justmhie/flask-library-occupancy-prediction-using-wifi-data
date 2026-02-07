"""
Student Survey Validation System
Collects and analyzes survey data to validate WiFi-based occupancy detection
and understand multi-device usage patterns
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class StudentSurveyValidator:
    """
    System for collecting and validating WiFi occupancy data against
    student self-reported behavior
    """

    def __init__(self):
        self.survey_file = 'student_survey_responses.csv'
        self.validation_results_dir = 'survey_validation_results'
        os.makedirs(self.validation_results_dir, exist_ok=True)

    def create_survey_template(self):
        """
        Generate a CSV template for collecting student survey responses
        """
        template_data = {
            'ResponseID': ['Example1', 'Example2', 'Example3'],
            'Date': ['2024-03-15', '2024-03-15', '2024-03-16'],
            'StudentID': ['STU001', 'STU002', 'STU003'],
            'VisitedLibrary': ['Yes', 'Yes', 'No'],
            'LibraryLocation': ['Miguel Pro Library', 'Gisbert 3rd Floor', 'N/A'],
            'ArrivalTime': ['09:00', '10:30', 'N/A'],
            'DepartureTime': ['12:00', '14:00', 'N/A'],
            'ConnectedToWiFi': ['Yes', 'Yes', 'N/A'],
            'NumberOfDevices': [2, 1, 0],
            'DeviceTypes': ['Laptop, Phone', 'Laptop', 'N/A'],
            'PrimaryDeviceMAC': ['XX:XX:XX:XX:XX:01', 'XX:XX:XX:XX:XX:02', 'N/A'],
            'ReasonForNotConnecting': ['N/A', 'N/A', 'Did not visit'],
            'UsagePurpose': ['Study', 'Research', 'N/A'],
            'Notes': ['', 'WiFi was slow', '']
        }

        template_df = pd.DataFrame(template_data)

        # Save template
        template_path = os.path.join(self.validation_results_dir, 'survey_template.csv')
        template_df.to_csv(template_path, index=False)

        print(f"✓ Survey template created: {template_path}")
        print("\nSurvey Fields:")
        print("  - ResponseID: Unique identifier for each response")
        print("  - Date: Date of library visit (YYYY-MM-DD)")
        print("  - StudentID: Anonymous student identifier")
        print("  - VisitedLibrary: Yes/No")
        print("  - LibraryLocation: Which library location")
        print("  - ArrivalTime: Time entered library (HH:MM)")
        print("  - DepartureTime: Time left library (HH:MM)")
        print("  - ConnectedToWiFi: Yes/No")
        print("  - NumberOfDevices: How many devices connected")
        print("  - DeviceTypes: Types of devices (comma-separated)")
        print("  - PrimaryDeviceMAC: MAC address of primary device (optional)")
        print("  - ReasonForNotConnecting: If didn't connect, why?")
        print("  - UsagePurpose: Study/Research/Social/Other")
        print("  - Notes: Additional comments")

        return template_path

    def create_google_form_guide(self):
        """
        Generate a guide for creating a Google Form for survey distribution
        """
        guide = """
================================================================================
GOOGLE FORM SETUP GUIDE - WiFi OCCUPANCY VALIDATION SURVEY
================================================================================

FORM TITLE:
Library WiFi Usage Survey - Help Us Improve Library Services

FORM DESCRIPTION:
We're conducting research to improve our library occupancy tracking system.
Your responses will help us validate our WiFi-based detection methods and
understand multi-device usage patterns. This survey takes 3-5 minutes.
All responses are anonymous and used only for research purposes.

================================================================================
QUESTIONS
================================================================================

1. Response ID (Auto-generated)
   Type: Short answer
   Required: Yes
   Help text: Auto-filled by Google Forms

2. What is today's date?
   Type: Date
   Required: Yes

3. Anonymous Student ID
   Type: Short answer
   Required: Yes
   Help text: Create a unique code (e.g., first 3 letters of mother's name +
              last 4 digits of student ID). This helps us track multiple responses
              without identifying you.

4. Did you visit any library today?
   Type: Multiple choice
   Required: Yes
   Options:
   - Yes
   - No
   [If No, skip to end]

5. Which library location did you visit?
   Type: Multiple choice
   Required: Yes
   Options:
   - Miguel Pro Library
   - American Corner
   - Gisbert 2nd Floor
   - Gisbert 3rd Floor
   - Gisbert 4th Floor
   - Gisbert 5th Floor
   - Other (please specify)

6. What time did you arrive at the library?
   Type: Time
   Required: Yes

7. What time did you leave the library? (If still there, estimate departure)
   Type: Time
   Required: Yes

8. Did you connect any devices to the university WiFi during your visit?
   Type: Multiple choice
   Required: Yes
   Options:
   - Yes
   - No
   [If No, go to Question 11]

9. How many devices did you connect to WiFi?
   Type: Multiple choice
   Required: Yes
   Options:
   - 1 device
   - 2 devices
   - 3 devices
   - 4 or more devices

10. What types of devices did you connect? (Check all that apply)
    Type: Checkboxes
    Required: Yes
    Options:
    - Laptop
    - Smartphone
    - Tablet
    - Smartwatch
    - E-reader
    - Other (please specify)

11. [If didn't connect] Why didn't you connect to WiFi?
    Type: Checkboxes
    Required: No
    Options:
    - Don't have a device
    - WiFi not needed for my tasks
    - WiFi connection issues
    - Security concerns
    - Prefer mobile data
    - Other (please specify)

12. What was the primary purpose of your library visit?
    Type: Multiple choice
    Required: Yes
    Options:
    - Study/Homework
    - Research
    - Group work
    - Using library resources
    - Social/Meeting friends
    - Quiet space
    - Other (please specify)

13. Additional comments or observations
    Type: Paragraph
    Required: No
    Help text: Any issues with WiFi, observations about library occupancy, etc.

================================================================================
RESPONSE SETTINGS
================================================================================

✓ Collect email addresses: NO (to maintain anonymity)
✓ Limit to 1 response: NO (students may visit multiple times)
✓ Allow response editing: YES (in case of mistakes)
✓ Show progress bar: YES
✓ Shuffle question order: NO
✓ Show link to submit another response: YES

================================================================================
DISTRIBUTION STRATEGY
================================================================================

1. QR Code Distribution:
   - Generate QR code linking to form
   - Place posters near library entrances/exits
   - Include on library website

2. Incentives (if approved):
   - Entry into raffle for prizes
   - Extra credit (with instructor approval)
   - Library fee discount voucher

3. Target Collection Period:
   - 2-4 weeks during regular semester
   - Aim for 200-500 responses
   - Include both exam and non-exam periods

4. Promotion:
   - Email blast to students
   - Social media posts
   - Announcements in classes
   - Table at library entrance

================================================================================
DATA EXPORT
================================================================================

After collection:
1. Go to "Responses" tab in Google Form
2. Click the three dots menu
3. Select "Download responses (.csv)"
4. Save as 'student_survey_responses.csv'
5. Use with this validation system

================================================================================
"""

        guide_path = os.path.join(self.validation_results_dir, 'GOOGLE_FORM_SETUP_GUIDE.txt')
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide)

        print(f"\n✓ Google Form setup guide created: {guide_path}")

        return guide_path

    def load_survey_data(self, filepath=None):
        """Load survey responses from CSV"""
        if filepath is None:
            filepath = self.survey_file

        if not os.path.exists(filepath):
            print(f"⚠ Survey data not found: {filepath}")
            print("  Use create_survey_template() to generate a template")
            return None

        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} survey responses")

        return df

    def validate_with_wifi_logs(self, survey_df, wifi_df):
        """
        Validate survey responses against WiFi logs

        Args:
            survey_df: Survey responses DataFrame
            wifi_df: WiFi logs DataFrame (with columns: Start_dt, Client MAC, Location)

        Returns:
            Validation results dictionary
        """
        print("\n" + "=" * 80)
        print("VALIDATING SURVEY RESPONSES AGAINST WiFi LOGS")
        print("=" * 80)

        validation_results = []

        for idx, response in survey_df.iterrows():
            if response['VisitedLibrary'].lower() != 'yes':
                continue

            result = {
                'ResponseID': response['ResponseID'],
                'Date': response['Date'],
                'StudentID': response['StudentID'],
                'ReportedConnection': response['ConnectedToWiFi'].lower() == 'yes',
                'ReportedDeviceCount': int(response['NumberOfDevices']) if pd.notna(response['NumberOfDevices']) else 0,
            }

            # Check if MAC address provided
            if pd.notna(response.get('PrimaryDeviceMAC')) and response['PrimaryDeviceMAC'] != 'N/A':
                mac_address = response['PrimaryDeviceMAC']

                # Convert date and time to datetime
                date = pd.to_datetime(response['Date'])
                arrival = pd.to_datetime(f"{response['Date']} {response['ArrivalTime']}")
                departure = pd.to_datetime(f"{response['Date']} {response['DepartureTime']}")

                # Filter WiFi logs for this MAC, location, and time range
                location_map = {
                    'Miguel Pro Library': 'miguel_pro',
                    'American Corner': 'american_corner',
                    'Gisbert 2nd Floor': 'gisbert_2',
                    'Gisbert 3rd Floor': 'gisbert_3',
                    'Gisbert 4th Floor': 'gisbert_4',
                    'Gisbert 5th Floor': 'gisbert_5'
                }

                location = location_map.get(response['LibraryLocation'])

                if location:
                    wifi_matches = wifi_df[
                        (wifi_df['Client MAC'] == mac_address) &
                        (wifi_df['Location'] == location) &
                        (wifi_df['Start_dt'] >= arrival) &
                        (wifi_df['Start_dt'] <= departure)
                    ]

                    result['WiFiDetected'] = len(wifi_matches) > 0
                    result['WiFiConnectionCount'] = len(wifi_matches)
                    result['ValidationMatch'] = result['ReportedConnection'] == result['WiFiDetected']
                else:
                    result['WiFiDetected'] = None
                    result['WiFiConnectionCount'] = 0
                    result['ValidationMatch'] = None
            else:
                result['WiFiDetected'] = None
                result['WiFiConnectionCount'] = None
                result['ValidationMatch'] = None
                result['Note'] = 'No MAC address provided'

            validation_results.append(result)

        results_df = pd.DataFrame(validation_results)

        # Calculate statistics
        validated_responses = results_df[results_df['ValidationMatch'].notna()]

        if len(validated_responses) > 0:
            match_rate = (validated_responses['ValidationMatch'].sum() / len(validated_responses)) * 100

            print(f"\nValidation Statistics:")
            print(f"  Total responses: {len(survey_df)}")
            print(f"  Responses with MAC: {len(validated_responses)}")
            print(f"  Match rate: {match_rate:.1f}%")
            print(f"  True positives: {((validated_responses['ReportedConnection'] == True) & (validated_responses['WiFiDetected'] == True)).sum()}")
            print(f"  True negatives: {((validated_responses['ReportedConnection'] == False) & (validated_responses['WiFiDetected'] == False)).sum()}")
            print(f"  False positives: {((validated_responses['ReportedConnection'] == False) & (validated_responses['WiFiDetected'] == True)).sum()}")
            print(f"  False negatives: {((validated_responses['ReportedConnection'] == True) & (validated_responses['WiFiDetected'] == False)).sum()}")

        return results_df

    def analyze_device_patterns(self, survey_df):
        """
        Analyze multi-device usage patterns from survey
        """
        print("\n" + "=" * 80)
        print("MULTI-DEVICE USAGE ANALYSIS")
        print("=" * 80)

        # Filter to library visitors who connected
        connected = survey_df[
            (survey_df['VisitedLibrary'].str.lower() == 'yes') &
            (survey_df['ConnectedToWiFi'].str.lower() == 'yes')
        ]

        if len(connected) == 0:
            print("⚠ No connected users in survey data")
            return None

        # Device count distribution
        device_counts = connected['NumberOfDevices'].value_counts().sort_index()

        print(f"\nDevice Count Distribution:")
        for count, freq in device_counts.items():
            percentage = (freq / len(connected)) * 100
            print(f"  {count} device(s): {freq} students ({percentage:.1f}%)")

        # Average devices per student
        avg_devices = connected['NumberOfDevices'].mean()
        print(f"\nAverage devices per student: {avg_devices:.2f}")

        # Device types distribution
        print(f"\nDevice Types Usage:")
        device_types = []
        for types in connected['DeviceTypes'].dropna():
            if types != 'N/A':
                device_types.extend([d.strip() for d in types.split(',')])

        device_type_counts = pd.Series(device_types).value_counts()
        for device, count in device_type_counts.items():
            percentage = (count / len(connected)) * 100
            print(f"  {device}: {count} ({percentage:.1f}%)")

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Device count distribution
        axes[0].bar(device_counts.index, device_counts.values, color='#3498db', alpha=0.8, edgecolor='black')
        axes[0].set_xlabel('Number of Devices', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Number of Students', fontsize=12, fontweight='bold')
        axes[0].set_title('Device Count Distribution', fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (count, freq) in enumerate(device_counts.items()):
            axes[0].text(count, freq, str(freq), ha='center', va='bottom', fontweight='bold')

        # Device type distribution
        colors = sns.color_palette("Set2", len(device_type_counts))
        axes[1].barh(device_type_counts.index, device_type_counts.values, color=colors, alpha=0.8, edgecolor='black')
        axes[1].set_xlabel('Number of Students', fontsize=12, fontweight='bold')
        axes[1].set_title('Device Types Usage', fontsize=13, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (device, count) in enumerate(device_type_counts.items()):
            axes[1].text(count, i, f' {count}', va='center', fontweight='bold')

        plt.tight_layout()
        plot_path = os.path.join(self.validation_results_dir, 'device_usage_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {plot_path}")
        plt.close()

        return {
            'total_connected_students': len(connected),
            'average_devices_per_student': float(avg_devices),
            'device_count_distribution': device_counts.to_dict(),
            'device_type_distribution': device_type_counts.to_dict()
        }

    def generate_correction_factor(self, survey_df):
        """
        Calculate correction factor for WiFi-based occupancy counts
        based on multi-device usage
        """
        connected = survey_df[
            (survey_df['VisitedLibrary'].str.lower() == 'yes') &
            (survey_df['ConnectedToWiFi'].str.lower() == 'yes')
        ]

        if len(connected) == 0:
            return None

        avg_devices = connected['NumberOfDevices'].mean()
        correction_factor = 1 / avg_devices

        print(f"\n" + "=" * 80)
        print("OCCUPANCY CORRECTION FACTOR")
        print("=" * 80)
        print(f"\nAverage devices per student: {avg_devices:.2f}")
        print(f"Correction factor: {correction_factor:.4f}")
        print(f"\nUsage:")
        print(f"  Actual Occupancy ≈ WiFi MAC Count × {correction_factor:.4f}")
        print(f"\nExample:")
        print(f"  WiFi detects 150 unique MACs")
        print(f"  Estimated actual students: 150 × {correction_factor:.4f} = {150 * correction_factor:.0f}")

        return {
            'average_devices_per_student': float(avg_devices),
            'correction_factor': float(correction_factor),
            'formula': f'Actual Occupancy = WiFi MAC Count × {correction_factor:.4f}'
        }


def main():
    """Demo and setup"""
    print("=" * 80)
    print("STUDENT SURVEY VALIDATION SYSTEM")
    print("=" * 80)

    validator = StudentSurveyValidator()

    # Create survey template
    print("\n1. Creating survey template...")
    template_path = validator.create_survey_template()

    # Create Google Form guide
    print("\n2. Creating Google Form setup guide...")
    guide_path = validator.create_google_form_guide()

    print("\n" + "=" * 80)
    print("SETUP COMPLETE")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. Review the survey template:")
    print(f"     {template_path}")
    print("\n  2. Create Google Form using the guide:")
    print(f"     {guide_path}")
    print("\n  3. Distribute survey to students (QR codes, email, etc.)")
    print("\n  4. After collecting responses, download as CSV")
    print("\n  5. Run validation analysis:")
    print("     from student_survey_validation import StudentSurveyValidator")
    print("     validator = StudentSurveyValidator()")
    print("     survey_df = validator.load_survey_data('responses.csv')")
    print("     results = validator.analyze_device_patterns(survey_df)")
    print("     correction = validator.generate_correction_factor(survey_df)")
    print("=" * 80)


if __name__ == "__main__":
    main()
