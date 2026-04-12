"""
Exam Period Tagging System
Tags historical and future dates as exam periods to improve model predictions
during high-stress academic periods
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

class ExamPeriodTagger:
    """
    System for tagging exam periods in library occupancy data

    Exam periods show different occupancy patterns due to:
    - Increased study time
    - Extended library hours
    - Different daily patterns
    """

    def __init__(self, config_file='exam_periods_config.json'):
        self.config_file = config_file
        self.exam_periods = []
        self.load_config()

    def load_config(self):
        """Load exam period configuration from JSON file"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.exam_periods = config.get('exam_periods', [])
            print(f"✓ Loaded {len(self.exam_periods)} exam periods from {self.config_file}")
        else:
            print(f"⚠ No config file found. Creating default configuration.")
            self.create_default_config()

    def create_default_config(self):
        """Create default exam period configuration with common academic dates"""
        # Default exam periods (customize based on your institution's calendar)
        default_periods = [
            # 2023 Exam Periods
            {
                "name": "Midterms Fall 2023",
                "start_date": "2023-10-16",
                "end_date": "2023-10-27",
                "type": "midterm",
                "semester": "fall",
                "year": 2023
            },
            {
                "name": "Finals Fall 2023",
                "start_date": "2023-12-04",
                "end_date": "2023-12-15",
                "type": "final",
                "semester": "fall",
                "year": 2023
            },
            # 2024 Exam Periods
            {
                "name": "Midterms Spring 2024",
                "start_date": "2024-03-11",
                "end_date": "2024-03-22",
                "type": "midterm",
                "semester": "spring",
                "year": 2024
            },
            {
                "name": "Finals Spring 2024",
                "start_date": "2024-05-06",
                "end_date": "2024-05-17",
                "type": "final",
                "semester": "spring",
                "year": 2024
            },
            {
                "name": "Midterms Fall 2024",
                "start_date": "2024-10-14",
                "end_date": "2024-10-25",
                "type": "midterm",
                "semester": "fall",
                "year": 2024
            },
            {
                "name": "Finals Fall 2024",
                "start_date": "2024-12-02",
                "end_date": "2024-12-13",
                "type": "final",
                "semester": "fall",
                "year": 2024
            },
            # 2025 Exam Periods (Future)
            {
                "name": "Midterms Spring 2025",
                "start_date": "2025-03-10",
                "end_date": "2025-03-21",
                "type": "midterm",
                "semester": "spring",
                "year": 2025
            },
            {
                "name": "Finals Spring 2025",
                "start_date": "2025-05-05",
                "end_date": "2025-05-16",
                "type": "final",
                "semester": "spring",
                "year": 2025
            },
            {
                "name": "Midterms Fall 2025",
                "start_date": "2025-10-13",
                "end_date": "2025-10-24",
                "type": "midterm",
                "semester": "fall",
                "year": 2025
            },
            {
                "name": "Finals Fall 2025",
                "start_date": "2025-12-01",
                "end_date": "2025-12-12",
                "type": "final",
                "semester": "fall",
                "year": 2025
            },
            # 2026 Exam Periods (Future)
            {
                "name": "Midterms Spring 2026",
                "start_date": "2026-03-09",
                "end_date": "2026-03-20",
                "type": "midterm",
                "semester": "spring",
                "year": 2026
            },
            {
                "name": "Finals Spring 2026",
                "start_date": "2026-05-04",
                "end_date": "2026-05-15",
                "type": "final",
                "semester": "spring",
                "year": 2026
            }
        ]

        config = {
            "exam_periods": default_periods,
            "pre_exam_buffer_days": 7,  # Tag 1 week before exams as "pre-exam"
            "description": "Exam period configuration for library occupancy prediction"
        }

        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

        self.exam_periods = default_periods
        print(f"✓ Created default configuration with {len(default_periods)} exam periods")

    def add_exam_period(self, name, start_date, end_date, exam_type, semester, year):
        """Add a new exam period to configuration"""
        exam_period = {
            "name": name,
            "start_date": start_date,
            "end_date": end_date,
            "type": exam_type,
            "semester": semester,
            "year": year
        }

        self.exam_periods.append(exam_period)
        self.save_config()
        print(f"✓ Added exam period: {name}")

    def save_config(self):
        """Save exam period configuration to JSON file"""
        config = {
            "exam_periods": self.exam_periods,
            "pre_exam_buffer_days": 7,
            "description": "Exam period configuration for library occupancy prediction"
        }

        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Configuration saved to {self.config_file}")

    def get_exam_status(self, date):
        """
        Get detailed exam status for a specific date

        Returns:
            dict: {
                'is_exam': bool,
                'is_pre_exam': bool,
                'type': str ('midterm'/'final'/None),
                'name': str/None
            }
        """
        is_exam, exam_info = self.is_exam_period(date)
        if is_exam:
            return {
                'is_exam': True,
                'is_pre_exam': False,
                'type': exam_info['type'],
                'name': exam_info['name']
            }

        is_pre_exam, exam_info = self.is_pre_exam_period(date, buffer_days=7)
        if is_pre_exam:
            return {
                'is_exam': False,
                'is_pre_exam': True,
                'type': exam_info['type'],
                'name': f"Pre-{exam_info['name']}"
            }

        return {
            'is_exam': False,
            'is_pre_exam': False,
            'type': None,
            'name': None
        }

    def is_exam_period(self, date):
        """Check if a date falls within any exam period"""
        date = pd.to_datetime(date)

        for period in self.exam_periods:
            start = pd.to_datetime(period['start_date'])
            end = pd.to_datetime(period['end_date'])

            if start <= date <= end:
                return True, period

        return False, None

    def is_pre_exam_period(self, date, buffer_days=7):
        """Check if a date falls within pre-exam buffer period"""
        date = pd.to_datetime(date)

        for period in self.exam_periods:
            start = pd.to_datetime(period['start_date'])
            pre_exam_start = start - timedelta(days=buffer_days)

            if pre_exam_start <= date < start:
                return True, period

        return False, None

    def tag_dataframe(self, df, date_column='Start_dt'):
        """
        Add exam period tags to a dataframe

        Args:
            df: DataFrame with date column
            date_column: Name of the date column

        Returns:
            DataFrame with exam period tags added
        """
        df = df.copy()

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])

        # Initialize tagging columns
        df['is_exam_period'] = 0
        df['is_pre_exam_period'] = 0
        df['exam_type'] = None
        df['exam_semester'] = None
        df['exam_name'] = None

        print(f"\nTagging {len(df)} records...")

        # Tag exam periods
        exam_count = 0
        pre_exam_count = 0

        for idx, row in df.iterrows():
            date = row[date_column]

            # Check if in exam period
            is_exam, exam_info = self.is_exam_period(date)
            if is_exam:
                df.at[idx, 'is_exam_period'] = 1
                df.at[idx, 'exam_type'] = exam_info['type']
                df.at[idx, 'exam_semester'] = exam_info['semester']
                df.at[idx, 'exam_name'] = exam_info['name']
                exam_count += 1
            else:
                # Check if in pre-exam period
                is_pre_exam, exam_info = self.is_pre_exam_period(date)
                if is_pre_exam:
                    df.at[idx, 'is_pre_exam_period'] = 1
                    df.at[idx, 'exam_type'] = exam_info['type']
                    df.at[idx, 'exam_semester'] = exam_info['semester']
                    df.at[idx, 'exam_name'] = f"Pre-{exam_info['name']}"
                    pre_exam_count += 1

        print(f"✓ Tagged {exam_count} exam period records")
        print(f"✓ Tagged {pre_exam_count} pre-exam period records")

        return df

    def get_exam_pattern_data(self, df, date_column='Start_dt'):
        """
        Extract historical exam period data for pattern matching

        Returns DataFrame filtered to exam periods only
        """
        tagged_df = self.tag_dataframe(df, date_column)
        exam_data = tagged_df[tagged_df['is_exam_period'] == 1].copy()

        print(f"\n✓ Extracted {len(exam_data)} exam period records for pattern matching")

        return exam_data

    def get_future_exam_dates(self, start_date=None, days_ahead=90):
        """
        Get list of future exam dates within specified range

        Args:
            start_date: Starting date (default: today)
            days_ahead: Number of days to look ahead

        Returns:
            List of exam period dictionaries within date range
        """
        if start_date is None:
            start_date = datetime.now()
        else:
            start_date = pd.to_datetime(start_date)

        end_date = start_date + timedelta(days=days_ahead)

        future_exams = []
        for period in self.exam_periods:
            period_start = pd.to_datetime(period['start_date'])
            period_end = pd.to_datetime(period['end_date'])

            # Check if period overlaps with future range
            if period_start <= end_date and period_end >= start_date:
                future_exams.append(period)

        print(f"\n✓ Found {len(future_exams)} exam periods in next {days_ahead} days")
        return future_exams

    def generate_exam_statistics(self, df, date_column='Start_dt'):
        """Generate statistics on exam period occupancy patterns"""
        tagged_df = self.tag_dataframe(df, date_column)

        # Separate exam and non-exam data
        exam_data = tagged_df[tagged_df['is_exam_period'] == 1]
        regular_data = tagged_df[tagged_df['is_exam_period'] == 0]

        stats = {
            'total_records': len(tagged_df),
            'exam_records': len(exam_data),
            'regular_records': len(regular_data),
            'exam_percentage': (len(exam_data) / len(tagged_df)) * 100 if len(tagged_df) > 0 else 0,
        }

        # Calculate occupancy differences if occupancy column exists
        if 'occupancy' in tagged_df.columns:
            stats['avg_occupancy_exam'] = exam_data['occupancy'].mean()
            stats['avg_occupancy_regular'] = regular_data['occupancy'].mean()
            stats['occupancy_difference'] = stats['avg_occupancy_exam'] - stats['avg_occupancy_regular']
            stats['occupancy_difference_percent'] = (stats['occupancy_difference'] / stats['avg_occupancy_regular']) * 100 if stats['avg_occupancy_regular'] > 0 else 0

        return stats


def main():
    """Demo and testing of exam period tagger"""
    print("=" * 80)
    print("EXAM PERIOD TAGGING SYSTEM - DEMO")
    print("=" * 80)

    # Initialize tagger
    tagger = ExamPeriodTagger()

    # Display configured exam periods
    print("\n" + "=" * 80)
    print("CONFIGURED EXAM PERIODS")
    print("=" * 80)

    for period in tagger.exam_periods:
        print(f"\n{period['name']}")
        print(f"  Type: {period['type'].upper()}")
        print(f"  Dates: {period['start_date']} to {period['end_date']}")
        print(f"  Semester: {period['semester'].title()} {period['year']}")

    # Check if data file exists for demo
    if os.path.exists('all_data_cleaned.csv'):
        print("\n" + "=" * 80)
        print("TAGGING EXISTING DATA")
        print("=" * 80)

        # Load data
        df = pd.read_csv('all_data_cleaned.csv', nrows=1000)  # Sample for demo
        print(f"\nLoaded {len(df)} records for demo")

        # Tag the data
        tagged_df = tagger.tag_dataframe(df)

        # Show sample of tagged data
        exam_sample = tagged_df[tagged_df['is_exam_period'] == 1].head()
        if len(exam_sample) > 0:
            print("\nSample of tagged exam period data:")
            print(exam_sample[['Start_dt', 'is_exam_period', 'exam_type', 'exam_name']].to_string())
        else:
            print("\n⚠ No exam periods found in sample data")

    # Show future exam dates
    print("\n" + "=" * 80)
    print("FUTURE EXAM PERIODS (Next 180 days)")
    print("=" * 80)

    future_exams = tagger.get_future_exam_dates(days_ahead=180)
    for exam in future_exams:
        print(f"\n{exam['name']}")
        print(f"  Dates: {exam['start_date']} to {exam['end_date']}")
        print(f"  Type: {exam['type'].upper()}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nConfiguration saved to: exam_periods_config.json")
    print("Use this module to tag your data before training models.")


if __name__ == "__main__":
    main()
