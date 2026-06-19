import pandas as pd
import numpy as np
import random
import os
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Task 1: Data Generation
    os.makedirs('sessions', exist_ok=True)
    df_original = pd.read_csv('new-data.csv')
    df_original = df_original.dropna(subset=['Name (original name)']).copy()

    start_date = datetime(2026, 6, 1)

    print("Generating 40 session files...")
    for i in range(1, 41):
        session_date = start_date + timedelta(days=i-1)

        # Sample exactly 980 students (rows)
        # Using replace=True to avoid error if unique students < 980,
        # or maybe we should just sample from the DataFrame directly which has 2163 rows.
        # df_original has 2163 rows. So replace=False works.
        df_sample = df_original.sample(n=980, replace=False).copy()

        join_times = []
        leave_times = []
        durations = []

        for _ in range(980):
            join_hour = 17
            join_minute = random.randint(0, 59)
            join_second = random.randint(0, 59)
            join_dt = session_date.replace(hour=join_hour, minute=join_minute, second=join_second)

            max_leave_dt = session_date.replace(hour=19, minute=0, second=0)
            max_duration_sec = int((max_leave_dt - join_dt).total_seconds())

            leave_dt = join_dt + timedelta(seconds=random.randint(0, max_duration_sec))
            duration_mins = int((leave_dt - join_dt).total_seconds() / 60)

            join_times.append(join_dt.strftime('%m/%d/%Y %H:%M:%S'))
            leave_times.append(leave_dt.strftime('%m/%d/%Y %H:%M:%S'))
            durations.append(duration_mins)

        df_sample['Join time'] = join_times
        df_sample['Leave time'] = leave_times
        df_sample['Duration (minutes)'] = durations

        df_sample.to_csv(f'sessions/session_{i}.csv', index=False)

    # Task 2: EDA & Preprocessing
    print("Combining and cleaning data...")
    all_files = glob.glob('sessions/session_*.csv')
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)

    df_combined = pd.concat(df_list, ignore_index=True)

    # Extract date
    df_combined['Date'] = pd.to_datetime(df_combined['Join time']).dt.date

    # Handle reconnects: aggregate by Name and Date
    df_cleaned = df_combined.groupby(['Name (original name)', 'Date'], as_index=False).agg({
        'Duration (minutes)': 'sum'
    })

    # Task 3: Feature Engineering
    print("Feature engineering...")
    attendance_counts = df_cleaned.groupby('Name (original name)')['Date'].count().reset_index()
    attendance_counts.rename(columns={'Date': 'Total Sessions Attended'}, inplace=True)

    attendance_counts['Certified'] = attendance_counts['Total Sessions Attended'].apply(
        lambda x: 'Yes' if x >= 32 else 'No'
    )

    attendance_counts.to_csv('final_attendance_data.csv', index=False)

    # Task 4: Visualization
    print("Generating visualization...")
    plt.figure(figsize=(10, 6))

    # Adding a jitter to x to make the scatter plot readable, or use index
    attendance_counts['Student_ID'] = range(len(attendance_counts))

    sns.scatterplot(
        data=attendance_counts,
        x='Student_ID',
        y='Total Sessions Attended',
        hue='Certified',
        palette={'Yes': 'green', 'No': 'red'},
        alpha=0.7
    )
    plt.title('Certified vs Uncertified Students based on Attendance Counts')
    plt.xlabel('Student ID')
    plt.ylabel('Total Sessions Attended')
    plt.axhline(32, ls='--', color='blue', label='Threshold (32 Sessions)')
    plt.legend()
    plt.show()
    plt.savefig('certification_scatter.png')
    print("Done!")

if __name__ == '__main__':
    main()
