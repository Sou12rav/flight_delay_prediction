import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

csv_file_path = r'https://github.com/Sou12rav/flight_delay_prediction/blob/25fe9b82a6833413a8a1be572ab92df03ef1ad0d/example_dataset.csv'

try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"File not found at path: {csv_file_path}")

print("Dataset Info:\n", df.info()) #basic info

# missing values
print("\nMissing Values:\n", df.isnull().sum())

#replacing the nan values
df = df.fillna(0)
float_cols = [col for col in df.columns if df[col].dtype == 'float64']
df[float_cols] = df[float_cols].fillna(0)

# duplicate data
duplicates = df[df.duplicated()]

#number of duplicates
print("\nNumber of duplicate rows:\n", len(duplicates))

# Remove duplicate data
df = df.drop_duplicates()

# Display summary statistics
print("\nSummary Statistics:\n", df.describe())

# Identify and remove data outliers(outliers: here the distance which is quite high and making it troublesome to show)
df = df[df['Distance'] < 5000]

#number of flights per month
monthly_flight = df.groupby('Month')['Month'].count()
plt.figure(figsize=(10, 6))
plt.bar(monthly_flight.index, monthly_flight.values, alpha=0.8)
plt.title('Number of Flights per Month')
plt.xlabel('Month')
plt.ylabel('Number of Flights')
plt.show()
# number of flights per origin airport
origin_flight = df.groupby('Origin')['Origin'].count()
plt.figure(figsize=(10, 6))
plt.bar(origin_flight.index, origin_flight.values, alpha=0.8)
plt.title('Number of Flights per Origin Airport')
plt.xlabel('Origin Airport')
plt.ylabel('Number of Flights')
plt.show()
#distribution of arrival delays
plt.figure(figsize=(10, 6))
sns.histplot(df['ArrDelay'], bins=30, kde=True)
plt.title('Distribution of Arrival Delays')
plt.xlabel('Arrival Delay (minutes)')
plt.ylabel('Frequency')
plt.show()

#departure delay vs. departure time
plt.figure(figsize=(10, 6))
sns.scatterplot(x='CRSDepTime', y='DepDelay', data=df)
plt.title('Departure Delay vs. Departure Time')
plt.xlabel('Departure Time (24-hour clock)')
plt.ylabel('Departure Delay (minutes)')
plt.show()

# Display summary statistics after removing outliers
print("\n Summary Statistics after removing outliers: \n", df.describe())

#departure delay and arrival delay
plt.figure(figsize=(10, 6))
sns.scatterplot(x='DepDelay', y='ArrDelay', data=df)
plt.title('Departure Delay vs Arrival Delay')
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Arrival Delay (minutes)')
plt.show()

#flight delays for each airline
flight_delay_by_airline = df.groupby('Airline')['DepDelay'].describe()
plt.figure(figsize=(10, 6))
plt.bar(flight_delay_by_airline.index, flight_delay_by_airline['mean'])
plt.title('Mean Flight Delay by Airline')
plt.xlabel('Airline')
plt.ylabel('Mean Flight Delay (minutes)')
plt.show()

#flight delays for each origin airport
flight_delay_by_origin = df.groupby('Origin')['DepDelay'].describe()
plt.figure(figsize=(10, 6))
plt.bar(flight_delay_by_origin.index, flight_delay_by_origin['mean'])
plt.title('Mean Flight Delay by Origin Airport')
plt.xlabel('Origin Airport')
plt.ylabel('Mean Flight Delay (minutes)')
plt.show()
