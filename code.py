import pandas as pd
import matplotlib.py as plt
importaborn as sn
import numpy as np
import
from sklearn.model_selection import train_test_split GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

csv_file_path = r'D:\project prediction of flight delay\Combined_Flights_2018.csv'

if not os.path.isfile(csv_file_path):
    print(f"File '{csv_file_path}' not found or not accessible.")
else:
    try:
        df = pd.read_csv(csv_file_path)
    except pd.errors.EmptyDataError:
        print(f"File '{csv_file_path}' is empty or contains no data.")
    except Exception as e:
        print(f"An error occurred while reading file '{csv_file_path}': {e}")

print("Dataset Info:\n", df.info())

print("\nMissing Values:\n", df.isnull().sum())


df = df.fillna(0)
float_cols = [col for col in df.columns if df[col].dtype == 'float64']
df[float_cols] = df[float_cols].fillna(0)

duplicates = df[df.duplicated()]

print("\nNumber of duplicate rows:\n", len(duplicates))

df = df.drop_duplicates()

print("\nSummary Statistics:\n", df.describe())

df = df[df['Distance'] < 5000]

X = df[['CRSDepTime', 'Distance', 'DayOfWeek', 'Month', 'DepDelay']]
y = df['ArrDelay']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest Decision Tree Regressor parameters found: ", grid_search.best_params_)
print("Lowest MSE found for Decision Tree Regressor: ", np.sqrt(np.abs(grid_search.best_score_)))

random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)


y_pred = random_forest.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error (MSE) for Random Forest Regressor:", mse)
print("R-squared for Random Forest Regressor:", r2)

new_data = pd.DataFrame([[900, 1000, 2, 6, 10]], columns=['CRSDepTime', 'Distance', 'DayOfWeek', 'Month', 'DepDelay'])
new_data = new_data.fillna(0)
predicted_delay = random_forest.predict(new_data)
print("\nPredicted Arrival Delay for Random Forest Regressor:", predicted_delay[0])
print("minutes")
