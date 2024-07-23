import pickle
import time
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Reading Dataset...")
dfsale = pd.read_csv('clean_immo.csv')

X = dfsale.drop(["Price"], axis=1)
y = dfsale["Price"]

print("Splitting data into train and test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
time.sleep(1)

print("Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
time.sleep(1)

with open('linear_regressor.pkl', 'rb') as file:
    loaded_regressor = pickle.load(file)

print("Testing LinearRegression")
y_pred = loaded_regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
time.sleep(1)

print('LinearRegression Scores:')
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")
time.sleep(1)

with open('forest_model.pkl', 'rb') as f:
    loaded_forest = pickle.load(f)

print("Testing Random Forest")
y_pred_f = loaded_forest.predict(X_test_scaled)
mae_f = mean_absolute_error(y_test, y_pred_f)
r2_f = r2_score(y_test, y_pred_f)
time.sleep(1)

print('RandomForest Scores:')
print(f"Mean Absolute Error: {mae_f}")
print(f"R^2 Score: {r2_f}")
time.sleep(1)


with open('xgb_regressor.pkl', 'rb') as f:
    loaded_xgb = pickle.load(f)

print("Testing XGBoost")
y_pred_xgb = loaded_xgb.predict(X_test_scaled)
r2_xgb = loaded_xgb.score(X_test_scaled, y_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
time.sleep(1)

print('XGBoost Scores:')
print('Mean Absolute Error:', mae_xgb)
print('R^2 Score:', r2_xgb)
time.sleep(1)

results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'MAE': [mae, mae_f, mae_xgb],
    'R^2': [r2, r2_f, r2_xgb]
})
print("\nModel Comparison:")
print(results)