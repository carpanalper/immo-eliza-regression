print('Importing packages...')
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle

print('Reading Dataset...')
dfsale = pd.read_csv('clean_immo.csv')

X = dfsale.drop(["Price"],axis=1)
y = dfsale["Price"]

print("Splitting data to train and test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
time.sleep(1)

print("Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
time.sleep(1)

print("Training with Linear Regression...")
reg = LinearRegression()
reg.fit(X_train, y_train)

with open('linear_regressor.pkl', 'wb') as file:
    pickle.dump(reg, file)

print("Training with Random Forest...")
forest = RandomForestRegressor(random_state=42)
forest.fit(X_train_scaled,y_train)

with open('forest_model.pkl', 'wb') as f:
    pickle.dump(forest, f)

print("Training with XGBoost")
model = XGBRegressor(n_estimators=100,random_state=42)
model.fit(X_train_scaled, y_train)

with open('xgb_regressor.pkl', 'wb') as f:
    pickle.dump(model, f)