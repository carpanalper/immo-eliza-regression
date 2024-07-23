# Immo Eliza Machine Learning Project

## Description

This project aims to create and evaluate machine learning models on a real estate dataset. The project includes data cleaning, feature engineering, model training, and testing. Algorithms used include Linear Regression, Random Forest, and XGBoost.

## Repository Structure

This repository contains the following files and directories:

- `cleaning.py` - Script for data cleaning and preprocessing.
- `training.py` - Script for training machine learning models.
- `testing.py` - Script for testing trained models and evaluating their performance.
- `linear_regressor.pkl` - Pickled file containing the trained Linear Regression model.
- `forest_model.pkl` - Pickled file containing the trained Random Forest model.
- `xgb_regressor.pkl` - Pickled file containing the trained XGBoost model.
- `README.md` - This README file containing project details and instructions.

- `final_dataset.json` - The raw dataset used for cleaning and preprocessing. (to be downloaded)
- `clean_immo.csv` - The cleaned and preprocessed dataset ready for model training.
ps:`clean_immo.csv` will be created automatically after running; `cleaning.py`

## Instructions

### Prerequisites

Ensure that you have the following packages installed and the raw data downloaded.

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`

You can install the required packages using the provided `requirements.txt` file:

```sh
pip install -r requirements.txt
```
**Raw Data File**: Please download the raw data from: `https://drive.google.com/file/d/1Ww_pnsQpAGj_LRGUNji6yFgeFeV9pzrd/view?usp=drive_link`

### Running the Scripts

1. **Data Cleaning (`cleaning.py`)**:
   - This script performs data cleaning and preprocessing on the raw dataset.
   - It reads `final_dataset.json`, processes the data, and saves the cleaned data to `clean_immo.csv`.
   - Run the script using the following command:

     ```sh
     python cleaning.py
     ```

2. **Model Training (`training.py`)**:
   - This script trains three machine learning models: Linear Regression, Random Forest, and XGBoost.
   - It reads the cleaned data from `clean_immo.csv`, trains the models, and saves them as pickle files.
   - Run the script using the following command:

     ```sh
     python training.py
     ```

3. **Model Testing (`testing.py`)**:
   - This script loads the trained models, tests them on the test set, and evaluates their performance.
   - It computes Mean Absolute Error (MAE) and R^2 scores, and prints the results.
   - Run the script using the following command:

     ```sh
     python testing.py
     ```

### The Workflow

1. **Load and Clean Data**:
   - The raw data from `final_dataset.json` file is read.
   - Properties of type 'for sale' are filtered.
   - Irrelevant columns and missing values are cleaned.
   - Qualitative values are converted to numerical values.
   - Outliers are removed and missing values are imputed using KNN Imputer.
   - Log transformations are applied, and dummies are created for categorical variables.

   The cleaned data is saved to `clean_immo.csv`. 

### Model Training

1. **Load and Split Data**:
   - The `clean_immo.csv` file is read.
   - Features (`X`) and target variable (`y`) are separated.
   - Data is split into training and test sets.
   - Data is scaled.

2. **Train and Save Models**:
   - **Linear Regression**: The model is trained and saved to `linear_regressor.pkl`.
   - **Random Forest**: The model is trained and saved to `forest_model.pkl`.
   - **XGBoost**: The model is trained and saved to `xgb_regressor.pkl`.

### Model Testing

1. **Load and Test Models**:
   - Trained models are loaded.
   - Models are evaluated on the test set.
   - Mean Absolute Error (MAE) and R^2 scores are computed.
   - Results are printed and a comparison table is created.

## Results

### Linear Regression
- **Mean Absolute Error (MAE)**: 111917.38785758428
- **R^2 Score**: 0.545100

### Random Forest
- **Mean Absolute Error (MAE)**: 33987.641195
- **R^2 Score**: 0.934661

### XGBoost
- **Mean Absolute Error (MAE)**: 99170.422124
- **R^2 Score**: 0.611674

## Conclusion

Based on the results, Random Forest achieved the best performance with the lowest MAE and the highest R^2 score. 

## Contributors

Contributors to this project:
- Alper Carpan

## Timeline

Major steps and estimated duration:
- Data Preprocessing: 18-19 July 2024
- Model Training: 22 July 2024
- Model Testing and Results Analysis: 23 July 2024