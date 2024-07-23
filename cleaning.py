print("Importing Packages...")
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import time

print("Reading the dataset...")
df = pd.read_json('final_dataset.json')

print("Filtering 'for sale' Properties...")
dfsale = df[df['TypeOfSale'] == 'residential_sale']
time.sleep(1)

dfsale = dfsale.copy()
print("Dropping Irrelavant Columns...")
dfsale.drop(['Url','Country','Fireplace','MonthlyCharges','PropertyId','TypeOfSale',
             'GardenArea','NumberOfFacades','RoomCount','ShowerCount','ToiletCount','FloodingZone','Locality',
             'FloodingZone'], axis=1, inplace=True)
time.sleep(1)

print("Dropping Missing Values...")
dfsale.dropna(subset=['District','Province','Region','LivingArea'], how = 'any', inplace=True)
time.sleep(1)

print("Turning Qualitative Values into Numerical Values...")
rank_state = {
    'AS_NEW': 6,
    'JUST_RENOVATED': 5,
    'GOOD': 4,
    'TO_RESTORE': 3,
    'TO_RENOVATE': 2,
    'TO_BE_DONE_UP': 1
}

PEB_to_drop = ['B_A', 'A_A+', 'F_C', 'F_D', 'F_E', 'E_D', 'G_F', 'G_C']
dfsale = dfsale[~dfsale['PEB'].isin(PEB_to_drop)]

rank_PEB = {
    'G': 1, 
    'F': 2, 
    'E': 3, 
    'D': 4, 
    'C': 5, 
    'B': 6, 
    'A': 7, 
    'A+': 8, 
    'A++': 9
}

rank_kitchen = {
    'USA_UNINSTALLED':1,
    'NOT_INSTALLED': 1, 
    'USA_SEMI_EQUIPPED':2,
    'SEMI_EQUIPPED': 2, 
    'USA_INSTALLED' : 3,
    'INSTALLED': 3, 
    'HYPER_EQUIPPED': 4,
    'USA_HYPER_EQUIPPED':4
     }

rank_district = {
    'Charleroi': 2.0,
    'Philippeville':2.0,
    'Mons': 2.0,
    'Thuin': 2.5,
    'Dinant': 2.5,
    'Mouscron': 2.5,
    'Soignies': 2.5,
    'Ath':2.5,
    'Liège': 2.5,
    'Huy': 3.0,
    'Neufchâteau': 3.0,
    'Virton': 3.0,
    'Namur': 3.0,
    'Ieper': 3.0,
    'Tournai': 3.0,
    'Roeselare': 3.0,
    'Oostend': 3.0,
    'Waremme': 3.0,
    'Diksmuide': 3.0,
    'Tongeren': 3.0,
    'Verviers': 3.0,
    'Marche-en-Famenne':3.0,
    'Kortrijk': 3.5,
    'Tielt' : 3.5,
    'Eeklo': 3.5,
    'Arlon': 3.5,
    'Hasselt': 3.5,
    'Bastogne': 3.5,
    'Aalst': 3.5,
    'Oudenaarde': 3.5,
    'Sint-Niklaas': 3.5,
    'Turnhout': 3.5,
    'Maaseik': 3.5,
    'Veurne': 3.5,
    'Dendermonde': 4.0,
    'Mechelen': 4.0,
    'Brugge': 4.0,
    'Gent': 4.0,
    'Antwerp': 4.5,
    'Leuven': 4.5,
    'Halle-Vilvoorde': 4.5,
    'Nivelles': 5.0,
    'Brussels': 5.0
}

rank_province = {
    'Brussels': 5.0,
    'Walloon Brabant': 4.9,
    'Flemish Brabant': 4.7,
    'Antwerp': 4.3,
    'East Flanders': 4.0,
    'Limburg': 3.7,
    'West Flanders': 3.6,
    'Luxembourg': 3.4,
    'Liège': 3.1,
    'Namur': 3.0,
    'Hainaut': 2.7
}

dfsale['Condition_Rank'] = df['StateOfBuilding'].map(rank_state)
dfsale['PEB_Rank'] = dfsale['PEB'].map(rank_PEB)
dfsale['Kitchen_Rank'] = dfsale['Kitchen'].map(rank_kitchen)
dfsale['District_Rank'] = dfsale['District'].map(rank_district)
dfsale['Province_Rank'] = dfsale['Province'].map(rank_province)

time.sleep(1)

print("Dropping Columns with Qualitative Values...")

dfsale.drop(['Kitchen','PEB','StateOfBuilding','District',
             'PostalCode','Province','SubtypeOfProperty'], axis=1, inplace=True)

time.sleep(1)

print("Filling Missing Values...")

dfsale['Furnished'] = dfsale['Furnished'].fillna(0)
dfsale['Garden'] = dfsale['Garden'].fillna(0)
dfsale['SwimmingPool'] = dfsale['SwimmingPool'].fillna(0)
dfsale['Terrace'] = dfsale['Terrace'].fillna(0)

time.sleep(1)

print("Removing Outliers...")
      
dfsale["ppsqm"] = df["Price"] / df["LivingArea"]

Q1 = dfsale['ppsqm'].quantile(0.25)
Q3 = dfsale['ppsqm'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

dfsale = dfsale[(dfsale['ppsqm'] > lower_bound) & (dfsale['ppsqm'] < upper_bound)]      
      
dfsale.drop(dfsale[dfsale.BathroomCount > 40].index,inplace=True)
dfsale.drop(dfsale[dfsale.BedroomCount > 50].index,inplace=True)
dfsale.drop(dfsale[dfsale.ConstructionYear > 2028].index,inplace=True)
dfsale.drop(dfsale[dfsale.LivingArea > 4000].index,inplace=True)
dfsale.drop(dfsale[dfsale.SurfaceOfPlot > 150000].index,inplace=True)      
dfsale.drop(['ppsqm'], axis=1, inplace=True)
      
time.sleep(1)      
          
print("Filling Missing Values with KNN Imputer...(about 20 minutes)")

columns_to_impute = ['BathroomCount','ConstructionYear','SurfaceOfPlot','Condition_Rank','PEB_Rank','Kitchen_Rank']

imputer = KNNImputer(n_neighbors=5)

df_imputed_values = imputer.fit_transform(dfsale[columns_to_impute])

dfsale[columns_to_impute] = df_imputed_values
print("Done!")

print("Arranging Data Types...")
dfsale = dfsale.astype({'BathroomCount': 'int64', 'ConstructionYear': 'int64', 'Furnished': 'int64', 'Garden': 'int64', 
                        'LivingArea': 'int64', 'SurfaceOfPlot': 'int64', 'SwimmingPool': 'int64', 'Terrace': 'int64',
                        'Condition_Rank':'int64','PEB_Rank':'int64','Kitchen_Rank':'int64'})

time.sleep(1)    
print("Log Transformation...")
dfsale["LivingArea"] = np.log(dfsale['LivingArea']+ 1)
dfsale["BathroomCount"] = np.log(dfsale['BathroomCount']+ 1)
dfsale["BedroomCount"] = np.log(dfsale['BedroomCount']+ 1)
dfsale["SurfaceOfPlot"] = np.log(dfsale['SurfaceOfPlot']+ 1)

time.sleep(1)    
print("Getting Dummies...")
dfsale = dfsale.join(pd.get_dummies(dfsale.Region)).drop(['Region'], axis=1)

time.sleep(1)    
print("Saving...")
dfsale.to_csv('clean_immo.csv',index=False)
time.sleep(1)    
print("The dataframe is ready for the training: 'clean_immo.csv")