import pandas as pd
import numpy as np

from joblib import load
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer

import sqlite3

raw_data = pd.read_csv('housing.csv')
model = load('model.joblib')

data = raw_data[['LONGITUDE', 'LAT', 'MEDIAN_AGE', 'ROOMS', 
             'BEDROOMS', 'POP', 'HOUSEHOLDS', 'MEDIAN_INCOME', 
             'OCEAN_PROXIMITY']]

#print(data.head(3))

number_attributes = ['LONGITUDE', 'LAT', 'MEDIAN_AGE', 'ROOMS', 
             'BEDROOMS', 'POP', 'HOUSEHOLDS', 'MEDIAN_INCOME']
non_number_attributes = 'OCEAN_PROXIMITY'

all_categories = [['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']]

#Initialize imputers/encoders/scalers for data preprocessing
knn = KNNImputer(n_neighbors=5)
si = SimpleImputer(strategy='most_frequent') 
sa = StandardScaler()
ohe = OneHotEncoder(handle_unknown='ignore', categories=all_categories)

#Data modifications
numerical_data = data[number_attributes].replace('Null', np.nan)
categorical_data = data[[non_number_attributes]].replace('Null', np.nan)

knn_data = knn.fit_transform(numerical_data)
sim_data = si.fit_transform(categorical_data)
scaled_data = sa.fit_transform(knn_data)
onehot_data = ohe.fit_transform(sim_data)
onehot_data = onehot_data.toarray()
ohe_columns = ohe.get_feature_names_out([non_number_attributes]).tolist()

#print("Shape of scaled_data:", scaled_data.shape)
#print("Shape of onehot_data:", onehot_data.shape)

#print(onehot_data[:3])
#print(scaled_data[:3])
#print(type(scaled_data), type(onehot_data))

preprocessed = np.concatenate((scaled_data, onehot_data), 1)
#print(preprocessed[:3])

#Predict Output
#test_results = model.predict(preprocessed[:10])
#print(test_results)
results = model.predict(preprocessed)

processed_data_df = pd.DataFrame(preprocessed, columns=number_attributes + ohe_columns)
predictions_df = pd.DataFrame(results, columns=['prediction'])

con = sqlite3.connect('housing_predictions.db')
processed_data_df.to_sql('preprocessed_data', con, if_exists='replace', index=False)
predictions_df.to_sql('predictions', con, if_exists='replace', index=False)
con.close()