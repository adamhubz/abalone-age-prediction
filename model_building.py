import pandas as pd
import numpy as np
# Load data train
abalone = pd.read_csv('datasets/abalone.csv')

# Data Preprocessing
abalone['trans_length'] = (abalone['length']) ** 2
abalone['trans_diameter'] = abalone['diameter'] ** 2

for num in ['height', 'whole_wt', 'shucked_wt', 'viscera_wt', 'shell_wt', 'rings', 'age']:
	abalone['trans_'+num] = np.sqrt(abalone[num])

# Drop Unnecessary Features
abalone_trans = abalone.drop(columns = ['length', 'diameter','height', 'whole_wt', 'shucked_wt', 'viscera_wt', 'shell_wt', 'rings', 'age'])

# Scaling Data
from sklearn.preprocessing import RobustScaler
features = abalone_trans.drop(columns = ['trans_rings', 'trans_age'])
X = pd.get_dummies(features, columns = ['sex'])
y = abalone_trans['trans_age']

# StandardScaler data
# Initialize StandardScaler into scaler
scaler = RobustScaler()

# Fit X and transform menjadi X_scaled
scaler.fit(X)
X_scaled = scaler.transform(X)

# assign 21 into SEED for reproductivity
SEED = 21

# Machine Learning Modelling
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state = SEED)
dt.fit(X_scaled, y)

# Saving the scaler, pca, model
import pickle
pickle.dump(scaler, open('scaler_reg.pkl', 'wb'))
pickle.dump(dt, open('model_reg.pkl', 'wb'))

