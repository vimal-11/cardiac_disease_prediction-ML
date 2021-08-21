import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.preprocessing import StandardScaler as sc

df = pd.read_csv("heart_data.csv")
#use required features
cdf = df[['age','cp','thal','thalach','oldpeak','ca','target']]

#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
x = cdf.iloc[:, :-1].values
y = cdf.iloc[:, -1].values
regressor = RandomForestClassifier()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model.pkl','wb'))


#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict(np.array([[20,0,2,160,1.2,0]])))