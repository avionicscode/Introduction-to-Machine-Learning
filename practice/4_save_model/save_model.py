import pandas as pd 
import numpy as np 
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")
df.head()

model = linear_model.LinearRegression()
model.fit(df[['area']],df.price)

model.coef_
model.intercept_
model.predict(5000)

import pickle
with open('model_pickle','wb') as f:
    pickle.dump(model,f)

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

    