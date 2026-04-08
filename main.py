import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.DataFrame({
    'hours_studied': [2,3,4,5,6,7,8,9,10],
    'attendance': [60,65,70,75,80,85,90,95,100],
    'previous_score': [50,55,60,65,70,75,80,85,90],
    'gender': [0,1,0,1,0,1,0,1,0],
    'study_time': [1,2,1,2,3,2,3,3,3],
    'final_score': [55,60,65,70,75,80,85,90,95]
})

X = data[['hours_studied','attendance','previous_score','gender','study_time']]
y = data['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
