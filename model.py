# Importing the required libraries.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Importing modules for evaluating the performance of the model.
from sklearn.metrics import r2_score,classification_report,confusion_matrix,accuracy_score
import pickle

# Reading the required dataset.
data = pd.read_csv('delivery_data_final.csv')                                  

x = data.drop('ontime', axis=1)
y = data['ontime']

#Encoding.
for column  in x.columns:
  if(x[column].dtype =='object'):
      x[column]=x[column].astype("category").cat.codes

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.25)

#Building the model.
rfc_cv=RandomForestClassifier(bootstrap= True,max_depth=20,max_features= 'auto',min_samples_leaf= 1,
                              min_samples_split=2,n_estimators= 500, criterion='gini')

rfc_cv.fit(x_train,y_train)
y_pred = rfc_cv.predict(x_test)

print(rfc_cv.score(x_test,y_test))
print(r2_score(y_test, y_pred))

# Writing the model into a pickle file
pickle.dump(rfc_cv, open('model.pickle', 'wb'))
