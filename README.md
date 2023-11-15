# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
# titanic_dataset.csv :

```
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from google.colab import files
upload = files.upload()
df = pd.read_csv('titanic_dataset.csv')
df
```

<img width="561" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex-07/assets/133136880/66d7bc97-67bb-4568-af57-2b2334fc6073">

```
df.isnull().sum()
```

<img width="157" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex-07/assets/133136880/908c919d-83df-42a8-9a85-8f7d0307f8dc">

```
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df[['Age']] = imputer.fit_transform(df[['Age']])
print("Feature selection")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
selector = SelectKBest(chi2, k=3)
X_new = selector.fit_transform(X, y)
print(X_new)
```

<img width="176" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex-07/assets/133136880/1227ffe4-e2d3-42cf-a90f-0dd4b50ccd90">

```
df_new = pd.DataFrame(X_new, columns=['Pclass', 'Age', 'Fare'])
df_new['Survived'] = y.values
df_new.to_csv('titanic_transformed.csv', index=False)
print(df_new)
```

<img width="233" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex-07/assets/133136880/508e0743-7fdb-41a5-b9be-c7e543f066e0">


# CarPrice.csv:

```
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("CarPrice.csv")
df
```

<img width="436" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex-07/assets/133136880/9b4de9c0-cdb7-45a8-a45a-b5a419bf64a0">

```
df = df.drop(['car_ID', 'CarName'], axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['fueltype'] = le.fit_transform(df['fueltype'])
df['aspiration'] = le.fit_transform(df['aspiration'])
df['doornumber'] = le.fit_transform(df['doornumber'])
df['carbody'] = le.fit_transform(df['carbody'])
df['drivewheel'] = le.fit_transform(df['drivewheel'])
df['enginelocation'] = le.fit_transform(df['enginelocation'])
df['enginetype'] = le.fit_transform(df['enginetype'])
df['cylindernumber'] = le.fit_transform(df['cylindernumber'])
df['fuelsystem'] = le.fit_transform(df['fuelsystem'])
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("Univariate Selection")
selector = SelectKBest(score_func=f_regression, k=10)
X_train_new = selector.fit_transform(X_train, y_train)
mask = selector.get_support()
selected_features = X_train.columns[mask]
model = ExtraTreesRegressor()
model.fit(X_train, y_train)
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
selected_features = X_train.columns[indices][:10]
df_new = pd.concat([X_train[selected_features], y_train], axis=1)
df_new.to_csv('CarPrice_new.csv', index=False)
print(df_new)
```

<img width="410" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex-07/assets/133136880/617e59d1-7adc-4673-b591-b6274a3ba107">

# RESULT:

Thus, the various feature selection techniques have been performed on a given dataset successfully.








# OUPUT
