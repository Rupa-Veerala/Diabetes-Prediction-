#import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#Data collection and analysis
#Loading the diabetes dataset to a pandas dataframes
diabetes_dataset=pd.read_csv('diabetes.csv')

#print the first 5 rows of the dataset
diabetes_dataset.head()

#Number of rows and columns in this dataset
diabetes_dataset.shape

#Getting the statistical measuers of the data
diabetes_dataset.describe()
#'0' represents non-diabetic
#'1' represents diabetc
diabetes_dataset["Outcome"].value_counts()

diabetes_dataset.groupby('Outcome').mean()

#separating the data and labels
X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']
print(X)
print(Y)

#Data Standardization
scaler=StandardScaler()
scaler.fit(X)
standardized_data=scaler.transform(X)
print(standardized_data)


