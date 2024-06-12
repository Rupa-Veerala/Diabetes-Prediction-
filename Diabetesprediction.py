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

X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

#split train and test data
#0.2 means 20% of test data based on y
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

#Training the model
classifier = svm.SVC(kernel='linear')
#Training the support vector machine classifier
classifier.fit(X_train,Y_train)

#Model evaluation
#Accuracy score
#accuracy score from the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('Accuracy score of the training data :',training_data_accuracy)

#accuracy score from the training data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy score of the test data :',test_data_accuracy)

#Making a predictive system
input_data = ()

#