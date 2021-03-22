import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Importing data
purchase_data = pd.read_csv("C:/Users/benuk/Desktop/benazir/Benazir/Datasets/Social_Network_Ads.csv")
purchase_data.head(5)
purchase_data.shape

# Analyzing data
sns.countplot(x="Purchased", data = purchase_data)
sns.boxplot(x="Purchased", y="Age", data=purchase_data)
sns.boxplot(x="Purchased", y="EstimatedSalary", data=purchase_data)
purchase_data["Age"].plot.hist()
purchase_data.isnull()                             # to check if there is any null value
purchase_data.isnull().sum()                       # checking null value for each field

purchase_data.drop("User ID", axis=1, inplace=True)  # removing unwanted fields
purchase_data.head(2)

gender = pd.get_dummies(purchase_data["Gender"], drop_first=True)   # converting string values to int/binary
gender.head(2)
purchase_data = pd.concat([purchase_data,gender], axis=1)
purchase_data.drop("Gender", axis=1, inplace=True)
purchase_data.head(2)

# Separating predictor and target variables
X = purchase_data.drop("Purchased", axis=1)
y = purchase_data["Purchased"]

# Splitting training and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Creating model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Testing and Prdictions
predictions =model.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
from sklearn.metrics import classification_report
classification_report(y_test, predictions)
