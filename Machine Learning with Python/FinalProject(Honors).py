import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv'
await download(path, "Weather_Data.csv")
filename ="Weather_Data.csv"
df = pd.read_csv("Weather_Data.csv")
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)
df_sydney_processed.drop('Date',axis=1,inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']

# Q1) Use the train_test_split function to split the features and Y dataframes with a test_size of 0.2 and the random_state set to 10.
x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=10)

# Q2) Create and train a Linear Regression model called LinearReg using the training data (x_train, y_train).
LinearReg = LinearRegression()
LinearReg.fit(x_train, y_train)

# Q3) Now use the predict method on the testing data (x_test) and save it to the array predictions.
predictions = LinearReg.predict(x_test)

# Q4) Using the predictions and the y_test dataframe calculate the value for each metric using the appropriate function.
LinearRegression_MAE = metrics.mean_absolute_error(y_test, predictions)
LinearRegression_MSE = metrics.mean_squared_error(y_test, predictions)
LinearRegression_R2 = metrics.r2_score(y_test, predictions)

# Q5) Show the MAE, MSE, and R2 in a tabular format using data frame for the linear model.
report = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'R2'])
report = report.append({'Model': 'Linear Regression', 'MAE': LinearRegression_MAE, 'MSE': LinearRegression_MSE, 'R2': LinearRegression_R2}, ignore_index=True)

# Q6) Create and train a KNN model called KNN using the training data (x_train, y_train) with the n_neighbors parameter set to 4.
KNN = KNeighborsClassifier(n_neighbors=4)
KNN.fit(x_train, y_train)

# Q7) Now use the predict method on the testing data (x_test) and save it to the array predictions.
predictions = KNN.predict(x_test)

# Q8) Using the predictions and the y_test dataframe calculate the value for each metric using the appropriate function.
KNN_Accuracy_Score = accuracy_score(y_test, predictions)
KNN_JaccardIndex = jaccard_score(y_test, predictions)
KNN_F1_Score = f1_score(y_test, predictions)

# Q9) Create and train a Decision Tree model called Tree using the training data (x_train, y_train).
Tree = DecisionTreeClassifier()
Tree.fit(x_train, y_train)

# Q10) Now use the predict method on the testing data (x_test) and save it to the array predictions.
predictions = Tree.predict(x_test)

# Q11) Using the predictions and the y_test dataframe calculate the value for each metric using the appropriate function
Tree_Accuracy_Score = accuracy_score(y_test, predictions)
Tree_JaccardIndex = jaccard_score(y_test, predictions)
Tree_F1_Score = f1_score(y_test, predictions)

# Q12) Use the train_test_split function to split the features and Y dataframes with a test_size of 0.2 and the random_state set to 1
x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=1)

# Q13) Create and train a LogisticRegression model called LR using the training data (x_train, y_train) with the solver parameter set to liblinear.
LR = LogisticRegression(solver='liblinear')
LR.fit(x_train, y_train)

# Q14) Now, use the predict and predict_proba methods on the testing data (x_test) and save it as 2 arrays predictions and predict_proba.
predictions = LR.predict(x_test)
predict_proba = LR.predict_proba(x_test)

# Q15) Using the predictions, predict_proba and the y_test dataframe calculate the value for each metric using the appropriate function
LR_Accuracy_Score = accuracy_score(y_test, predictions)
LR_JaccardIndex = jaccard_score(y_test, predictions)
LR_F1_Score = f1_score(y_test, predictions)
LR_Log_Loss = log_loss(y_test, predict_proba)

# Q16) Create and train a SVM model called SVM using the training data (x_train, y_train).
SVM = svm.SVC()
SVM.fit(x_train, y_train)

# Q17) Now use the predict method on the testing data (x_test) and save it to the array predictions.
predictions = SVM.predict(x_test)

# Q18) Using the predictions and the y_test dataframe calculate the value for each metric using the appropriate function
SVM_Accuracy_Score = accuracy_score(y_test, predictions)
SVM_JaccardIndex = jaccard_score(y_test, predictions)
SVM_F1_Score = f1_score(y_test, predictions)

#Q19) Show the Accuracy,Jaccard Index,F1-Score and LogLoss in a tabular format using data frame for all of the above models.
#*LogLoss is only for Logistic Regression Model

# Create a DataFrame to store the metrics
report = pd.DataFrame(columns=['Model', 'Accuracy', 'Jaccard Index', 'F1-Score', 'Log Loss'])

# Add the metrics for Linear Regression
report = report.append({'Model': 'Linear Regression', 'Accuracy': LR_Accuracy_Score, 'Jaccard Index': '', 'F1-Score': '', 'Log Loss': ''}, ignore_index=True)

# Add the metrics for KNN
report = report.append({'Model': 'KNN', 'Accuracy': KNN_Accuracy_Score, 'Jaccard Index': KNN_JaccardIndex, 'F1-Score': KNN_F1_Score, 'Log Loss': ''}, ignore_index=True)

# Add the metrics for Decision Tree
report = report.append({'Model': 'Decision Tree', 'Accuracy': Tree_Accuracy_Score, 'Jaccard Index': Tree_JaccardIndex, 'F1-Score': Tree_F1_Score, 'Log Loss': ''}, ignore_index=True)

# Add the metrics for Logistic Regression
report = report.append({'Model': 'Logistic Regression', 'Accuracy': LR_Accuracy_Score, 'Jaccard Index': LR_JaccardIndex, 'F1-Score': LR_F1_Score, 'Log Loss': LR_Log_Loss}, ignore_index=True)

# Add the metrics for SVM
report = report.append({'Model': 'SVM', 'Accuracy': SVM_Accuracy_Score, 'Jaccard Index': SVM_JaccardIndex, 'F1-Score': SVM_F1_Score, 'Log Loss': ''}, ignore_index=True)

# Display the DataFrame
print(report)

