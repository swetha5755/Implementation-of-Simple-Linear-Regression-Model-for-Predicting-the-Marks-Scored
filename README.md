# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages. 
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.

## Program:

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Swetha S
RegisterNumber: 212224040344
*/

python
# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)



## Output:

To Read Head and Tail Files

<img width="189" height="149" alt="Screenshot 2025-08-29 153214" src="https://github.com/user-attachments/assets/51540895-4509-4a43-b675-e37b9d46b7af" />

<img width="225" height="155" alt="Screenshot 2025-08-29 153307" src="https://github.com/user-attachments/assets/52e76fbd-6987-451f-bdbc-e0a0d61ec4fd" />



Compare Dataset

<img width="753" height="586" alt="Screenshot 2025-08-29 153347" src="https://github.com/user-attachments/assets/15dd9bc9-cc78-4671-a231-c1ff5cb8a1c3" />



Predicted Value

<img width="833" height="76" alt="Screenshot 2025-08-29 153423" src="https://github.com/user-attachments/assets/90a07346-43af-4da7-9bc7-f70cacbddd4a" />




Graph For Training Set

<img width="930" height="698" alt="Screenshot 2025-08-29 153500" src="https://github.com/user-attachments/assets/630df118-2fea-45af-84c4-0ac9157cf3e7" />



Graph For Testing Set

<img width="893" height="702" alt="Screenshot 2025-08-29 153551" src="https://github.com/user-attachments/assets/9980206e-7a86-4276-a9e3-3ac66dd706b9" />



Error

<img width="507" height="79" alt="Screenshot 2025-08-29 153635" src="https://github.com/user-attachments/assets/cd4403c9-252a-4008-b259-1aa2754d0bc4" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
