#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv('salary_predict_dataset.csv')


df['test_score'].fillna(df['test_score'].mean(), inplace = True)
df['interview_score'].fillna(df['interview_score'].mean(), inplace = True)
df['experience'].fillna(0, inplace = True)

# We need to convert string values of experinece to integer/numerical
#so we will create a function string_to_num

def string_to_num(word):
    dic = {'one':1, 'two': 2, 'three': 3, 'four': 4, 'five':5, 'six':6, 'seven': 7, 'eight':8, 'nine':9, 'ten':10, 
           'eleven': 11, 'twelve':12, 'thirteen':13, 'fourteen':14, 'fifteen':15, '0':0, 0:0}
    return dic[word]

df['experience'] = df['experience'].apply(lambda x: string_to_num(x))

X = df.iloc[:,:3]
Y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size = 0.10, random_state = 5)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xtrain,ytrain)


pickle.dump(lr, open('model.pkl', 'wb'))

y_pred = lr.predict(xtest)

#load model to compare result
model = pickle.load(open('model.pkl', 'rb'))

#experience = 5 years, test_score = 8, interview_score = 7 then predict what will be the salary
print(model.predict([[5,8,7]]))

#print("salary =", y)

