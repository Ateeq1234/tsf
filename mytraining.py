import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


if __name__=="__main__":
    df= pd.read_csv('data.csv')
    X = df.iloc[:,:-1].values
    Y = df.iloc[:,-1].values

    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
    
    linear_reg = LinearRegression()
    linear_reg.fit(x_train,y_train)
    
    
    # line = linear_reg.coef_*x_train + linear_reg.intercept_
    file= open('model.pkl','wb')
    pickle.dump(linear_reg,file)
    file.close()
    
    Hour = np.array([[9.25]])
    Result = linear_reg.predict(Hour)