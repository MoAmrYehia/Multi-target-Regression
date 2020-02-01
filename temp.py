import numpy as np
import pandas as pd



#Import Training Data
DataSet = pd.read_csv("train.csv",error_bad_lines=False)
#Import Testing Data
DataTestSet = pd.read_csv('test.csv')



SimValu = DataSet.iloc[:,[2,9,10,11,12,13,14]].values #Simulation Values 

RealValuP = DataSet.iloc[:,3:6].values #Real Position Values
RealValuV = DataSet.iloc[:,6:9].values #Real Velocity Values

X_test =DataTestSet.iloc[:,[1,3,4,5,6,7,8]].values #Test Values

#Import LinearRegression
from sklearn.linear_model import LinearRegression

#Apply LinearRegression on Position values
regressorP = LinearRegression()
regressorP.fit(SimValu,RealValuP)
Y_predP = regressorP.predict(X_test)

#Apply LinearRegression on Velocity values
regressorV = LinearRegression()
regressorV.fit(SimValu,RealValuV)
Y_predV = regressorV.predict(X_test)


#Create submission DataFrame and export it as a CSV file 
submission = pd.DataFrame({'id': DataTestSet.iloc[:, 0], 'x': Y_predP[:, 0],'y':Y_predP[:,1],'z':Y_predP[:,2],'Vx':Y_predV[:,0],'Vy':Y_predV[:,1],'Vz':Y_predV[:,2]})
submission.to_csv('submission.csv', index=False)

