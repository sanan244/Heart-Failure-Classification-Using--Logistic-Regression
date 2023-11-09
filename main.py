import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
pd.set_option("display.max_rows", 25, "display.max_columns", 100)

# Load Data
#data = np.array()
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
#print(df)

# containes all features
X = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11]]
X = X.to_numpy()
#print(X)
# contains death events
Y = df['DEATH_EVENT']
Y = Y.to_numpy()
#print(Y)

# Create Logistic Regression Model
##Split data
train_x, test_x, train_y, test_y = train_test_split(
 X, Y, test_size=.3, random_state=2)

##define model
logit = LogisticRegression(solver = 'lbfgs')

# Train Model
## fit model
logit.fit(train_x, train_y)

# Display Results
predictions = logit.predict(test_x)
print("Logistic Regression with Scikit-learn for Heart Disease failure prediction")
print("...Test predictions:\n",predictions)
print("...Correct Test Labels:\n",test_y)
score = logit.score(test_x, test_y)
print("...accuracy:",score)


headers=['age','anaemia','creatinine_phosp','diabetes','ejection_fra..','high_blood_pr..','platelets','serum_creati.','serum_sod','sex(1-M,0-F)','smoking','time']
W = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]]

W=W.to_numpy()
avg1=[]
avg2=[]
val1=[]
val2=[]
#separating data in death positive and death negative
for i in range(0,299):
    if W[i][12]==0:
        val1.append(list(W[i]))
    else:
        val2.append(list(W[i]))
            
val1=np.array(val1)
val2=np.array(val2)
for i in range(0,12):
    a=np.mean(val1[:,i])
    avg1.append(a)
for i in range(0,12):
    a=np.mean(val2[:,i])
    avg2.append(a)    
bar_width = 0.35
r2 = np.arange(len(avg1))+bar_width


#displaying data as bar plot
plt.bar(headers, avg1, color='r', width=bar_width, edgecolor='grey', label='Death positive',log=True)
plt.bar(r2, avg2, color='b', width=bar_width, edgecolor='grey', label='Death negative',log=True)
plt.xlabel('Average_attribute_value', fontweight='bold')
plt.ylabel('Values', fontweight='bold')
plt.title('Mean values', fontweight='bold')
plt.legend()

plt.show()  