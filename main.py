import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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