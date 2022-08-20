# Import the required libraries
import pandas as pd
data_test = pd.read_csv(r'C:\Users\Sai\Downloads\test_data_evaluation_part2.csv')
data_train = pd.read_csv(r'C:\Users\Sai\Downloads\train_data_evaluation_part_2.csv')

"""Project Mangement Methodology: CRISP ML (Q) 

---

Step 1: Business Understanding & Data Understanding
"""
a = pd.DataFrame(data_train['Nationality'].unique())
b = pd.DataFrame(data_train['Nationality'].unique())
data_train.info()

data_train.shape # Dimesion of the train dataset

data_train.head() # First '5' observations

data_train.iloc[:,2:].describe() # Returns Max,Min,Quantiles - Q1,Q2,Q3 values

"""Step 2: Data Preprocessing/EDA/Feature Engineering 

"""

# Dummy Variable Creation for categorical features
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data_train['Nationality']= label_encoder.fit_transform(data_train['Nationality'])
data_train['DistributionChannel']= label_encoder.fit_transform(data_train['DistributionChannel'])
data_train['MarketSegment']= label_encoder.fit_transform(data_train['MarketSegment'])

# Dropping the constant input features
from sklearn.feature_selection import VarianceThreshold
var_thresh = VarianceThreshold(threshold = 0)
var_thresh.fit(data_train.iloc[:,2:])

# Non zero variance features
data_train.iloc[:,2:].columns[var_thresh.get_support()]

# Zero variance features
constant_columns = [column for column in data_train.iloc[:,2:].columns if column not in data_train.iloc[:,2:].columns[var_thresh.get_support()]]
constant_columns # No constant features are found

# Handling Missing values
data_train.isna().sum() # NaN values are found in 'Age' column
data_train.dropna(inplace=True)
# Replacing missing values in 'Age' column
# with median of that column
# data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())

# Checking for negative values & replacing with '0'
(data_train < 0).sum()

import numpy as np
data_train['Age'] = np.where(data_train['Age'] < 0, 0, data_train['Age'])
data_train['AverageLeadTime'] = np.where(data_train['AverageLeadTime'] < 0, 0, data_train['AverageLeadTime'])
data_train['DaysSinceLastStay'] = np.where(data_train['DaysSinceLastStay'] < 0, 0, data_train['DaysSinceLastStay'])
data_train['DaysSinceFirstStay'] = np.where(data_train['DaysSinceFirstStay'] < 0, 0, data_train['DaysSinceFirstStay'])

# Scaling the input features:
def norm_func(i):
    x = (i - i.min())/(i.max() - i.min())
    return (x)

data_train = norm_func(data_train.iloc[:,2:])
data_train.columns

# Discretization/Binning/Grouping
data_train["is_checkedin"]='Yes'
data_train.loc[data_train["BookingsCheckedIn"]==0,"is_checkedin"]='No'
data_train = data_train.drop(['BookingsCheckedIn'],axis=1)

# Feature Selection: Using "K-Best & Chi2" Algorithm 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = data_train.iloc[:,2:-1];Y = data_train.iloc[:,-1]
best_ftr = SelectKBest(score_func=chi2,k='all')
allftr_scores = best_ftr.fit(X,Y)
allftr_scores = pd.DataFrame(allftr_scores.scores_,columns=['Score'])
allftr_names = pd.DataFrame(X.columns,columns=['Features'])
bst_ftrs = pd.concat([allftr_names,allftr_scores],axis=1).sort_values(by='Score',ascending = False)
bst_ftrs

# Dropping the least significant input features
data_train = data_train.drop(['SRKingSizeBed','SRQuietRoom','BookingsNoShowed','BookingsCanceled','SRNoAlcoholInMiniBar','SRLowFloor','SRAwayFromElevator','SRAccessibleRoom','SRBathtub','SRNearElevator','DistributionChannel','MarketSegment','SRMediumFloor','SRShower'],axis=1)

# Checking for Multi Collinearity between input features
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (24, 12))
corr = data_train.corr()
colormap = sns.color_palette("Blues")
sns.heatmap(corr, cmap=colormap, annot = True, linewidths = 0.5)
plt.show()

"""From the above figure, there is a high correlation observed between input features

---
1.RoomNights & PersonsNights

---
2.DaysSinceFirstStay & DaysSinceLastStay

---
3.DaysSinceLastStay & DaysSinceCreation

---


So, among those pair retaining those feature which is highly correlated with output feature



"""

output = data_train['is_checkedin'].map({'Yes':1,"No":0})
print('Correlation between RoomNights & is_checkedin', data_train['RoomNights'].corr(output))
print('Correlation between PersonsNights & is_checkedin', data_train['PersonsNights'].corr(output))
print('Correlation between DaysSinceLastStay & is_checkedin', data_train['DaysSinceLastStay'].corr(output))
print('Correlation between DaysSinceFirstStay & is_checkedin', data_train['DaysSinceFirstStay'].corr(output))
print('Correlation between DaysSinceCreation & is_checkedin', data_train['DaysSinceCreation'].corr(output))

# Dropping the least significant input features
data_train = data_train.drop(['PersonsNights','DaysSinceCreation','DaysSinceLastStay'],axis=1)

"""Step 4: Data Mining

---
Train data set : Divided into train data(80%) & validation data(20%)

---
After finializing the model(after evaluating on validation data), entire train data is trained & prediciting the output on test data


"""

data_train.columns

data_test.columns

"""Test dataset preparation"""

# Handling Missing values
data_test.isna().sum() # NaN values are found in 'Age' column

data_test['Age'] = data_test['Age'].fillna(data_test['Age'].median())

data_test1 = data_test[['Nationality', 'Age', 'AverageLeadTime', 'LodgingRevenue', 'OtherRevenue', 'RoomNights', 'DaysSinceFirstStay', 'SRHighFloor', 'SRCrib', 'SRTwinBed']]

data_test1['Nationality']= label_encoder.fit_transform(data_test1['Nationality'])

(data_test1 < 0).sum()

data_test1['DaysSinceFirstStay'] = np.where(data_test1['DaysSinceFirstStay'] < 0, 0, data_test1['DaysSinceFirstStay'])

data_test1 = norm_func(data_test1.iloc[:,:])
data_test1

data_test["is_checkedin"]='Yes'
data_test.loc[data_test["BookingsCheckedIn"]==0,"is_checkedin"]='No'
data_test = data_test.drop(['BookingsCheckedIn'],axis=1)

data_test = pd.concat([data_test1,data_test.iloc[:,-1]],axis=1)
data_test

# Data Partition into train data & test data
x_train = data_train.iloc[:,:-1]
y_train = data_train.iloc[:,-1]
x_test = data_test.iloc[:,:-1]
y_test = data_test.iloc[:,-1]

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

"""Step 4: Model Building"""

# Decision Tree Classifier:
# Hyper Parameter Optimization
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train,y_train)

# Evaluation on Test Data
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test, model.predict(x_test)))
print(classification_report(y_test, model.predict(x_test)))

# Evaluation on Train Data
print(confusion_matrix(y_train, model.predict(x_train)))
print(classification_report(y_train, model.predict(x_train)))

# Saving the model
import pickle
pickle.dump(model,open('model.pkl','wb'))

# Load the model from disk
model = pickle.load(open('model.pkl','rb'))

# Predicting the output for the test data points
predicted_values = pd.DataFrame(data_test.iloc[0:1,:-1])
output = [model.predict(predicted_values)]
output


data_train['Nationality'].value_counts()
