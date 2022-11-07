#!/usr/bin/env python
# coding: utf-8

# In[231]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC,SVC,SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, LogisticRegression, Perceptron
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[196]:


train = pd.read_csv("/Users/apple/Downloads/train.csv")
test = pd.read_csv("/Users/apple/Downloads/train.csv")


# In[197]:


train.head()
train.info()
train.describe()
train


# In[198]:


#Looking at correlations
corr_matrix = train.corr()
corr_matrix["Survived"].sort_values(ascending=False)


# In[199]:


#Group by different attributes by survived - correlation
train[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[200]:


train[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[201]:


train[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[202]:


#Map categorical value to numerical
train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[203]:


#Data Visualisation
#Age distribution
train.hist(["Age"]) #Most passengers aged between 20-35


# In[204]:


#Age distribution and Survived
g = sns.FacetGrid(train,col="Survived")
g.map(plt.hist,"Age",bins=20) #Different age band has different survided rate - may divided age into groups


# In[205]:


#Age, PClass and Survived
g1 = sns.FacetGrid(train,col="Survived",row="Pclass")
g1.map(plt.hist,"Age",bins=20) #Pclass3: most didn't survive - use Pclass as input


# In[206]:


#Fare and Survived
g3 = sns.FacetGrid(train,col="Survived",row="Embarked")
g3.map(sns.barplot,"Fare") #Add fare to input


# In[207]:


train = train.drop(["Ticket","Cabin"],axis=1)
test = test.drop(["Ticket","Cabin"],axis=1)


# In[208]:


#Title extraction
for dataset in (train,test):
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])

#replace to normal title
for dataset in (train,test):
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#Covert category to ordinal variable
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in (train,test):
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


# In[209]:


train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)


# In[210]:


#Covert sex to 0,1
test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[211]:


#Missing value of age
#More accurate way of guessing missing values is to use other correlated features. 
#In our case we note correlation among Age, Gender, and Pclass. 
#Guess Age values using median values for Age across sets of Pclass and Gender feature combinations. 
#So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on
guess_age = np.zeros((2,3))
guess_age

for dataset in (train,test):
    for i in range(0,2):
        for j in range(0,3):
            guess = dataset[dataset["Sex"] == i & (dataset["Pclass"] == j+1)]["Age"].dropna().median()
            guess_age[i,j] = guess
            
for dataset in (train,test):
    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[
                (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_age[i,j]
    dataset['Age'] = dataset['Age'].astype(int)


# In[212]:


#Divide age into different bands
train["AgeBand"] = pd.cut(train["Age"],bins = [-1,8,16,24,32,np.inf],labels=[0,1,2,3,4]).astype(float)
test["AgeBand"] = pd.cut(test["Age"],bins = [-1,8,16,24,32,np.inf],labels=[0,1,2,3,4]).astype(float)


# In[213]:


#AgeBand and Survived grouped by AgeBand
train[["AgeBand","Survived"]].groupby(["AgeBand"]).mean().sort_values(by="AgeBand",ascending=False)

for dataset in (train,test):
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='FamilySize', ascending=False)
test.info()

corr_matrix = train.corr()
corr_matrix["Survived"].sort_values(ascending=False)

train = train.drop(["Age"],axis=1)
test = test.drop(["Age"],axis=1)


# In[214]:


#Deal with missing value of Embarked
train["Embarked"].value_counts()
port = train["Embarked"].dropna().mode()[0]
train["Embarked"]=train["Embarked"].fillna(port)
test["Embarked"]=test["Embarked"].fillna(port)

train[["Embarked","Survived"]].groupby(["Embarked"]).mean().sort_values(by="Survived",ascending=False)

for dataset in (train,test):
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[215]:


train["FareBand"] = pd.cut(train["Fare"],bins=[-5,8,14,31,np.inf],labels=[0,1,2,3]).astype(np.int)
test["FareBand"] = pd.cut(test["Fare"],bins=[-5,8,14,31,np.inf],labels=[0,1,2,3]).astype(np.int)


# In[216]:


train[["FareBand","Survived"]].groupby(["FareBand"]).mean().sort_values(by="FareBand",ascending=False)

x_train = train.drop(["Survived","Fare","Parch"],axis=1)
y_train = train["Survived"].copy()

x_test = test.drop(["Survived","Fare","Parch"],axis=1)
y_test = test["Survived"].copy()


# In[217]:


#Build model
#1. SVM
#2. Logistic Regression
#3. Decision Tree
#4. Random Forest
#5. Naive Bayes
#6. KNN

def accuracy_model(model,x,y,cv=5):
    acc = cross_val_score(model,x,y,cv=cv,scoring="accuracy")
    return acc.mean()

######################
#########SVM##########
######################
#1.1 Linear SVM
lin_svc = SVC(probability=True)
lin_svc.fit(x_train,y_train)
acc_svc = round(lin_svc.score(x_train, y_train) * 100, 2)
acc_svc


# In[218]:


#1.2 Poly kernel SVM
poly_svm = SVC(kernel="poly",degree=2,coef0=1,C=5,probability=True)
poly_svm.fit(x_train,y_train )

#######################################
#########Logistic Regression##########
######################################
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)

####################
#######KNN##########
####################
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)

###########################
#######Perceptron##########
###########################

perceptron = Perceptron()
perceptron.fit(x_train,y_train)

###########################
#########Random Forest#####
###########################
rf = RandomForestClassifier(n_estimators=200)
rf.fit(x_train,y_train)

###########################
#######Decision Tree#######
###########################
tree = DecisionTreeClassifier()
tree.fit(x_train,y_train)

models = [lin_svc,poly_svm,log_reg,knn,perceptron,rf,tree]
names = ["LinearSVC","PolySVM","LogisticRegression","KNN","Perceptron","RandomForest","DecisionTree"]
scores = [0,0,0,0,0,0,0]

for i in range(len(models)):
    model = models[i]
    name = names[i]
    scores[i] = accuracy_model(model,x_train,y_train,cv=5)
    
#Model Evaluation

models_eva = pd.DataFrame({
    'Model': names,
    'Score': scores})
                               
models_eva.sort_values(by='Score', ascending=False)


# In[219]:


#Hyperparameter tune


# In[220]:


for name,score in zip(list(x_train),rf.feature_importances_):
    print(name,score)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[236]:


#Intestigate Ensemble Learning

#Bagging
bag_tree = BaggingClassifier(DecisionTreeClassifier(),n_estimators = 500,bootstrap=True,n_jobs=-1,max_samples=100)
bag_svm = BaggingClassifier(SVC(),n_estimators = 500,bootstrap=True,n_jobs=-1,max_samples=600)
bag_log = BaggingClassifier(LogisticRegression(),n_estimators = 100,bootstrap=True,n_jobs=-1,max_samples=600)

#Vote
vote_clf = VotingClassifier([("lr",log_reg),("svm",poly_svm),("linsvc",lin_svc),("rf",rf)],voting="soft")

#AdaBoost
ada_clf = AdaBoostClassifier(
                              DecisionTreeClassifier(max_depth=1),n_estimators=200,
                              algorithm = "SAMME.R",learning_rate = 0.5
)
ada_clf.fit(x_train,y_train)

#GradientBoost
gbr = GradientBoostingClassifier(max_depth=2,n_estimators=50,learning_rate=1)
gbr.fit(x_train,y_train)


# In[237]:


models = [bag_log,bag_tree,bag_svm,vote_clf,ada_clf,gbr]
names = ["LRBagging","DecisionTreeBagging","SVMBagging","Voting Ensemble","AdaBoost","GradientBoost"]
scores = [0,0,0,0,0,0]

for i in range(len(models)):
    model = models[i]
    name = names[i]
    scores[i] = accuracy_model(model,x_train,y_train,cv=5)
    
#Model Evaluation

models_eva = pd.DataFrame({
    'Model': names,
    'Score': scores})
                               
models_eva.sort_values(by='Score', ascending=False)


# In[ ]:




