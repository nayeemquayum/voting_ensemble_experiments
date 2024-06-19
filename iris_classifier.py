import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#import the models
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

#load data in bunch format
iris=load_iris()
#check the features of sklearn data set
print("iris features:",iris.feature_names)
#create dataframe from bunch
iris_df=pd.DataFrame(iris.data,columns=iris.feature_names)
iris_df['species'] = iris.target
print("iris features:",iris_df.info())
iris_df.rename(columns={'sepal length (cm)' : 'sepal_length_cm',
                     'sepal width (cm)' : 'sepal_width_cm',
                     'petal length (cm)': 'petal_length_cm',
                     'petal width (cm)' : 'petal_width_cm'}, inplace=True)
iris_df.drop(['petal_length_cm', 'petal_width_cm'], axis='columns', inplace=True)
print("iris data with 2 features:",iris_df.info())
import seaborn as sns
# sns.pairplot(iris_df,hue='species')
# plt.show()
#SVC
#Using GridSearchCV identified the optimal value for hyperparameter kernel, gamma and C
X = iris_df.iloc[:,0:2]
y = iris_df.iloc[:,2]
svc_model=SVC()

svc_parameters = {'kernel':('linear', 'rbf','poly','sigmoid'),
                  'C':[0.1,.05,1,3,5,7,9,10],
                  'gamma' : ('scale','auto')
                  }
svc_grid=GridSearchCV(svc_model,svc_parameters,verbose=4,refit=True)
svc_grid.fit(X,y)
print("GridSearchCV best parameters for SVC:",svc_grid.best_params_)
print("GridSearchCV best parameters score for SVC:",svc_grid.best_score_)
#DecisionTree
from sklearn.tree import DecisionTreeClassifier
decision_tree_clf = DecisionTreeClassifier()
decision_tree_parameters = {'criterion':('gini', 'entropy'),
                  'max_depth':[2,3,4,5,6,7,8,9,10]
                  }
svc_grid_DT=GridSearchCV(decision_tree_clf,decision_tree_parameters,verbose=4,refit=True)
svc_grid_DT.fit(X,y)
print("GridSearchCV best parameters for Decision Tree classifier:",svc_grid_DT.best_params_)
print("GridSearchCV best parameters score for Decision Tree classifier:",svc_grid_DT.best_score_)

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
knn_parameters = {'n_neighbors':[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                  }
svc_grid_knn=GridSearchCV(knn_classifier,knn_parameters,verbose=4,refit=True)
svc_grid_knn.fit(X,y)
print("GridSearchCV best parameters for KNN classifier:",svc_grid_knn.best_params_)
print("GridSearchCV best parameters score for KNN classifier:",svc_grid_knn.best_score_)

#Now let's create the base models with optimal hyperparameters
optimal_svc_model=SVC(kernel= 'rbf',C= 3,gamma= 'auto')
optimal_DT_model=DecisionTreeClassifier(criterion='gini',max_depth=4)
optimal_Knn_model=KNeighborsClassifier(n_neighbors=17)
gaussianNB_model=GaussianNB()
#Voting ensemble classifier
from sklearn.ensemble import VotingClassifier
estimators = [('svc',optimal_svc_model),('DT',optimal_DT_model),('knn',optimal_Knn_model),('GNB',gaussianNB_model)]
voting_classifier = VotingClassifier(estimators=estimators,voting='hard')
#split data in train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris_df.drop(columns=['species']),
                                               iris_df['species'],test_size=0.2,random_state=20)
voting_classifier.fit(X_train,y_train)
y_prediction= voting_classifier.predict(X_test)
#print classification report
from sklearn.metrics import accuracy_score, recall_score,precision_score,f1_score, confusion_matrix
# Model Accuracy
print("Accuracy:",accuracy_score(y_test, y_prediction))
#confusion_matrix
confusion_matrix=pd.DataFrame(confusion_matrix(y_test,y_prediction),columns=list(range(0,3)))
print("Confusion matrix",confusion_matrix.head())
print("Precision:  ",precision_score(y_test,y_prediction,average='weighted'))
print("Recall: ",recall_score(y_test,y_prediction,average='weighted'))
print("F1 score: ",f1_score(y_test,y_prediction,average='weighted'))
print("-"*80)
#another way is using classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_prediction))