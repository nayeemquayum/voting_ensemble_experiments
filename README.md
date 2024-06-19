#We'll apply voting ensemble technique on iris dataset.
From previous experience we know that the species for iris flowers are not linearly separated based on Sepal length and weidth.
To experiment with complex data, we therefore select sepal length and width as input features.

We'll run voting ensemble technique using SVC, decision tree, knn and Naive bayes classier models as base models. First, we'll
run each models individually and using GridSearchCV we will identify the optimal hyperparameter values. After identifying
the optimal hyperparameters will run the voting ensemble technique with them.
For SVC:
Using the GridSearchCV we identified the optimal value for hyperparameter kernel, gamma and C. The result is

GridSearchCV best parameters: {'C': 3, 'gamma': 'auto', 'kernel': 'rbf'}
GridSearchCV best parameters score: 0.8333333333333334

For Decision DecisionTreeClassifier:
Using the GridSearchCV we identified the optimal value for hyperparameter criterion and max_depth. The result is

GridSearchCV best parameters for Decision Tree classifier: {'criterion': 'gini', 'max_depth': 4}
GridSearchCV best parameters score for Decision Tree classifier: 0.8

For KNN classifier:
Using the GridSearchCV we identified the optimal value for hyperparameter n_neighbors. The result is

GridSearchCV best parameters for KNN classifier: {'n_neighbors': 17}
GridSearchCV best parameters score for KNN classifier: 0.82

For the final voting classifier we used
estimators = [('svc',optimal_svc_model),('DT',optimal_DT_model),('knn',optimal_Knn_model),('GNB',gaussianNB_model)]
voting_classifier = VotingClassifier(estimators=estimators,voting='hard')

The result:

Accuracy: 0.7666666666666667
Confusion matrix    0  1  2
0  8  0  0
1  0  7  4
2  0  3  8
Precision:   0.7677777777777777
Recall:  0.7666666666666667
F1 score:  0.7661835748792271
--------------------------------------------------------------------------------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         8
           1       0.70      0.64      0.67        11
           2       0.67      0.73      0.70        11

    accuracy                           0.77        30
   macro avg       0.79      0.79      0.79        30
weighted avg       0.77      0.77      0.77        30
