
### To Run Linear Discriminate Analysis for Kaggle

Run LCA.py. This file will generate the variable called:pred_kaggle,which is the prediction.


### To Run Naive Bayes prediction for Kaggle

First Run TrainingNB.py. This file will generate the conditional probabilities of all the attributes and will save them into storekaggle.pckl.

Then Run TestingNB.py. This function will generate an excel sheet with predictions

Depending on computer, training takes about 1 minute. Testing about 30 seconds.

Note:
storeinfo.pckl stores the mutual information data generated using training data. Mutual information code is not submitted but explained in the report since it was a part of preprocessing


### To Run AdaBoost on Logistic Regression

Run Adaboost_LR_sklearn.py. This will generate the prediciton of testing data by doing adaboost on sklearn logistic regression. 
Note: change k from 1 to 3 to get the Adaboost results. It will not continueing if k>=4. 

Run Adaboost_LR_failed.py. This one without using any packages cannot go through Adaboost since the LR accuracy is lower than 0.5. 