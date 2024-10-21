# Naive_Bayes_Ads_Classifier
This project applies Naive Bayes classification models (BernoulliNB, MultinomialNB, and GaussianNB) on the Social Network Ads dataset to predict whether a user will purchase a product based on age and estimated salary. The model performance is evaluated using confusion matrix, accuracy score, bias, variance, and classification report.
___________________________________________________________________________________________________________________________________________________________________________________
confusion matrix:
 [[58  0]
 [22  0]]
accuracy score: 0.725
bias: 0.621875
variance: 0.725
classification Rreport:
               precision    recall  f1-score   support

           0       0.72      1.00      0.84        58
           1       0.00      0.00      0.00        22

    accuracy                           0.72        80
   macro avg       0.36      0.50      0.42        80
weighted avg       0.53      0.72      0.61        80
****************************************************************************

**Conclusion:**

The final evaluation provides insights into how well the Naive Bayes model generalizes for predicting user behavior based on social network advertisements.
