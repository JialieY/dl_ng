# Structuring Machine Learning Projects

## Evaluation Matrix
* Optimize
    * Precision
    * Recall
    * F1 Score
    * AUC
    * ROC
* Satisfying

## Error Analysis

<pre>


Human-level
   |
   |                  s1: bigger model 
   | avoidable bias   s2: better optimization algorithm
   |                  s3: different NN stucture, hyperparameter tune
   |
Train error
   |
   |                  s1: more data 
   | variance         s2: regularization
   |                  s3: different NN stucture, CNN, RNN
   |
Train-Val error
   |
   |                 
   | data-mismatch    s1: add artifiical data synthesis to training set
   |                  s2: add more similiar data to train-val set
   |
 Val error
   |
   |                   
   | overfit          s1: regularization
   |                  s2: more val data
   |
 Test error
  
