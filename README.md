## Kaggle Quora Questions Pairs Competition


__14th place solution. My part.__
Code is uncleaned, latest versions are uploaded.
Not every feature, that can be created with features notebooks was contained in final model - idea of this repository is to give more of an overview of methods used and those that could be used for similar problems.

Big thanks to the authors of all kernels & posts, which were of great inspiration and some features were derived based on them.


### Features

* Data Encoding:
  * Pipeline for text cleaning using Textacy
  * Lemmatization
  * Stemming
  * NER Encoding (based on Kernel)
* NLP Features:
  * Features based on Kaggle Kernels & Discussions posts by: Abhishek, SRK, Jared Turkewitz, the_1owl, Mephistopheles & more
  * Latent Semantic Analysis, Latent Dirichlet Allocation, tSVD
  * Word2Vec
  * Doc2Vec
  * Distances based on data transformations - similarity measures
  * Textacy-based features
  * KNN-based features 
* Magic Features:
  * Jared Turkewitz's frequency features
  * NetworkX features
  
__& some more.__


### Models:

* XGB & LGBM models
  * Training
  * BayesianOptimization
  * Test Predictions
* [SpaCy Decomposable Attention Model on Quora data](https://github.com/explosion/spaCy/tree/master/examples/keras_parikh_entailment)
* LSTM Experiments
* MLP models
* Stacking
  * Sklearn Models Ensemble
  * Stacking with LGBM
  * Finding weights for ensemble using Scipy minimize function in-fold
 
