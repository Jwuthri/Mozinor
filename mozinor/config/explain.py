# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
PolynomialFeatures = """
    Polynomial Features: generate a new feature matrix
    consisting of all polynomial combinations of the features.
    For 2 features [a, b]:
        the degree 1 polynomial give [a, b]
        the degree 2 polynomial give [1, a, b, a^2, ab, b^2]
    ...
"""
OobError = """
    OOB: this is the average error for each training observations,
    calculted using the trees that doesn't contains this observation
    during the creation of the tree.
"""
Elbow = """
    ELBOW: explain the variance as a function of clusters.
"""
Stacking = """
    Stacking: is a model ensembling technique used to combine information
    from multiple predictive models to generate a new model.
"""
DecisionTreeClassifier = """
    Decision Tree Classifier: poses a series of carefully crafted questions
    about the attributes of the test record. Each time time it receive an answer,
    a follow-up question is asked until a conclusion about the calss label
    of the record is reached.
"""
DecisionTreeRegressor = """
    Decision Tree Regressor: poses a series of carefully crafted questions
    about the attributes of the test record with addition noisy observation.
"""
LogisticRegression = """
    LogisticRegression: explain the relationship between one dependent binary
    variable and one or more nominal
"""
GaussianNB = """
    GaussianNB: the likelihood of the features is assumed to be Gaussian.
"""
BernoulliNB = """
    BernoulliNB: for data that is distributed according to
    multivariate Bernoulli distributions.
"""
KNeighborsClassifier = """
    KNeighborsClassifier: Majority vote of its k nearest neighbors.
"""
KNeighborsRegressor = """
    KNeighborsRegressor: Average of its k nearest neighbors.
"""
ElasticNetCV = """
    ElasticNetCV: linear regression with combined
    L1 (lasso penalty) and L2(ridge penalty) priors as regularizer.
"""
RidgeCV = """
    RidgeCV: performs on L2 regularization, it adds a factor of sum of squares
    of coefficients in the optimization objective.
    Usefull with higly correlated features.
"""
LassoLarsCV = """
    LassoLarsCV: performs L1 regularization, it adds a factor of sum of
    absolute value of coefficients in the optimization objective.
    Usefull with lot of features, made some feature selection.
"""
GradientBoostingRegressor = """
    GradientBoostingRegressor: as in random forests, a random subset of
    candidate features is used, but the trees are builds one-by-one,
    then the predictions of the individual trees are summed.
"""
GradientBoostingClassifier = """
    GradientBoostingClassifier: as in random forests, a random subset of
    candidate features is used, but the trees are builds one-by-one,
    then the predictions of the individual trees are summed.
"""
RandomForestClassifier = """
    RandomForestClassifier: builds an ensemble of 'weeks learner' named decision
    tree, but all taken together are a 'strong learner'.
    The 'weeks learner' are build thanks to a split from the training set,
    each split is the best split among a random subset of the features
"""
RandomForestRegressor = """
    RandomForestRegressor: builds an ensemble of 'weeks learner' named decision
    tree, but all taken together are a 'strong learner'.
    The 'weeks learner' are build thanks to a split from the training set,
    each split is the best split among a random subset of the features
"""
ExtraTreesClassifier = """
    ExtraTreesClassifier: as in random forests, a random subset of candidate
    features is used, but instead of looking for the most discriminative
    thresholds, thresholds are drawn at random for each candidate feature and
    the best of these randomly-generated thresholds is picked as
    the splitting rule.
"""
ExtraTreesRegressor = """
    ExtraTreesRegressor: as in random forests, a random subset of candidate
    features is used, but instead of looking for the most discriminative
    thresholds, thresholds are drawn at random for each candidate feature and
    the best of these randomly-generated thresholds is picked as
    the splitting rule.
"""
AdaBoostRegressor = """
    AdaBoostRegressor: as in random forests, adaboost fit a sequence of
    weak learners from training set. The predictions from all of them are then
    combined through a weighted sum to produce the final prediction.
"""
XGBClassifier = """
    Gradient boosting is an approach where new models are created that predict
    the residuals or errors of prior models and then added together to make
    the final prediction. It is called gradient boosting because it uses a
    gradient descent algorithm to minimize the loss when adding new models.
"""
XGBRegressor = """
    Gradient boosting is an approach where new models are created that predict
    the residuals or errors of prior models and then added together to make
    the final prediction. It is called gradient boosting because it uses a
    gradient descent algorithm to minimize the loss when adding new models.
"""
