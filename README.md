Package information: 
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

<img src="icon.jpg" align="right" />

# Mozinor [![Mozinor](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
> pip install is coming

For now you must clone this repo and add him to your PYTHON_PATH

### Installation

Mozinor requires [Python 3.6](https://www.python.org/downloads/release/python-360/).

Install dependencies thanks to setup.py
```
$ python setup.py
```

### Plugins

| Plugin | URL |
| ------ | ------ |
| sklearn | [https://github.com/scikit-learn/scikit-learn] |
| pydotplus | [https://pypi.python.org/pypi/pydotplus] |
| tqdm | [https://pypi.python.org/pypi/tqdm] |
| vecstack | [https://github.com/vecxoz/vecstack] |

### Notebook

> regression:
  https://github.com/Jwuthri/Mozinor/blob/master/mozinor/example/Mozinor%20example%20Reg.ipynb
> classification:
  https://github.com/Jwuthri/Mozinor/blob/master/mozinor/example/Mozinor%20example%20Class.ipynb

### Run

```python
from mozinor.baboulinet import Baboulinet

cls = Baboulinet(filepath="toto.csv", y_col="predict", regression=False)
res = cls.babouline()
```
Show best model, and best stack model:
```python
res.best_model
res.best_stack_models
```
You got:
```python
  Estimator    (ExtraTreeClassifier(class_weight=None, criter...
  Score                                                 0.864667
  Degree                                                       1
  Name: 0, dtype: object
```
Build the generate code for the best model, and best stack model:

```python
cls.bestModelScript()
cls.bestStackModelScript()
```
Example of generated code for best model:

```python
# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble.forest import ExtraTreesClassifier

from vecstack import stacking

# Read the csv file
data = pd.read_csv("toto.csv")
regression = False

# Split dependants and independant variables
y = data[["predict"]]
X = data.drop("predict", axis=1)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Apply Some Featuring
poly_reg = PolynomialFeatures(degree=1)

# Transform into numpy object
x_train = poly_reg.fit_transform(X_train)
X_test = poly_reg.fit_transform(X_test)
y_test = np.array(y_test.ix[:,0])
y_train = np.array(y_train.ix[:,0])

# Build model with good params
model = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='entropy',
           max_depth=None, max_features=0.6, max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=4, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)

# Fit the model
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Scoring
if regression:
    print('Score on test set:', mean_absolute_error(y_test, y_pred))
else:
    print('Score on test set:', accuracy_score(y_test, y_pred))
```

### Todos

 - Write Tests
 - Make pip install
 - ...

License
----

MIT


**Free Software !**
