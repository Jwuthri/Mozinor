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
Build the generate code for the best model, and best stack model:

```python
cls.bestModelScript()
cls.bestStackModelScript()
``

### Todos

 - Write Tests
 - Make pip install
 - ...

License
----

MIT


**Free Software !**
