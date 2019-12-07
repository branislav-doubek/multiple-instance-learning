# Multiple instance learning framework

Multiple instance learning framework which uses cardinality potential for infering bag labels.

## Getting Started
Download datasets for multiple instance learning at:
* matlab files: [Fox, Elephant, Tiger](http://www.cs.columbia.edu/~andrews/mil/datasets.html), 
* clean data files: [Musk1](https://archive.ics.uci.edu/ml/datasets/Musk+(Version+1)) and [Musk2](https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2))

## Prerequisites
Install required packages:
```
quadprog - pip install quadprog
scipy - pip install scipy
numpy - pip install numpy
scikit-learn - pip install scikit-learn
pandas - pip install pandas
```
## Installation
Download files using:
```
git clone https://github.com/Frovis/MultipleInstanceLearning.git
```
## Runinng sript
For starting the application run: 
```
python3 main.py <mode> <kernel> <dataset> <cardinality potential> <parameter=value>
```
### Keywords:
mode:
```
train - trains classifier on train dataset and saves it into _models file
test - tests classifier on test dataset
run - trains classifier and evaluates accuracy and f1 score on testing dataset
cv - cross-validates classifier on train dataset and evaluates accuracy and f1 score on testing dataset
```

kernel:
```
bgd = batch gradient descent
inf-svm = inference based SVM
lp = linear programming
```
dataset:
```
fox
tiger
elephant
musk1
musk2
camelyon
synthetic
image
```
cardinality potential:
```
mimn = multiple instance markov network
rmimn =  ratio constrained multiple instance markov network
gmimn =  generalized multiple instance markov network
```
and with parameters:
```
-ro = used in rmimn potential as a constraint [float]
-c = sets parameter C [float]
-iterations = sets maximum number of permitted iterations [int]
-k = used in gmimn potential as a constraint [int]
-rs = random seed [int]
-v = visualize [bool]
-cv = cross-validate on hyperparameter ['rho'(ONLY RMIMN), 'c', 'lr', 'k'(only GMIMN)]
-lr = learning rate [float]
-cardinality_lr = cardinality learning rate [float]
-cm = confusion matrx [bool]
-s = split ratio for training/testing[float]
-norm = regularization norm ['l1', 'l2'] (Only for bgd)
-lpm = linear programming methods ['interior-point', 'revised simplex', 'simplex'] (Only for lp)
```
## Example
If we want to cross validate rho parameter using batch gradient descent with rmimn cardinality potential on fox dataset:
```
python3 main.py cv bgd fox rmimn -cv='rho'
```


## Authors

* **Branislav Doubek** -  - [Github](https://github.com/branislav-doubek)
