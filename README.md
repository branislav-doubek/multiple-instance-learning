# Multiple instance learning framework

Multiple instance learning framework which uses cardinality potential for infering bag labels.

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
Download datasets for multiple instance learning at:
* matlab files: [Fox, Elephant, Tiger](http://www.cs.columbia.edu/~andrews/mil/datasets.html), 
* clean data files: [Musk1](https://archive.ics.uci.edu/ml/datasets/Musk+(Version+1)) and [Musk2](https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2))

and place them into data folder of our project.
## Runinng sript
For starting the application run: 
```
python3 main.py <mode> <kernel> <dataset> <cardinality potential> <parameter=value>
```
### Keywords:
mode:
```
train - trains classifier on train dataset and saves it into _models file
test - tests trained classifier on test dataset
run - trains classifier and evaluates its accuracy testing dataset
cv - uses k-fold cross validation on training dataset to estimate hyperparemeters of our classifier and evaluates accuracy on testing dataset
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
image
noise
```
cardinality potential:
```
mimn = multiple instance markov network
rmimn =  ratio constrained multiple instance markov network
gmimn =  generalized multiple instance markov network
```
### Parameters
We provide list of parameters for our classifier:
```
-ro = used in rmimn potential as a constraint [float]
-c = sets parameter C [float]
-iterations = sets maximum number of permitted iterations [int]
-k = used in gmimn potential as a constraint [int]
-rs = random seed [int]
-v = visualize [bool]
-cv = cross-validate on hyperparameter (Described in section below)
-lr = learning rate [float]
-cm = visualize confusion matrx [bool]
-s = split ratio for training/testing[float]
-norm = regularization norm ['l1', 'l2'] (Only for bgd)
-lpm = linear programming methods ['interior-point', 'revised simplex', 'simplex'] (Only for lp)
```
### Cross validation
If we run cross-validation mode, k-fold cross validation is called on training dataset and we measure accuracy of our classifier for different hyperparameter values. We provide list of tunable hyperparameters:
```
-lr
-c
-ro
-k
-clr
-iterations
```
If we want to tune all hyperparameters of classifier we can just run cross validation mode without more parameters, but for cross-validating classifier on single parameter(i.e. C paramter) we can just add -cv='c' for validation only on C parameter.
## Example
If we want to cross validate rho parameter using batch gradient descent with rmimn cardinality potential on fox dataset:
```
python3 main.py cv bgd fox rmimn -cv='rho'
```
For visualization of loss function of mi-SVM algorithm with gmimn potential and kappa set to 10 run on musk1 dataset:
```
python3 main.py train svm musk1 gmimn -v=True -k=10
```


## Authors

* **Branislav Doubek** -  - [Github](https://github.com/branislav-doubek)
