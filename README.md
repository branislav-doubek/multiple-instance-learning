# Multiple instance learning framework

Multiple instance learning framework which uses cardinality potential for infering bag labels.

## Installation
Download files using:
```
git clone https://github.com/Frovis/MultipleInstanceLearning.git
```
Download datasets for multiple instance learning:
* matlab files: [Fox, Elephant, Tiger, Musk1, Musk2](http://www.cs.columbia.edu/~andrews/mil/datasets.html)
and extract them into data folder of our project.
## Libraries
to install libraries run:
```
pip3 install -r requirements.txt 
```
The only problem might be installing library lp solvers(https://pypi.org/project/lpsolvers/), which requires pycddlib library (https://pypi.org/project/pycddlib/).
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
validate - runs 1 run of k-fold cross validation on training dataset
```

kernel:
```
bgd = batch gradient descent
svm = inference based SVM
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
-valid_iter = index of fold used as validation sample in validate split [int]
-mom = momentum parameter [float]
```
### Cross validation
If we run cross-validation mode, k-fold cross validation is called on training dataset and we measure accuracy of our classifier for different hyperparameter values. We provide list of tunable hyperparameters:
```
-lr
-mom
-c
-ro
-k
```
If we want to tune all hyperparameters of classifier we can just run cross validation mode without more parameters, but for cross-validating classifier on single parameter(i.e. C paramter) we can just add -cv='c' for validation only on C parameter.
## Example
If we want to cross validate rho parameter using batch gradient descent with rmimn cardinality potential on fox dataset:
```
python3 main.py cv bgd fox rmimn -cv='ro'
```
For visualization of loss function of batch gradient descent optimization algorithm with gmimn potential and kappa set to 10 run on musk1 dataset:
```
python3 main.py train bgd musk1 gmimn -v=True -k=10
```


## Authors

* **Branislav Doubek** -  - [Github](https://github.com/branislav-doubek)
