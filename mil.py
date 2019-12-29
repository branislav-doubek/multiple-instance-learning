import numpy as np
from bag_class import Bag
from _methods.bgd import BatchGradientDescent as BGD
from _methods.inference import Inference
from _methods.svm import Mi_SVM
from _methods.setup_logger import logger
from _methods.lp import Lp
from _methods.qp import Qp
import pickle
import traceback


class MIL(Inference, BGD, Mi_SVM, Lp, Qp):
    """
    Model for solving multiple instance learning problems
    """

    def __init__(self, args):
        """
        Initialization function
        :param args: configuration object
        :return: None
        """
        self.logger = logger
        self.cardinality = args.cardinality_potential  # cardinality type
        self.ro = args.ro  # value of ro for RMIMN potential
        self.kernel = args.kernel  # Kernel type
        self.lamb = 1/args.c  #lambda parameter value
        self.max_iterations = args.iterations  # sets max iterations
        self.k = args.k
        self.pos_c_weights = np.zeros(self.k)
        self.neg_c_weights = np.zeros(self.k)
        self.intercept = 0
        self.lr = args.lr
        self.cardinality_lr = args.lr
        self.rs = np.random.RandomState(seed=args.rs)
        self.loss_history = []
        self.visualize = args.v # Visualize instance labels
        self.norm = args.norm
        self.lpm = args.lpm # linear programming method
        self.momentun_beta = args.mom
        self.bgdm = args.bgdm

    def append_training_bags(self, features, bag_labels):
        """
        Batches data into training bags
        :param features: list of bag features containing list of data points
        :param bag_labels: lits of bag labels
        :return:None
        """
        self.training_bags = [] # list for training set
        feature_dimension = len(features[0][0]) # infers dimension of data
        self.logger.debug('Appending training bags')
        if "lp" in self.kernel or "qp" in self.kernel:
            self.weights = np.zeros(feature_dimension)
        else:
            self.weights = self.rs.rand(feature_dimension) # bgd and svm kernel
            self.momentum = np.zeros(feature_dimension+1+2*self.k)
        self.training_bags = [Bag(x,y) for x,y in zip(features, bag_labels)]



    def append_testing_bags(self, features):
        """
        Batches data into testing bags
        :param features: list of bag features containing list of data points
        :return: None
        """
        self.testing_bags = [Bag(x) for x in features]  # list for testing set

    def scoring_function(self, features, labels, bag_label):
        """
        Calculates score of bag
        :param features: np array containing instances
        :param labels: list containing instance labels
        :param bag_label: bag label {-1, 1}
        :return: score of the bag
        """
        score = 0
        positive_instances = 0
        for i in range(len(labels)):  # goes through all instance labels
            if labels[i] == 1:
                score += self.instance_potential(features[i], 1)
                positive_instances += 1
            if labels[i] == 0:
                score += self.instance_potential(features[i], 0)
        score = score + self.cardinality_potential(positive_instances, len(labels), bag_label)
        return score

    def loss_function(self):
        """
        Calculates loss function on dataset
        :returns:
        """
        loss = 0
        for bag in self.training_bags:
            pos_bag_labels = self.inference_on_bag(bag.features, bag.bag_label)
            neg_bag_labels = self.inference_on_bag(bag.features, -bag.bag_label)
            bag_loss = self.calculate_zeta(bag.features, bag.bag_label, pos_bag_labels, neg_bag_labels)
            loss += bag_loss
        if self.norm == 1:
            for weight in self.weights:
                loss+= abs(weight)*self.lamb/2
        else:
            for weight in self.weights:
                loss += (weight**2)*self.lamb/2
        return loss

    def predict(self, features):
        """
        Predicts the class of unseen np array
        :param features: list of features used for prediction
        :return: accuracy of model
        """
        #Instance based weights for noise dataset
        #self.weights = np.array([3.52007138, 3.57159067])
        #self.intercept = -3.5305451486024304

        #Instance based weights for image dataset
        #self.weights = np.array([-2.44514843e-05,  7.84113313e-03, -5.66825165e-05])
        #self.intercept = -0.99965145

        self.append_testing_bags(features)
        self.testing_logger = logger
        self.testing_logger.debug('Testing debugger Initiated\n')
        pred_bag_labels = []
        pred_instance_labels = []
        for t_bag in self.testing_bags:
            guessed_label, instance_labels = self.predict_bag_label(t_bag.features)
            pred_bag_labels.append(guessed_label)
            pred_instance_labels.append(instance_labels)
        return pred_bag_labels, pred_instance_labels

    def predict_bag_label(self, features):
        """
        Predicts bag label and  on unseen bag using inference
        :param features: list of instances
        :return: bag label {-1,1}
        """
        self.testing_logger.debug('Predicting bag label on testing bag\n')
        positive_bag_instances = self.inference_on_bag(features, 1)  # bag is positive find labels
        negative_bag_instances = self.inference_on_bag(features, -1)  # bag is negative find labels
        self.testing_logger.debug('Positive bag label instances: {}\n'.format(positive_bag_instances))
        self.testing_logger.debug('Negative bag label instances: {}\n'.format(negative_bag_instances))
        score_positive = self.scoring_function(features, positive_bag_instances, 1)  # calculate positive score
        score_negative = self.scoring_function(features, negative_bag_instances, -1)  # calculate negative score
        self.testing_logger.debug('Positive bag label score: {}\n'.format(score_positive))
        self.testing_logger.debug('Negative bag label score: {}\n'.format(score_negative))
        if score_positive > score_negative:  # determining maximum score
            return 1, positive_bag_instances
        else:
            return -1, negative_bag_instances

    def fit(self, features, bag_labels):
        """
        Fits model for the training data with different kernels
        :param features: list of bag features containing list of data points
        :param bag_labels: lits of bag labels
        :return: None
        """
        self.append_training_bags(features, bag_labels)
        if 'svm' in self.kernel:
            self.svm_train()
        if 'bgd' in self.kernel:
            self.bgd_train()
        if 'lp' in self.kernel:
            self.norm=1
            self.lp_train()
        if 'qp' in self.kernel:
            self.qp_train()
        self.logger = None #Unable to save model to pkl if logger is part of class
        #print('Vahy: {}, intercept: {}'.format(self.weights, self.intercept))

    def return_loss_history(self):
        return self.loss_history

    def return_weights(self):
        return self.weights, self.intercept
