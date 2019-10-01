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
        self.training_bags = []
        self.testing_bags = []  # list for testing set
        self.cardinality = args.cardinality_potential  # cardinality type
        self.ro = args.rho  # value of rho for RMIMN potential
        self.kernel = args.kernel  # Kernel type
        self.lamb = 1/args.c  #lambda parameter value
        self.max_iterations = args.iterations  # sets max iterations
        self.k = args.k
        self.pos_c_weights = np.zeros(self.k)
        self.neg_c_weights = np.zeros(self.k)
        self.intercept = 0
        self.lr = args.lr
        self.cardinality_lr = args.lr
        self.rd = np.random.RandomState(seed=args.rd)
        self.loss_history = []
         
                
    def append_training_bags(self, features, bag_labels):
        """
        Batches data into training bags
        :param features: list of bag features containing list of data points
        :param bag_labels: lits of bag labels
        :return:None
        """

        feature_dimension = len(features[0][0])
        if "lp" in self.kernel or "qp" in self.kernel:
            self.weights = np.zeros(feature_dimension)
        else:
            self.weights = self.rd.rand(feature_dimension)
        for instance, bag_label in zip(features, bag_labels):
            bag = Bag(instance, bag_label)
            self.training_bags.append(bag)

    def append_testing_bags(self, features):
        """
        Batches data into testing bags
        :param features: list of bag features containing list of data points
        :param bag_labels: lits of bag labels
        :return: None
        """
        for instance in features:
            bag = Bag(instance)
            self.testing_bags.append(bag)

    def scoring_function(self, features, labels, bag_label):
        """
        Function calculating score of the bag
        :param features: np array
        :param labels: list of 1 and 0 with size of len(features)
        :param bag_label: int 1 or 0
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
        loss = 0
        for bag in self.training_bags:
            pos_bag_labels = self.inference_on_bag(bag.features, bag.bag_label)
            neg_bag_labels = self.inference_on_bag(bag.features, -bag.bag_label)
            bag_loss = self.calculate_zeta(bag.features, bag.bag_label, pos_bag_labels, neg_bag_labels)
            loss += bag_loss
        return loss

    def predict(self, features):
        """
        Predicts the class of unseen np array
        :param features: list of features used for prediction
        :return: accuracy of model
        """
        self.append_testing_bags(features)
        pred_y = []
        for t_bag in self.testing_bags:
            guessed_label = self.predict_bag_label(t_bag.features)
            pred_y.append(guessed_label)
        return pred_y

    def predict_bag_label(self, features):
        """
        Determines bag label of unseen bag
        :param features: list of instances
        :return: bag label {-1,1}
        """
        positive_bag_instances = self.inference_on_bag(features, 1)  # bag is positive find labels
        negative_bag_instances = self.inference_on_bag(features, -1)  # bag is negative find labels
        score_positive = self.scoring_function(features, positive_bag_instances, 1)  # calculate positive score
        score_negative = self.scoring_function(features, negative_bag_instances, -1)  # calculate negative score
        if score_positive > score_negative:  # determining maximum score
            return 1
        else:
            return -1

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
            self.lp_train()
        if 'qp' in self.kernel:
            self.qp_train()
        self.logger = None
        
    def return_loss_history(self):
        return self.loss_history