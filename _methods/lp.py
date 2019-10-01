from _methods import inference
import numpy as np
from scipy.optimize import linprog
from .utilities import total_instance_labels
from cvxopt import matrix, solvers
from warnings import filterwarnings


class Lp:
    def bags_to_matrix(self):
        """
        Puts each bag into a matrix form with columns:
        weights - dimension - (d,1)
        intercept - dimension - (1,1)
        pos cardinality weights - dimension - (k,1)
        neg cardinality weights - dimension - (k,1)
        bags - dimension - (n,1)
        :return: matrix G - dimension - (d+2*k+1+n, 2*n)
        """
        d, n = len(self.weights), len(self.training_bags)
        g = np.zeros((n, n + d + 2 * self.k + 1))
        counter = 0
        for bag in self.training_bags:
            true_bag_labels = self.inference_on_bag(bag.features,
                                                    bag.bag_label)  # finds labels for true label of the bag
            false_bag_labels = self.inference_on_bag(bag.features, -bag.bag_label)  # labels for false label
            total_instance_list = total_instance_labels(false_bag_labels, true_bag_labels)  # false - true labels
            pos_count = sum(true_bag_labels)  # count of positive instances in true bag label
            neg_count = sum(false_bag_labels)  # count of positive instances in false bag label
            for i in range(len(bag.features)):
                for k in range(d):
                    g[counter][k] += total_instance_list[i] * bag.features[i][k]  # weights w grad
                g[counter][d] += total_instance_list[i]  # intercept point b grad
            if bag.bag_label == 1:
                for k_value in range(self.k):
                    if k_value / self.k < pos_count / len(true_bag_labels) <= (k_value + 1) / self.k:
                        g[counter][d + 1 + k_value] += -1  # positive cardinality weights grad
                    if k_value / self.k < neg_count / len(true_bag_labels) <= (k_value + 1) / self.k:
                        g[counter][d + 1 + self.k + k_value] += 1  # negative cardinality weights grad
            else:
                for k_value in range(self.k):
                    if k_value / self.k < pos_count / len(true_bag_labels) <= (k_value + 1) / self.k:
                        g[counter][d + 1 + k_value] += 1  # positive cardinality weights grad
                    if k_value / self.k < neg_count / len(true_bag_labels) <= (k_value + 1) / self.k:
                        g[counter][d + 1 + self.k + k_value] += -1
            g[counter][d + 1 + 2 * self.k + counter] = 1  # negative cardinality weights grad
            counter += 1
        return np.r_[g, np.c_[np.zeros((n, len(self.weights) + 2 * self.k + 1)), np.diag(np.ones(n))]]

    def lp_train(self):
        """
        Function which calculates linear solution to problem described as:
        Px
        s.t; Gx < h
        :return: None
        """
        filterwarnings('ignore')
        cond = True
        last_loss  = float('inf')
        for epoch in range(self.max_iterations):
            loss = self.loss_function()
            self.logger.debug('Vahy={}, b={}, pos_c={}, neg_c={}, lr={}, '
                              'c_lr={}, loss={}'.format(self.weights,
                                                        self.intercept,
                                                        self.pos_c_weights,
                                                        self.neg_c_weights,
                                                        self.lr,
                                                        self.cardinality_lr, loss))
            last_loss = loss
            d, n = len(self.weights), len(self.training_bags)  # basic dimensions
            p = np.r_[self.c*np.ones(d, dtype=float), np.zeros(2 * self.k + 1, dtype=float), np.ones(n, dtype=float)]  # creates P matrix
            g = self.bags_to_matrix()  # creates G matrix
            h = np.r_[-np.ones(n), np.zeros(n)]  # creates h matrix
            sol = linprog(p, g, h, method='interior-point')  # finds solution to linear programming problem
            self.get_weights(sol['x'])  # applies solution for new weights


    
    def get_weights(self, sol):
        """
        :param sol:
        :return:
        """
        d, n = len(self.weights), len(self.training_bags)
        self.weights -= self.lr * sol[:d]
        self.intercept -= self.lr * sol[d]
        self.pos_c_weights -= self.cardinality_lr * sol[d+1:d+1+self.k]
        self.neg_c_weights -= self.cardinality_lr * sol[d + 1 + self.k:d + 1 + 2 * self.k]