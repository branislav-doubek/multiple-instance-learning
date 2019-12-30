from _methods import inference
import numpy as np
from scipy.optimize import linprog
from .utilities import total_instance_labels
from cvxopt import matrix, solvers
from warnings import filterwarnings
import sys

np.set_printoptions(threshold=sys.maxsize)

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
            print(bag.features)
            break
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
            g[counter][d + 1 + 2 * self.k + counter] = 1
            counter += 1
            p_matrix = np.r_
        return np.r_[g, np.c_[np.zeros((n, len(self.weights) + 2 * self.k + 1)), np.diag(np.ones(n))]]


    def a_matrix(self):
        counter = 0
        d,n = len(self.weights), len(self.training_bags)
        A = np.zeros(shape=(n, 2*d+1+2*self.k+n), dtype=np.float16)
        for bag in self.training_bags:
            q = self.q_matrix(bag)
            zeta = np.zeros(n+d)
            zeta[counter] = -1
            q = np.r_[q, zeta]
            A[counter] = q
            counter+=1
        helper_matrix = np.c_[np.zeros(shape=(n,d+1+2*self.k)), -np.eye(n), np.zeros(shape=(n, d))]
        absolute_constraint_1 = np.c_[self.lamb*np.eye(d), np.zeros(shape=(d, 1+2*self.k+n)), -np.eye(d)]
        absolute_constraint_2 = np.c_[-self.lamb*np.eye(d), np.zeros(shape=(d, 1+2*self.k+n)), -np.eye(d)]
        A = np.r_[A, helper_matrix, absolute_constraint_1, absolute_constraint_2]
        return A

    def q_matrix(self, bag):
        true_bag_labels = self.inference_on_bag(bag.features,
                                                bag.bag_label)  # finds labels for true label of the bag
        false_bag_labels = self.inference_on_bag(bag.features, -bag.bag_label)  # labels for false label
        total_instance_list = total_instance_labels(false_bag_labels, true_bag_labels)  # false - true labels
        weight = np.dot(bag.features[:][:].T, total_instance_list)
        intercept = np.sum(total_instance_list)
        pos_weights = np.zeros(self.k)
        neg_weights = np.zeros(self.k)
        pos_count = sum(true_bag_labels)
        neg_count = sum(false_bag_labels)
        length = len(true_bag_labels)
        if bag.bag_label == 1:
            for k_value in range(self.k):
                if k_value / self.k < pos_count / length <= (k_value + 1) / self.k:
                    pos_weights[k_value] += -1
                if k_value / self.k <= neg_count / length < (k_value + 1) / self.k:
                    neg_weights[k_value] += 1
        if bag.bag_label == -1:
            for k_value in range(self.k):
                if k_value / self.k <= pos_count / length < (k_value + 1) / self.k:
                    pos_weights[k_value] += 1
                if k_value / self.k < neg_count / length <= (k_value + 1) / self.k:
                    neg_weights[k_value] += -1
        return np.r_[weight, intercept, pos_weights, neg_weights]

    def lp_train(self):
        """
        Function which calculates linear solution to problem described as:
        Px
        s.t; Gx < h
        :return: None
        """
        filterwarnings('ignore')
        max_lr = self.lr
        epochs = 0
        for epoch in range(self.max_iterations):
            loss = self.loss_function()
            self.logger.error('{}-th iteration, loss = {}'.format(epoch, loss))
            self.loss_history.append(loss)
            d, n = len(self.weights), len(self.training_bags)  # basic dimensions
            c = np.r_[np.zeros(d+2 * self.k + 1, dtype=np.float16), np.ones(n+d, dtype=np.float16)]
            A = self.a_matrix() # creates G matrix
            b = np.r_[-np.ones(n), np.zeros(n+2*d)]  # creates h matrix
            sol = linprog(c, A, b, method=self.lpm, bounds=(-10,10))  # finds solution to linear programming problem
            if loss <= min(self.loss_history):
                self.save_parameters()
            self.get_solution(sol['x'])  # applies solution for new weights
        self.load_parameters()
    #def lp(self):


    def get_solution(self, sol):
        """
        :param sol:
        :return:
        """
        d, n = len(self.weights), len(self.training_bags)
        self.weights = sol[:d]
        self.intercept = sol[d]
        self.pos_c_weights = sol[d+1:d+1+self.k]
        self.neg_c_weights = sol[d + 1 + self.k:d + 1 + 2 * self.k]
        '''
        for alpha in alphas:
            self.weights = alpha*sol[:d]
            self.intercept = alpha*sol[d]
            self.pos_c_weights = alpha*sol[d+1:d+1+self.k]
            self.neg_c_weights = alpha*sol[d + 1 + self.k:d + 1 + 2 * self.k]
            loss = self.loss_function()
            print(loss)
            if self.loss_history[-1] <= loss:
                print(alpha)
                print('Does not minimize, other alpha')
                self.load_parameters()
            else:
                self.loss_history.append(loss)
                break
        '''
        #self.logger.error('New weights: {}\n'.format(self.weights))
        #self.logger.error('New intercept: {}\n'.format(self.intercept))
        #self.logger.error('New pos c weights: {}\n'.format(self.pos_c_weights))
        #self.logger.error('New neg c weights: {}\n'.format(self.neg_c_weights))
