import numpy as np
from _methods.utilities import total_instance_labels
import logging

class BatchGradientDescent:
    def bgd_train(self):
        """
        Minimizes loss function via batch gradient descent
        :return: None
        """
        epochs = 0
        last_loss = float('inf')
        while True:
            loss, grad = self.batch_hinge_loss()
            #Uncomment for loss visualization
            if epochs % 2 == 0:
                print('Epoch={}: Loss = {}'.format(epochs,loss))
            self.loss_history.append(loss)
            if epochs >= self.max_iterations or self.lr <= 1e-8 or loss < 0.01:
                break
            if last_loss <= loss:
                self.lr /= 10
                epochs +=1
            else:
                if self.bgdm == 2:
                    self.update_weights(grad)
                else:
                    self.momentum_weights(grad)
                last_loss = loss
                epochs += 1

    def batch_hinge_loss(self):
        """
        Calculates the loss function across all bags and its gradients
        :return: total loss and gradient
        """
        grad1 = np.zeros(len(self.weights)+1, dtype=float)  # gradients of normal weights + intercept point
        total_loss = 0
        grad2 = np.zeros(2 * self.k, dtype=float) # gradients of cardinality weights
        for bag in self.training_bags:
            pos_bag_labels = self.inference_on_bag(bag.features, bag.bag_label)
            neg_bag_labels = self.inference_on_bag(bag.features, -bag.bag_label)
            self.logger.info('Positive instance labels = {}\n'.format(pos_bag_labels))
            self.logger.info('Negative instance labels= {}\n'.format(neg_bag_labels))
            bag_loss = self.calculate_zeta(bag.features, bag.bag_label, pos_bag_labels, neg_bag_labels)
            total_instance_list = total_instance_labels(neg_bag_labels, pos_bag_labels)
            total_loss += bag_loss
            self.logger.info('Bag loss = {}\n'.format(bag_loss))
            if bag_loss != 0:
                for feature, label in zip(bag.features, total_instance_list):
                    for index, element in enumerate(feature * label):
                        grad1[index] += element # weights
                    grad1[-1] += label # intercept point
                num_instances = len(pos_bag_labels) #
                pos_ratio = sum(pos_bag_labels) / num_instances
                neg_ratio = sum(neg_bag_labels) / num_instances
                if 'gmimn' in self.cardinality: # only for gmimn
                    if bag.bag_label == 1:
                        grad2[int(pos_ratio // (1 / self.k) - 1)] += -1
                        grad2[self.k + int(neg_ratio // (1 / self.k) - 1)] += 1
                    else:
                        grad2[int(pos_ratio // (1 / self.k) - 1)] += 1
                        grad2[self.k + int(neg_ratio // (1 / self.k) - 1)] += -1
                else: # mimn and rmimn
                    if bag.bag_label == 1:
                        grad2[0] += -1
                        grad2[1] += 1
                    else:
                        grad2[0] += 1
                        grad2[1] += -1
        if self.norm == 2: # reguralization norm l2
            grad1[:len(self.weights)] += self.lamb * self.weights
            for weight in self.weights:
                total_loss += self.lamb * (weight ** 2)/2
        else: # reguralization norm l1
            grad1[:len(self.weights)] += self.lamb / 2 * np.sign(self.weights)
            for weight in self.weights:
                total_loss += self.lamb * abs(weight)/2
        grad = np.r_[grad1, grad2]
        return total_loss, grad

    def update_weights(self, grad):
        """
        Updates weights with newly calculated gradient
        :param grad: np array with the size 2K+1+d
        :return: None
        """
        d = len(self.weights)
        weights = np.r_[self.weights, self.intercept, self.pos_c_weights, self.neg_c_weights]
        c_weights = weights[d+1:] - self.cardinality_lr * grad[d+1:]
        weights = weights[:d+1] - self.lr * grad[:d+1]
        self.logger.info('New weights: {}\n'.format(self.weights))
        self.weights = weights[:d]
        self.logger.info('New intercept: {}\n'.format(self.intercept))
        self.intercept = weights[d]
        self.logger.info('New pos c weights: {}\n'.format(self.pos_c_weights))
        self.pos_c_weights = c_weights[:self.k]
        self.logger.info('New neg c weights: {}\n'.format(self.neg_c_weights))
        self.neg_c_weights = c_weights[self.k:]

    def momentum_weights(self,grad):
        d = len(self.weights)
        weights = np.r_[self.weights, self.intercept, self.pos_c_weights, self.neg_c_weights]
        delta_weights = self.momentun_beta*self.momentum - self.lr * grad
        self.weights = weights[:d] + delta_weights[:d]
        self.intercept = weights[d] + delta_weights[d]
        self.pos_c_weights = weights[d+1:d+1+self.kappa] + delta_weights[d+1:d+1+self.kappa]
        self.neg_c_weights = weights[d+1+self.kappa:] + delta_weights[d+1+self.kappa:]

    def calculate_zeta(self, features, b_label, true_labels, false_labels):
        """
        Calculates the zeta function for bag and returns hinge loss like output
        :param features: array of instances in bag
        :param b_label: label of bag
        :param true_labels: labels of true labeled bag
        :param false_labels: labels of falsely labeled bag
        :return: maximum value out of hinge-like function
        """
        score = 0
        length = len(true_labels)
        pos_count = sum(true_labels)
        neg_count = sum(false_labels)
        delta_labels = total_instance_labels(false_labels, true_labels)
        self.logger.info('True bag label: {}, delta labels: {}\n'.format(b_label, delta_labels))
        for feature, delta_label in zip(features, delta_labels):
            sumant = delta_label * (np.matmul(feature, self.weights) + self.intercept)
            self.logger.info('Multiplication: {}({} x {} + {}) = {}\n'.format(delta_label, self.weights, feature, self.intercept, sumant))
            score += sumant
        if b_label == 1:
            score += self.cardinality_potential(neg_count, length, -b_label) - \
                     self.cardinality_potential(pos_count, length, b_label)
        else:
            score += self.cardinality_potential(pos_count, length, b_label) - \
                     self.cardinality_potential(neg_count, length, -b_label)
        return max(0, 1 + score)
