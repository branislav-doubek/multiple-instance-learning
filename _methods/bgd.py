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
        speed_up_counter = 0
        last_loss = float('inf')
        while True:
            loss, grad = self.batch_hinge_loss()
            self.logger.debug('Loss={}'.format(loss))
            self.loss_history.append(loss)
            if epochs % 10 == 0:
                self.logger.debug('{}-th iteration, loss = {}'.format(epochs, loss))
                #print('{}-th iteration, loss = {}'.format(epochs, loss))
            if speed_up_counter == 5:
                self.lr = self.lr * 2
                speed_up_counter = 0
            if epochs >= self.max_iterations or self.lr <= 1e-12 or loss < 0.01:
                break
            if last_loss <= loss:
                self.lr = self.lr / 2
                epochs +=1
            else:
                speed_up_counter += 1
                self.update_weights(grad)
                last_loss = loss
                epochs += 1

    def batch_hinge_loss(self):
        """
        Calculates the loss function across all bags
        :return: total loss
        """
        grad1 = np.zeros(len(self.weights)+1, dtype=float)  # for gradients of normal weights
        total_loss = 0
        grad2 = np.zeros(2 * self.k, dtype=float) # for gradients of intercept point and cardinality weights
        for bag in self.training_bags:
            pos_bag_labels = self.inference_on_bag(bag.features, bag.bag_label)
            neg_bag_labels = self.inference_on_bag(bag.features, -bag.bag_label)
            bag_loss = self.calculate_zeta(bag.features, bag.bag_label, pos_bag_labels, neg_bag_labels)
            total_instance_list = total_instance_labels(neg_bag_labels, pos_bag_labels)
            total_loss += bag_loss
            if bag_loss != 0:
                for feature, label in zip(bag.features, total_instance_list):
                    for index, element in enumerate(feature * label):
                        grad1[index] += element
                    grad1[-1] += label # intercept point
                total_instances = len(pos_bag_labels)
                pos_ratio = sum(pos_bag_labels) / total_instances
                neg_ratio = sum(neg_bag_labels) / total_instances
                if 'gmimn' in self.cardinality:
                    if bag.bag_label == 1:
                        grad2[int(pos_ratio // (1 / self.k) - 1)] += -1
                        grad2[self.k + int(neg_ratio // (1 / self.k) - 1)] += 1
                    else:
                        grad2[int(pos_ratio // (1 / self.k) - 1)] += 1
                        grad2[self.k + int(neg_ratio // (1 / self.k) - 1)] += -1
                else:
                    if bag.bag_label == 1:
                        grad2[0] += -1
                        grad2[1] += 1
                    else:
                        grad2[0] += 1
                        grad2[1] += -1
        grad1[:len(self.weights)] += self.lamb * self.weights
        grad = np.r_[grad1, grad2]
        for weight in self.weights:
            total_loss += self.lamb * (weight ** 2)/2
        self.loss_history.append(total_loss)
        return total_loss, grad

    def update_weights(self, grad):
        """
        uses momentum gradient descent to update weights, intercept point and cardinality weights
        :param self: object - classifier
        :param grad: np array with the size 2K+1+weights
        :return: weights, b, w_c+, w_c-
        """
        d = len(self.weights)
        weights = np.r_[self.weights, self.intercept, self.pos_c_weights, self.neg_c_weights]
        c_weights = weights[d+1:] - self.cardinality_lr * grad[d+1:]
        weights = weights[:d+1] - self.lr * grad[:d+1]
        self.weights = weights[:d]
        self.intercept = weights[d]
        self.pos_c_weights = c_weights[:self.k]
        self.neg_c_weights = c_weights[self.k:]

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
        for feature, delta_label in zip(features, delta_labels):
            score += delta_label * (np.matmul(feature, self.weights) + self.intercept)
        if b_label == 1:
            score += self.cardinality_potential(neg_count, length, -b_label) - \
                     self.cardinality_potential(pos_count, length, b_label)
        else:
            score += self.cardinality_potential(pos_count, length, b_label) - \
                     self.cardinality_potential(neg_count, length, -b_label)
        return max(0, 1 + score)
    