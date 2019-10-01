import numpy as np

class Inference:
    def cardinality_potential(self, positive_instances, total_instances, bag_label):
        """
        Used to calculate cardinality_potential
        :param positive_instances: count of positive instances
        :param total_instances:  count of all instances
        :param bag_label:  bag label
        :return:  cardinality weights
        """
        if 'gmimn' in self.cardinality:  # GMIMN
            if bag_label == 1:
                if positive_instances == 0:
                    return -float('inf')
                else:
                    for k_value in range(self.k):
                        if k_value / self.k < (positive_instances / total_instances) <= (k_value + 1) / self.k:
                            return self.pos_c_weights[k_value]
            else:
                if total_instances - positive_instances == 0:
                    return -float('inf')
                else:
                    for k_value in range(self.k):
                        if k_value / self.k <= (positive_instances / total_instances) < (k_value + 1) / self.k:
                            return self.neg_c_weights[k_value]
        if 'rmimn' in self.cardinality:  # RMINM
            if bag_label == 1:
                if positive_instances / total_instances >= self.ro:
                    return self.pos_c_weights
                else:
                    return -float('inf')
            else:
                if positive_instances / total_instances >= self.ro:
                    return -float('inf')
                else:
                    return self.neg_c_weights
        if 'mimn' in self.cardinality:  # MIMN
            if bag_label == 1:
                if positive_instances != 0:
                    return self.pos_c_weights
                else:
                    return -float('inf')
            else:
                if positive_instances != 0:
                    return -float('inf')
                else:
                    return self.neg_c_weights

    def instance_potential(self, feature, label=1):
        """
        :param feature: np.array()
        :param label: label of an instance
        :return: float
        """
        return label * (np.dot(feature, self.weights) + self.intercept)

    def inference_on_bag(self, features, bag_label):
        """
        Finds instance labels via inference
        :param features:  list of np.arrays
        :param bag_label:  int
        :returns: list of labels (positive = 1, negative = 0)
        """
        total_features = len(features)
        if self.logger:
            self.logger.debug('Inference on bag with label {} containing {} instances'.format(bag_label, total_features))
            #self.logger.debug('Instances={}'.format(features))
        potential = np.zeros(total_features, dtype=float)  # creates an array
        for i in range(total_features):
            potential[i] = self.instance_potential(features[i])
        # finds the maximum sum and returns indexes for positive instances
        positive_indexes = self.find_max_sum(potential, bag_label)
        labels = np.zeros(total_features)
        for index in positive_indexes:  # positive instances=1, negative = 0
            labels[index] = 1
        if self.logger:
            self.logger.debug('Count of positive instance labels={}'.format(len(positive_indexes)))
            self.logger.debug('Count of negative instance labels={}'.format(total_features-len(positive_indexes)))
        return labels

    def find_max_sum(self, h, bag_label):
        """
        Finds indexes, for which score of the bag is maximized
        :param h: list of potentials (product of weights and features)
        :param bag_label: int
        :return: indexes of positive labels
        """
        indexes = h.argsort()[:]  # get indexes from sorting h in ascending form
        indexes = np.flipud(indexes)  # descending indexes
        sorted_h = -np.sort(-h)  # array sorted in decreasing order
        max_sum, f_length = float('-inf'), 0
        for i in range(len(h) + 1):
            curr_sum = sum(sorted_h[:i]) + self.cardinality_potential(i, len(h), bag_label)
            if curr_sum >= max_sum:
                max_sum = curr_sum
                f_length = i
            if self.logger:
                self.logger.debug('{}-iteration: current sum = {}, max sum = {}'.format(i, curr_sum, max_sum))

        return indexes[:f_length]
