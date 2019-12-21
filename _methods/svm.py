from sklearn.svm import LinearSVC
import logging
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class Mi_SVM:
    @ignore_warnings(category=ConvergenceWarning)
    def svm_train(self):
        self.logger.debug('Starting svm\n')
        clf = LinearSVC(loss='squared_hinge', random_state=0, max_iter=1, C=1/self.lamb)
        epochs = 0
        labels_changed = True
        last_labels = []
        while labels_changed and epochs <= self.max_iterations:
            loss = self.loss_function()
            if epochs % 1 == 0:
                self.logger.error('Epoch={}: Loss = {}'.format(epochs,loss))
            self.loss_history.append(loss)
            labels = []
            features = []
            for bag in self.training_bags:
                guessed_labels = self.inference_on_bag(bag.features, bag.bag_label)
                features.extend(bag.features)
                labels.extend(guessed_labels)
            if labels == last_labels:
                self.logger.debug('Labels stopped changing')
                labels_changed = False
            clf.fit(features, labels)
            self.weights = clf.coef_[0]
            self.intercept = float(clf.intercept_)
            epochs += 1
            last_labels = labels
