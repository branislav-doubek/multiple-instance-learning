import numpy as np
import quadprog
from .utilities import total_instance_labels


class Qp:
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
        d, n = len(self.weights), len(self.training_list_of_buckets)
        g = np.zeros((n, n + d + 2 * self.k + 1))
        counter = 0
        for bag in self.training_list_of_buckets:
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

    def qp_train(self):
        """
        Function which calculates solution to quadratic programming problem described as:
        Solves equation
        0.5x^t* P * x + q^t * x
        subject to: Gx < h
        :return: None
        """
        for _ in range(self.max_iterations):
            d, n = len(self.weights), len(self.training_bags)
            self.logger.debug(self.loss_function())
            p = self.lamb/2 * np.diag(
                np.r_[np.ones(d),  1e-108*np.ones(n + 2 * self.k + 1)])
            q = np.r_[np.zeros(d + 2*self.k + 1), np.ones(n)]
            g = self.bags_to_matrix()
            h = np.r_[-np.ones(n), np.zeros(n)]
            solution = quadprog_solve_qp(p, q, g, h)
            print(solution)
            self.get_weights(solution)

    def get_weights(self, sol):
        """
        :param sol:
        :return:
        """
        d, n = len(self.weights), len(self.training_bags)
        self.weights -= self.lr * sol[:d]
        self.intercept -= self.lr * sol[d]
        self.pos_c_weights -= self.cardinality_lr * sol[d + 1:d + 1 + self.k]
        self.neg_c_weights -= self.cardinality_lr * sol[d + 1 + self.k:d + 1 + 2 * self.k]


def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    """
    Solves equation
    0.5x^t* P * x + q^t * x
    subject to: Gx < h
    :param P: matrix describing quadratic coefficients in loss function
    :param q: matrix describing linear coefficients in loss function
    :param G: matrix describing linear coefficients for constraints
    :param h: matrix describing scalar values in the constraints
    :return: the solution for described qudratic programming problem
    """
    qp_G = .5 * (P + P.T)  # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
