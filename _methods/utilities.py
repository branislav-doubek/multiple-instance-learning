import numpy as np
import random
from .setup_logger import logger
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def shuffle_dataset(features, bag_label, random_seed=42):
    """
    Random permutation of the dataset
    :param features: [list]
    :param bag_label: [list]
    :param random_seed: [int]]
    :return: shuffled dataset
    """
    index = list(zip(features, bag_label))
    random.seed(random_seed)
    random.shuffle(index)
    features, bag_label = zip(*index)
    return list(features), list(bag_label)


def f1_score(tp, fp, fn):
    """
    Calculates F1 score of the model
    :param tp: true_positive [int]
    :param fp: false_positive [int]
    :param fn: false_negative [int]
    :return: F1 score between 0 (worst) and 1 (best) [float]
    """
    try:
        return 2 * tp / (2 * tp + fp + fn)
    except Exception as e:
        logger.exception(e)
        logger.error('Division by zero')


def calculate_metrics(y_pred, y_actual):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            FP += 1
        if y_actual[i] == y_pred[i] == -1:
            TN += 1
        if y_pred[i] == -1 and y_actual[i] != y_pred[i]:
            FN += 1
    #print(confusion_matrix(y_actual, y_pred))
    return precision(TP, FP), recall(TP, FN), accuracy(TP, TN, FP, FN), f1_score(TP, FP, FN)
    
def recall(tp, fn):
    """
    Calculates recall
    :param tp: true positives [int]
    :param fn: false negatives [int]
    :return: [float]
    """
    if (fn + tp) != 0:
        return tp / (tp + fn)
    else:
        return 0


def precision(tp, fp):
    """
    Calculates precision
    :param tp: true positive [int]
    :param fp: false positive [int]
    :return: [float]
    """
    if (fp + tp) != 0:
        return tp/(tp+fp)
    else:
        return 0


def accuracy(tp, tn, fp, fn):
    """
    Calculates accuracy of the model
   :param tp:  true positive [int]
   :param tn:  true negative [int]
   :param fp:  false positive [int]
   :param fn:  false negative [int]
   :return: accuracy of the predictions [float]
    """
    return (tp + tn) / (tp + tn + fp + fn)


def total_instance_labels(b_neg_labels, b_pos_labels):
    """
    Helper function for subtracting 2 lists with the same length together
    :param b_neg_labels: list of labels for falsely labeled bag  [list]
    :param b_pos_labels: list of labels for true label of the bag [list]
    :return: [list]
    """
    total_instance_list = [negative - positive for negative, positive in
                           zip(b_neg_labels, b_pos_labels)]
    return total_instance_list


def batch_set(x, y, current_iter = 1, total_iter=5):
    """
    function - for each iterations i <= k will create new testing and training datasets for cross-validation
    :param features: list of features [list]
    :param bag_labels:  list of bag_labels [list]
    :param i: current iteration [int]
    :param k:  total number of iterations [int]
    :return: batched sets for training and testing
    """
    num_bags = len(x)
    batch = int((num_bags / (total_iter))//1)
    x_ds = [x[i:i + batch] for i in range(0, num_bags, batch)]
    y_ds = [y[i:i + batch] for i in range(0, num_bags, batch)]
    x_val = x_ds.pop(current_iter)  # validation set
    y_val = y_ds.pop(current_iter)  # validation set
    x_train = [item for sublist in x_ds for item in sublist]
    y_train = [item for sublist in y_ds for item in sublist]
    return x_train, x_val, y_train, y_val


def into_dictionary(index, features):
    """
    helper function for transforming dataset from list to dict
    :param index: index of the instance [list]
    :param features: features  [list]
    :return: dictionary [dict]
    """
    dictionary = {}
    for index, value in zip(index, features):
        if index in dictionary:
            dictionary[index].append(value)
        else:
            dictionary[index] = [value]
    return dictionary
