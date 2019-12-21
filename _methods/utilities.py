import numpy as np
import random
from .setup_logger import logger
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import sqrt
from itertools import combinations

def shuffle_dataset(features, bag_label, random_seed=42, instance_labels=None):
    """
    Random permutation of the dataset
    :param features: [list]
    :param bag_label: [list]
    :param random_seed: [int]]
    :return: shuffled dataset
    """
    if instance_labels:
        index = list(zip(features, bag_label, instance_labels))
    else:
        index = list(zip(features, bag_label))
    random.seed(random_seed)
    random.shuffle(index)
    if instance_labels:
        features, bag_label, instance_labels = zip(*index)
        return list(features), list(bag_label), list(instance_labels)
    else:
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


def calculate_metrics(y_pred, y_true, visualize):
    arr_pred = np.array(y_pred)
    arr_true = np.array(y_true)
    tp = np.sum(np.logical_and(arr_pred == 1, arr_true == 1))
    tn = np.sum(np.logical_and(arr_pred == -1, arr_true == -1))
    fp = np.sum(np.logical_and(arr_pred == 1, arr_true == -1))
    fn = np.sum(np.logical_and(arr_pred == -1, arr_true == 1))
    if visualize:
        print(confusion_matrix(y_true, y_pred))
    return precision(tp, fp), recall(tp, fn), accuracy(tp, tn, fp, fn), f1_score(tp, fp, fn)

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

def visualize_inference(args, dataset, y_pred, iy_pred):
    x_test, y_true, iy_true = dataset.return_all_testing_data()
    counter = 0
    #iterate through all testing bags
    #over testing bag, true bag label, true instance labels, predicted bag label, predicted isntance labels
    for X_data, yb_true, ybi_true, yb_pred, ybi_pred in zip(x_test, y_true, iy_true,  y_pred, iy_pred):
        tp=[]
        tn=[]
        fp=[]
        fn=[]
        pos_data = []
        neg_data = []
        if 'noise' in dataset.return_name():
            for data, true_label, pred_label in zip(X_data, ybi_true, ybi_pred):
                if true_label ==1:
                    pos_data.append(data)
                    if pred_label == 1:
                        tp.append(data)
                    else:
                        fn.append(data)
                else:
                    neg_data.append(data)
                    if pred_label == 0:
                        tn.append(data)
                    else:
                        fp.append(data)
            tp_arr = np.array(tp)
            tn_arr = np.array(tn)
            fp_arr = np.array(fp)
            fn_arr = np.array(fn)
            pos_arr = np.array(pos_data)
            neg_arr = np.array(neg_data)
            fig = plt.figure(figsize=(10,6))
            ax1 = fig.add_subplot(121)
            if yb_true == 1:
                plt.title('Positive bag')
            else:
                plt.title('Negative bag')
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(-2,2)
            plt.ylim(-2,2)
            if pos_arr.size !=0:
                ax1.scatter(pos_arr[:,0], pos_arr[:,1], s=20, c='g', marker="s", label='Positive instance')
            if neg_arr.size !=0:
                ax1.scatter(neg_arr[:,0], neg_arr[:,1], s=20, c='r', marker="s", label='Negative instance')
            plt.legend(loc=0)
            ax1 = fig.add_subplot(122)
            plt.title('Predicted bag label {}'.format(yb_pred))
            plt.xlim(-2,2)
            plt.ylim(-2,2)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid()
            if tp_arr.size !=0:
                ax1.scatter(tp_arr[:,0], tp_arr[:,1], s=20, c='g', marker="s", label='True positive')
            if tn_arr.size !=0:
                ax1.scatter(tn_arr[:,0], tn_arr[:,1], s=20, c='r', marker="s", label='True negative')
            if fn_arr.size !=0:
                ax1.scatter(fn_arr[:,0], fn_arr[:,1], s=20, c='g', marker="x", label='False negative')
            if fp_arr.size !=0:
                ax1.scatter(fp_arr[:,0], fp_arr[:,1], s=20, c='y', marker="x", label='False positive')
            plt.legend(loc=0)
            if yb_true == yb_pred and yb_true == 1:
                plt.savefig(os.getcwd() + '/data/scatter_plot_{}.png'.format(counter))
            if yb_true == yb_pred and yb_true == -1:
                plt.savefig(os.getcwd() + '/data/scatter_plot_{}.png'.format(counter))
            if yb_true != yb_pred and yb_true == 1:
                plt.savefig(os.getcwd() + '/data/scatter_plot_{}.png'.format(counter))
            if yb_true != yb_pred and yb_true == -1:
                plt.savefig(os.getcwd() + '/data/scatter_plot_{}.png'.format(counter))
            counter+=1
        else:
            h,w = dataset.return_dimensions()
            true_image = np.array(X_data).reshape((h,w,3))
            modified_image= np.copy(true_image)
            instance_labels_true = np.array(ybi_true).reshape(h,w)
            instance_labels_pred = np.array(ybi_pred).reshape(h,w)
            idx_tp=(instance_labels_pred == 1) & (instance_labels_true == 1)
            idx_tn=(instance_labels_pred == 0) & (instance_labels_true == 0)
            idx_fp=(instance_labels_pred == 1) & (instance_labels_true == 0)
            idx_fn=(instance_labels_pred == 0) & (instance_labels_true == 1)
            modified_image[idx_tp]=[0,255,0]
            modified_image[idx_tn]=[0,0,0]
            modified_image[idx_fn]=[255,0,0]
            modified_image[idx_fp]=[0,0,255]
            red_patch = mpatches.Patch(color='red', label='False negative')
            green_patch = mpatches.Patch(color=[0,1,0], label='True postive')
            black_patch = mpatches.Patch(color='blue', label='False positive')
            blue_patch = mpatches.Patch(color='black', label='True negative')
            plt.subplot(121)
            plt.title('True image')
            plt.imshow(true_image, interpolation='nearest')
            plt.axis('off')
            plt.subplot(122)
            plt.title('Predicted instance labels')
            plt.imshow(modified_image, interpolation='nearest')
            plt.axis('off')
            handles = [red_patch, green_patch, black_patch, blue_patch]
            plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0., handles= handles)
            plt.suptitle('Inference visualization, true label {}, predicted {}'.format(yb_true, yb_pred))
            if yb_true == yb_pred and yb_true == 1:
                plt.savefig(os.getcwd() + '/data/image_plot_{}.png'.format(counter))
            if yb_true == yb_pred and yb_true == -1:
                plt.savefig(os.getcwd() + '/data/image_plot_{}.png'.format(counter))
            if yb_true != yb_pred and yb_true == 1:
                plt.savefig(os.getcwd() + '/data/image_plot_{}.png'.format(counter))
            if yb_true != yb_pred and yb_true == -1:
                plt.savefig(os.getcwd() + '/data/image_plot_{}.png'.format(counter))
            #plt.show()
            counter+=1

def train_test_split(features, bag_labels, instance_labels=None):
    split = 0.75
    num_bags = len(features)
    index = int(split * num_bags)
    if instance_labels:
        return features[:index], features[index:], bag_labels[:index], bag_labels[index:], instance_labels[:index], instance_labels[index:]
    else:
        return features[:index], features[index:], bag_labels[:index], bag_labels[index:]


def model_path(args, split='train'):
    return '/model_{}_{}_{}_{}_{}.pkl'.format(split, args.kernel, args.dataset,
                                                args.cardinality_potential,
                                                args.rs)

def visualize_ro(args, ro_results):
    x = np.linspace(0.1,0.9, num=9)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.grid()
    plt.ylim(0,1)
    plt.xlabel('ρ')
    plt.xlim(0.1,0.9)
    plt.ylabel('accuracy')
    plt.plot(x,ro_results, '-ro')
    plt.savefig(os.getcwd() + '/{}_{}_ro.png'.format(args.dataset, args.kernel))

def visualize_kappa(args, kappa_results):
    x = [3, 5, 7, 10]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.grid()
    plt.ylim(0,1)
    plt.xlabel('κ')
    plt.xlim(3,10)
    plt.ylabel('accuracy')
    plt.plot(x,kappa_results, '-ro')
    plt.savefig(os.getcwd() + '/{}_{}_kappa.png'.format(args.dataset, args.kernel))

def visualize_loss(args, loss):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    x = [a for a in range(len(loss))]
    plt.grid()
    plt.xlabel('Epochs')
    plt.xlim(0, len(loss))
    plt.ylim(0, max(loss))
    plt.ylabel('Loss')
    plt.plot(x,loss, '-ro')
    plt.savefig(os.getcwd() + '/{}_{}_{}_loss.png'.format(args.dataset, args.kernel, args.cardinality_potential))

def ravel_image(data):
    dataset = []
    for bag in data:
        image_data = []
        for row in bag:
            for instance in row:
                image_data.append(instance)
        dataset.append(image_data)
    return dataset

def average(lst):
    return sum(lst) / len(lst)

def standard_deviaton(lst, mean):
    std = 0
    for element in lst:
        std += (element-mean)**2
    std = std/(len(lst)-1)
    return sqrt(std)

def multiply_features(list_of_instance):
    num, length = list_of_instance.shape
    all_combinations = list(combinations([a for a in range(length)],2))
    for el1, el2 in all_combinations:
        #print('({},{}) combination'.format(el1, el2))
        list_of_instance = np.c_[list_of_instance, list_of_instance[:,el1][:,] * list_of_instance[:,el2][:,]]
    return list_of_instance
