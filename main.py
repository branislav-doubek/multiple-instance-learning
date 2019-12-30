import warnings
import importlib
import argparse
import logging
import joblib
from _methods.utilities import *
from mil import MIL
import pickle
import os
from collections.abc import Iterable


def train(args, dataset):
    x_train, y_train = dataset.return_training_set()
    filepath = os.getcwd() + model_path(args)
    model = MIL(args)
    with open(filepath, 'wb') as model_file:
        model.fit(x_train, y_train)
        pickle.dump(model, model_file)
    y_pred, y_instance_pred = model.predict(x_train)
    if args.v:
        loss = model.return_loss_history()
        visualize_loss(args, loss)

def test(args, dataset):
    x_test, y_test = dataset.return_testing_set()
    filepath = os.getcwd() + model_path(args)
    try:
        with open(filepath, 'rb') as model_file:
            model = pickle.load(model_file)
            y_pred, y_instance_pred = model.predict(x_test)
            rec, prec, acc, f1 = calculate_metrics(y_pred, y_test, args.cm)
            print(acc)
            if args.v:
                visualize_inference(args, dataset, y_pred, y_instance_pred)
    except FileNotFoundError as e:
        logger.exception(e)
        logger.error('No module named: {}'.format(filepath))

def run(args, dataset):
    accuracies = []
    for run in range(5):
        dataset.random_shuffle()
        x_train, y_train = dataset.return_training_set()
        x_test, y_test = dataset.return_testing_set()
        model = MIL(args)
        model.fit(x_train, y_train)
        y_pred, y_instance_pred = model.predict(x_test)
        rec, prec, acc, f1 = calculate_metrics(y_pred, y_test, args.cm)
        accuracies.append(acc)
        print('Acc={}'.format(acc))
    mean = average(accuracies)
    std_dev = standard_deviaton(accuracies, mean)
    print('Result of evaluation: mean = {}, std={}'.format(mean, std_dev))

def k_validation(args, features, bag_labels, k_valid=5):
    """
    Uses k_cross_validation to evaluate model
    :param args: arguments from parser [parser]
    :param features: list of bags  [list]
    :param bag_labels: list of bag labels [list]
    :return:
    """
    accuracies = []
    #calculates 1 iterations of k-fold cv
    if 'validate' in args.split and args.valid_iter <= k_valid:
        cur_iteration = args.valid_iter
        x_train, x_val, y_train, y_val = batch_set(features, bag_labels, cur_iteration, k_valid)
        model = MIL(args)
        model.fit(x_train, y_train)
        y_pred, y_instance_pred = model.predict(x_val)
        rec, prec, acc, f1 = calculate_metrics(y_pred, y_val, args.cm)
        print('Acc={}'.format(acc))
        return acc
    else:
        for cur_iteration in range(k_valid):
            x_train, x_val, y_train, y_val = batch_set(features, bag_labels, cur_iteration, k_valid)
            model = MIL(args)
            model.fit(x_train, y_train)
            y_pred, y_instance_pred = model.predict(x_val)
            rec, prec, acc, f1 = calculate_metrics(y_pred, y_val, args.cm)
            accuracies.append(acc)
            print('Acc={}'.format(acc))
        mean = average(accuracies)
        print('Result of k-validation: mean = {}, std={}'.format(mean, standard_deviaton(accuracies, mean)))
        return mean

def cross_validate(args, dataset):
    features, bag_labels  = dataset.return_training_set()
    cross_validate = {'c': [0.01, 0.1, 1, 10, 1000],
                      'lr': [1e-5, 1e-4, 1e-3],
                     'ro': [(a+1)/10 for a in range(10)],
                      'k': [3, 5, 7, 10]}
    best_c = 0
    best_lr = 0
    best_ro = 0
    best_k = 0
    best_clr = 0
    best_iterations = 0
    result_ro = []
    result_kappa = []
    print('Starting cross validation')
    args.cardinality_lr = args.lr

    if (args.cv == 'lr' or 'all' in args.cv) and args.kernel != 'svm':
        print('Testing learning rate')
        best_results = 0
        for param in cross_validate['lr']:
            args.lr = param
            results = k_validation(args, features, bag_labels)
            print("C-validation acc with lr={} is: {}".format(param, results))
            if best_results <= results:
                best_results = results
                best_param = param
        args.lr = best_param
        print('Selected lr parameter:{}'.format(args.lr))
    args.cardinality_lr = args.lr

    if 'c' is args.cv or 'all' in args.cv:
        best_results = 0
        print('Testing c')
        for param in cross_validate['c']:
            args.c = param
            results = k_validation(args, features, bag_labels)
            print("C-validation acc with C={} is: {}".format(param, results))
            if best_results < results:
                best_results = results
                best_param = param
        args.c = best_param
        print('Selected c parameter:{}'.format(args.c))

    if ('ro' in args.cv or 'all' in args.cv) and 'rmimn' in args.cardinality_potential:
        print('Testing ro')
        best_results = 0
        for param in cross_validate['ro']:
            args.ro = param
            results = k_validation(args, features, bag_labels)
            result_ro.append(results)
            print("C-validation acc with ro={} is: {}".format(param, results))
            if best_results <= results:
                best_results = results
                best_param = param
        args.ro = best_param
        print('Selected ro parameter:{}'.format(args.ro))

    if ('k' in args.cv or 'all' in args.cv) and 'gmimn' in args.cardinality_potential:
        print('Testing k')
        best_results = 0
        for param in cross_validate['k']:
            args.k = param
            results = k_validation(args, features, bag_labels)
            print("C-validation acc with k={} is: {}".format(param, results))
            result_kappa.append(results)
            if best_results < results:
                best_results = results
                best_param = param
        args.k = best_param
        print('Selected k parameter:{}'.format(args.k))

    if 'gmimn' in args.cardinality_potential and args.v:
        visualize_kappa(args, result_kappa)

    if 'rmimn' in args.cardinality_potential and args.v:
        visualize_ro(args, result_ro)

    print('Selected args={}'.format(args))
    return args

def run_parser(args):
    has_effect = False
    if args.split and args.kernel and args.dataset and args.cardinality_potential:
        try:
            dataset_path = 'data.{}'.format(args.dataset)
            mod = importlib.import_module(dataset_path)
            dataset = mod.get_dataset(args)
            if 'run' in args.split:
                run(args, dataset)
            if 'test' in args.split:
                test(args, dataset)
            if 'train' in args.split:
                train(args, dataset)
            if 'cv' in args.split:  # run
                best_args = cross_validate(args, dataset)
                run(best_args, dataset)
            if 'validate' in args.split:
                features, bag_labels  = dataset.return_training_set()
                k_validation(args, features, bag_labels)

        except Exception as e:
            logger.exception(e)
            logger.error('Script ended with an error')
    else:
        if not has_effect:
            logger.error("Script halted without any effect. Check the command")


def main():

    parser = argparse.ArgumentParser(description='Run examples from MIL framework:')
    parser.add_argument('split', nargs="?", choices=['train', 'test', 'cv' ,'run', 'validate'],
                        help='select action you want to perform with model')
    parser.add_argument('kernel', nargs="?",
                        choices=['bgd', 'svm', 'lp', 'qp'],
                        help='Select kernel for fitting model')
    parser.add_argument('dataset', nargs="?",
                        choices=['fox', 'tiger', 'elephant', 'musk1', 'musk2', 'image', 'noise', 'camelyon', 'synthetic'], help='Select the dataset')
    parser.add_argument('cardinality_potential', nargs="?",
                        choices=['mimn', 'rmimn', 'gmimn'],
                        help='Select the cardinality potential')
    parser.add_argument('-ro', nargs='?', default=0.5, type=float,
                        help='Select ro value used in rmimn potential')
    parser.add_argument('-c', nargs='?', default=1, type=float, help='Select value for C')
    parser.add_argument('-iterations', nargs='?', default=1500, type=int,
                        help='Select number of iterations the model will train on')
    parser.add_argument('-k', nargs='?', default=1, type=int,
                        help='Select the number of bins used in gmimn potential')
    parser.add_argument('-rs', nargs='?', default=42, type=int, help='Select the random seed')
    parser.add_argument('-v', nargs='?', default=False, type=bool, help='Visualize data')
    parser.add_argument('-cv', nargs='?', default='all', choices=['c', 'lr', 'k', 'ro', 'clr', 'all'], help='Select which hyperparameter you want to cross validate model on')
    parser.add_argument('-lr', nargs='?', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('-cm', nargs='?', default=False, type=bool, help='Display confusion matrix at the end of testing')
    parser.add_argument('-norm', nargs='?', default=2, type=int, help='Select reguralization norm')
    parser.add_argument('-lpm', nargs='?', default='interior-point', choices=['interior-point', 'revised simplex', 'simplex'], help='Method for linear programming')
    parser.add_argument('-bgdm', nargs='?', default=1, choices=[1,2], help='Select batch gradient method, 1-momentum, 2- classical bgd')
    parser.add_argument('-mom', nargs='?', default=0.5, type=float, help='Momentum decay')
    parser.add_argument('-valid_iter', nargs='?', default=1, type=int, help='Select part of training dataset to validate model on')
    parser.add_argument('-multiply', nargs='?', default=False, type=bool, help='Multiply columns of instances to increase dimension' )
    run_parser(parser.parse_args())


if __name__ == '__main__':
    main()
