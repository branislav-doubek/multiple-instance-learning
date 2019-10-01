import warnings
import importlib
import argparse
import logging
from _methods.utilities import batch_set, calculate_metrics
from sklearn.model_selection import train_test_split
from mil import MIL
import pickle
import os
from collections.abc import Iterable

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
logging.basicConfig(level=logging.DEBUG, handlers=[console])
logging.getLogger("mil").setLevel(logging.DEBUG)
logger = logging.getLogger("main")


def model_path(args, split='train'):
    return '/_models/model_{}_{}_{}_{}.pkl'.format(split, args.dataset,
                                                args.cardinality_potential,
                                                args.rd)

def train_model(args, features, bag_labels):
    x_train, x_val, y_train, y_val = batch_set(features, bag_labels)
    model = MIL(args)
    model.fit(x_train, y_train)
    filepath = os.getcwd() + model_path(args)
    file = open(filepath, 'wb')
    pickle.dump(model, file)


def test_model(args, features, bag_labels):
    x_train, x_test, y_train, y_test = batch_set(features, bag_labels)
    filepath = os.getcwd() + model_path(args)
    try:
        with open(filepath, 'rb') as model_file:
            model = pickle.load(model_file)
            y_pred = model.predict(x_test)
            rec, prec, acc, f1 = calculate_metrics(y_pred, y_val)
            if args.v:
                visualize_inference(model, y_test, x_test)
    except FileNotFoundError as e:
        logger.exception(e)
        logger.error('No module named: {}'.format(filepath))

def run(args, features, bag_labels):
    x_train, x_test, y_train, y_test = train_test_split(features, bag_labels, shuffle=False, test_size=0.2)
    model = MIL(args)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    rec, prec, acc, f1 = calculate_metrics(y_pred, y_test)
    1e-8
    
    
def k_validation(args, features, bag_labels, k_valid=5):
    """
    Uses k_cross_validation to evaluate model
    :param args: arguments from parser [parser]
    :param features: list of bags  [list]
    :param bag_labels: list of bag labels [list]
    :return:
    """
    total_f1 = 0
    for cur_iteration in range(k_valid):
        x_train, x_val, y_train, y_val = batch_set(features, bag_labels, cur_iteration, k_valid)
        model = MIL(args)
        model.fit(x_train, y_train)
        filepath = os.getcwd() + model_path(args, cur_iteration)
        file = open(filepath, 'wb')
        pickle.dump(model, file)
        y_pred = model.predict(x_val)
        rec, prec, acc, f1 = calculate_metrics(y_pred, y_val)
        total_f1 += f1
    total_f1 /= k_valid
    return total_f1
        
def cross_validate(args, features, bag_labels):
    cross_validate = {'c': [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
                      'lr': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
                     'rho': [(a+1)/20 for a in range(18)],
                      'k': [a+3 for a in range(40)]}
    best_results = 0
    
    if 'lr' in args.cv:
        for param in cross_validate['lr']:
            args.lr = param
            results = k_validation(args, features, bag_labels)
            print("C-validation acc with lr={} is: {}".format(param, results))
            if best_results < results:
                best_results = results
          
    if 'c' in args.cv:
        for param in cross_validate['c']:
            args.c = param
            results = k_validation(args, features, bag_labels)
            print("C-validation acc with C={} is: {}".format(param, results))
            if best_results < results:
                best_results = results
            
                
    if 'rho' in args.cv:
        for param in cross_validate['rho']:
            args.rho = param
            results = k_validation(args, features, bag_labels)
            print("C-validation acc with rho={} is: {}".format(param, results))

    if 'k' in args.cv:
        for param in cross_validate['k']:
            args.k = param
            results = k_validation(args, features, bag_labels)
            print("C-validation acc with k={} is: {}".format(param, results))
            if best_results < results:
                best_results = results


def run_parser(args):
    has_effect = False
    if args.split and args.kernel and args.dataset and args.cardinality_potential:
        try:
            dataset_path = 'data.{}'.format(args.dataset)
            mod = importlib.import_module(dataset_path)
            features, bag_labels = mod.get_dataset(args.rd)

            x_train, x_test, y_train, y_test = train_test_split(features, bag_labels, shuffle=False, test_size=0.2)
            #x_train = [[[1, 1], [2,2]], [[1, 1], [2,2]], [[-1,-1],[-2,-2]], [[-1,-1],[-2,-2]], [[-1,1],[-2,2]], [[-1,1],[-2,2]], [[1,-1],[2,-2]], [[1,-1], [2,-2]],[[0, 1], [1,0]],[[100, 0], [1,1000]]]
            #y_train = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
            if 'test' in args.split:
                test_model(args, x_test, y_test)
            if 'train' in args.split:
                train_model(args, x_train, y_train)
            if 'cv' in args.split:  # run
                cross_validate(args, x_train, y_train)
            if 'run' in args.split:
                run(args, features, bag_labels)
                
        except Exception as e:
            logger.exception(e)
            logger.error('Script ended with an error')
    else:
        if not has_effect:
            logger.error("Script halted without any effect. Check the command")


def main():

    parser = argparse.ArgumentParser(description='Run examples from MIL framework:')
    parser.add_argument('split', nargs="?", choices=['train', 'test', 'cv' ,'run'],
                        help='select action you want to perform with model')
    parser.add_argument('kernel', nargs="?",
                        choices=['bgd', 'svm', 'lp', 'qp'],
                        help='Select kernel for fitting model')
    parser.add_argument('dataset', nargs="?",
                        choices=['fox', 'tiger', 'elephant', 'musk1', 'musk2', 'camelyon', 'synthetic', 'image'], help='Select the dataset')
    parser.add_argument('cardinality_potential', nargs="?",
                        choices=['mimn', 'rmimn', 'gmimn'],
                        help='Select the cardinality potential')
    parser.add_argument('-rho', nargs='?', default=0.5, type=float,
                        help='Select rho value used in rmimn potential')
    parser.add_argument('-c', nargs='?', default=1e-4, type=float, help='Select value for C')
    parser.add_argument('-iterations', nargs='?', default=1000, type=int,
                        help='Select number of iterations the model will train on')
    parser.add_argument('-k', nargs='?', default=1, type=int,
                        help='Select the number of bins used in gmimn potential')
    parser.add_argument('-rd', nargs='?', default=42, type=int, help='Select the random seed')
    parser.add_argument('-v', nargs='?', default=False, type=bool, help='Visualize data')
    parser.add_argument('-cv', nargs='?', default='c', choices=['c', 'lr', 'k', 'rho'], help='select which hyperparameter you want to cross validate model on')
    parser.add_argument('-dl', nargs='?', default=True, type=bool, help='Disable logger')
    parser.add_argument('-lr', nargs='?', default=1e-6, type=float, help='Learning rate')
    run_parser(parser.parse_args())


if __name__ == '__main__':
    main()
