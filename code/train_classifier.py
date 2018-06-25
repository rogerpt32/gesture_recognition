
## Trains a One vs. Rest SVM classifier on the fisher vector video outputs.
## Outputs the optimal choice of hyperparameters in the GridSearch_output file

import os, sys, collections, random, string
import argparse
import pickle
import numpy as np
from tempfile import TemporaryFile
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as metrics
from sklearn.decomposition import PCA

import classify_library

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing different classifiers and print and analysis.')
    parser.add_argument('--no_pca', dest='no_pca', action='store_const',
                   const=True, default=False,
                   help='Training without PCA reduction (default: uses PCA)')
    parser.add_argument('--linear', dest='linear', action='store_const',
                   const=True, default=False,
                   help='Training with linear kernel (default: non-linear)')
    args = parser.parse_args()

    linear = args.linear

    class_index_file = "../data/class_index.npz"
    class_index_file_loaded = np.load(class_index_file)
    class_index = class_index_file_loaded['class_index'][()]
    index_class = class_index_file_loaded['index_class'][()]


    # In[7]:

    training_output = '../data/fishers/train'
    testing_output = '../data/fishers/test'

    training = [filename for filename in os.listdir(training_output) if filename.endswith('.fisher.npz')]
    testing = [filename for filename in os.listdir(testing_output) if filename.endswith('.fisher.npz')]


    training_dict = classify_library.toDict(training)
    testing_dict = classify_library.toDict(testing)

    ####################################################################
    ####################################################################
    ################################## Script starts




    X_train_vids = classify_library.limited_input1(training_dict, 1000)
    X_test_vids = classify_library.limited_input1(testing_dict, 1000)


    #GET THE TRAINING AND TESTING DATA.
    X_train, Y_train = classify_library.make_FV_matrix(X_train_vids,training_output, class_index)
    X_test, Y_test = classify_library.make_FV_matrix(X_test_vids,testing_output, class_index)

    X_total = np.concatenate((X_train, X_test),0)
    Y_total = np.concatenate((Y_train, Y_test),0)

    if not args.no_pca:
        #PCA reduction
        training_PCA = classify_library.limited_input1(training_dict,40)
        X_PCA, _ = classify_library.make_FV_matrix(training_PCA,training_output, class_index)

        n_components = 1000
        pca = PCA(n_components=n_components)
        pca.fit(X_PCA)
        classify_library.save_model(pca,'../data/models/pca')
        X_train_final = pca.transform(X_train)
        X_test_final = pca.transform(X_test)
        X_total_final = pca.transform(X_total)
    else:
        X_train_final = X_train
        X_test_final = X_test
        X_total_final = X_total
    if linear:
        #Linear SVM
        classifier = OneVsRestClassifier(LinearSVC(random_state=0, C=10, loss='squared_hinge', penalty='l2')).fit(X_total_final, Y_total)
    else:
        #non-linear SVM
        classifier = OneVsRestClassifier(svm.SVC(random_state=0, C=1000, kernel='rbf', gamma=0.01)).fit(X_total_final, Y_total)

    ## Save the best classifier
    if args.no_pca:
        save_name = '../data/models/svm_nopca'
    else:
        save_name = '../data/models/svm'

    classify_library.save_model(classifier,save_name)
