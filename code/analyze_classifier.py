
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
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as metrics
from sklearn.decomposition import PCA

import classify_library

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing different classifiers and print and analysis.')
    parser.add_argument('--no_pca', dest='no_pca', action='store_const',
                   const=True, default=False,
                   help='Testing without PCA reduction (default: uses PCA)')
    args = parser.parse_args()

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
    #Exhaustive Grid Search

    C = [1, 10, 50, 100, 1000]
    loss = ['hinge', 'squared_hinge']
    penalty = ['l2']
    kernel = ['poly', 'rbf', 'sigmoid']
    gamma = [0.01, 0.001, 0.0001]

    best_linear = 0
    best_non_linear = 0

    best_c_l = 1
    best_c_nl = 1
    best_loss = 'hinge'
    best_pen = 'l2'
    best_ker = 'rbf'
    best_gam = 0.01

    for c in C:
        #Optimize the linear kernel first
        for lo in loss:
            for pen in penalty:
                classifier = OneVsRestClassifier(LinearSVC(C=c, loss=lo, penalty=pen))
                scores = cross_val_score(classifier, X_total_final, Y_total, cv=9)
                Y_pred = cross_val_predict(classifier,X_total_final,Y_total, cv=9)
                conf_mat = metrics.confusion_matrix(Y_total,Y_pred)
                if best_linear < scores.mean():
                    conf_mat_l = conf_mat
                    best_linear = scores.mean()
                    best_c_l = c
                    best_loss = lo
                    best_pen = pen
                print 'Linear: C = '+str(c)+', Loss = '+lo+', Penalty = '+str(pen)
                print 'Precisions of CV = '+str(scores)
                print 'Mean Precision = '+str(scores.mean())
                print 'Confusion Matrix = '
                print conf_mat
                print '\n-----------------------------------------------------\n'
        #Optimize the non-linear kernels
        for ker in kernel:
            for gam in gamma:
                classifier = OneVsRestClassifier(svm.SVC(C=c, kernel=ker, gamma=gam))
                scores = cross_val_score(classifier, X_total_final, Y_total, cv=9)
                Y_pred = cross_val_predict(classifier,X_total_final,Y_total, cv=9)
                conf_mat = metrics.confusion_matrix(Y_total,Y_pred)
                if best_non_linear < scores.mean():
                    conf_mat_nl = conf_mat
                    best_non_linear = scores.mean()
                    best_c_nl = c
                    best_ker = ker
                    best_gam = gam
                print 'Non-linear: C = '+str(c)+', kernel = '+ker+', Gamma = '+str(gam)
                print 'Precisions of CV = '+str(scores)
                print 'Mean Precision = '+str(scores.mean())
                print 'Confusion Matrix = '
                print conf_mat
                print '\n-----------------------------------------------------\n'
    print '\n########################################################\n'
    print 'BEST LINEAR SVM (CV precision = '+str(best_linear*100)+' %)\n'
    #Best Linear SVM
    classifier_l = OneVsRestClassifier(LinearSVC(random_state=0, C=best_c_l, loss=best_loss, penalty=best_pen)).fit(X_train_final, Y_train)
    Scores = classify_library.metric_scores(classifier_l, X_test_final, Y_test)
    print "Settings: Linear SVM, C: %d, loss: %s, penalty: %s" % (best_c_l,best_loss,best_pen)
    print "Scores in test: (%f, %f, %f, %f)\n" % (Scores[0], Scores[1], Scores[2], Scores[3])
    print "Confusion Matrix = "
    print conf_mat_l

    print '\n########################################################\n'
    print 'BEST NON-LINEAR SVM (CV precision = '+str(best_non_linear*100)+' %)\n'
    #Best non-linear SVM
    classifier_nl = OneVsRestClassifier(svm.SVC(random_state=0, C=best_c_nl, kernel=best_ker, gamma=best_gam)).fit(X_train_final, Y_train)
    Scores = classify_library.metric_scores(classifier_nl, X_test_final, Y_test)
    print "Settings: SVM kernel: %s, C: %d, gamma: %f" % (best_ker,best_c_nl,best_gam)
    print "Scores in test: (%f, %f, %f, %f)\n" % (Scores[0], Scores[1], Scores[2], Scores[3])
    print "Confusion Matrix = "
    print conf_mat_nl

    print '\n########################################################'
    ## Save the best classifier
    if args.no_pca:
        save_name = '../data/models/svm_nopca'
    else:
        save_name = '../data/models/svm'
    if best_linear>best_non_linear:
        classify_library.save_model(classifier_l,save_name)
    else:
        classify_library.save_model(classifier_nl,save_name)
