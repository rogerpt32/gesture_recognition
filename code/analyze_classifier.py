
## Trains a One vs. Rest SVM classifier on the fisher vector video outputs.
## Outputs the optimal choice of hyperparameters in the GridSearch_output file

import os, sys, collections, random, string
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

#PCA reduction
training_PCA = classify_library.limited_input1(training_dict,40)
X_PCA, _ = classify_library.make_FV_matrix(training_PCA,training_output, class_index)

n_components = 1000
pca = PCA(n_components=n_components)
pca.fit(X_PCA)
classify_library.save_model(pca,'../data/models/pca')
X_train_PCA = pca.transform(X_train)
X_test_PCA = pca.transform(X_test)
X_total_PCA = pca.transform(X_total)

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
            scores = cross_val_score(classifier, X_total_PCA, Y_total, cv=9)
            if best_linear < scores.mean():
                best_linear = scores.mean()
                best_c_l = c
                best_loss = lo
                # best_pen = pen
            print 'Linear: C = '+str(c)+', Loss = '+lo+', Penalty = '+str(pen)
            print 'Precisions of CV = '+str(scores)
            print 'Mean Precision = '+str(scores.mean())+'\n'
    #Optimize the non-linear kernels
    for ker in kernel:
        for gam in gamma:
            classifier = OneVsRestClassifier(svm.SVC(C=c, kernel=ker, gamma=gam))
            scores = cross_val_score(classifier, X_total_PCA, Y_total, cv=9)
            if best_non_linear < scores.mean():
                best_non_linear = scores.mean()
                best_c_nl = c
                best_ker = ker
                best_gam = gam
            print 'Non-linear: C = '+str(c)+', kernel = '+ker+', Gamma = '+str(gam)
            print 'Precisions of CV = '+str(scores)
            print 'Mean Precision = '+str(scores.mean())+'\n'
print '\n########################################################\n'
print 'BEST LINEAR SVM (CV precision = '+str(best_linear*100)+' %)\n'
#Best Linear SVM
classifier = OneVsRestClassifier(LinearSVC(random_state=0, C=best_c_l, loss=best_loss, penalty=best_pen)).fit(X_train_PCA, Y_train)
Scores = classify_library.metric_scores(classifier, X_test_PCA, Y_test)
print "Settings: Linear SVM, C: %d, loss: %s, penalty: %s" % (c,lo,pen)
print "Scores in test: (%f, %f, %f, %f)\n" % (Scores[0], Scores[1], Scores[2], Scores[3])

print '\n########################################################\n'
print 'BEST NON-LINEAR SVM (CV precision = '+str(best_non_linear*100)+' %)\n'
#Best non-linear SVM
classifier = OneVsRestClassifier(svm.SVC(random_state=0, C=best_c_nl, kernel=best_ker, gamma=best_gam)).fit(X_train_PCA, Y_train)
Scores = classify_library.metric_scores(classifier, X_test_PCA, Y_test)
print "Settings: SVM kernel: %s, C: %d, gamma: %f" % (ker,c,gam)
print "Scores in test: (%f, %f, %f, %f)\n" % (Scores[0], Scores[1], Scores[2], Scores[3])

print '\n########################################################'
