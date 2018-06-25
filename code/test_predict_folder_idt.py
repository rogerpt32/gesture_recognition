import os, sys, collections, random, string, subprocess
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

import gmm, computeFV, IDT_feature
import classify_library

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing different classifiers and print and analysis.')
    parser.add_argument('--no_pca', dest='no_pca', action='store_const',
                   const=True, default=False,
                   help='Testing without PCA reduction (default: uses PCA)')
    args = parser.parse_args()

    tmp_dir = "./tmp_dir/"
    dtBin = "./iDT-master/release/DenseTrackStab"
    video_input = "../data/camera_test/videos"

    class_index_file = "../data/class_index.npz"
    class_index_file_loaded = np.load(class_index_file)
    class_index = class_index_file_loaded['class_index'][()]
    index_class = class_index_file_loaded['index_class'][()]

    if not args.no_pca:
        pca=classify_library.load_model('../data/models/pca.sav')
        svm=classify_library.load_model('../data/models/svm.sav')
    else:
        svm=classify_library.load_model('../data/models/svm_nopca.sav')

    traj_length = 15
    density = 6
    wind = 32
    spatial = 2
    temporal = 3


    # In[7]:
    if os.path.isdir(tmp_dir):
        subprocess.call('rm -rf %s' % (os.path.join(tmp_dir,'*')),shell=True)
    else:
        subprocess.call('mkdir %s' % tmp_dir,shell=True)
    videos = [filename for filename in os.listdir(video_input) if filename.endswith('.avi')]
    features = []
    for vid in videos:
        outputBase=os.path.join(tmp_dir,vid[:-4]+'.features')
        features.append(vid[:-4]+'.features')
        subprocess.call('%s %s -L %d -W %d -N %d -s %d -t %d > %s' % (dtBin, os.path.join(video_input, vid), traj_length, density, wind, spatial, temporal, outputBase), shell=True)
    gmm_list = gmm.populate_gmms(tmp_dir,features,os.path.join(tmp_dir,'gmm_list'),120)
    x_total = []
    y_total = []
    for feat in features:
        points=[]
        f=open(os.path.join(tmp_dir,feat),'r')
        for line in f:
            if line[0]!='[':
                points.append(IDT_feature.IDTFeature(line))
        video_desc = IDT_feature.vid_descriptors(points)
        fish = computeFV.create_fisher_vector_unsaved(gmm_list,video_desc)
        action = feat.split('_')[-1].split('.')[0]
        x_total.append(fish)
        y_total.append(class_index[action])

    if not args.no_pca:
        #PCA reduction
        X_total_final = pca.transform(x_total)
    else:
        X_total_final = x_total
    Y_pred = svm.predict(X_total_final)
    conf_mat = metrics.confusion_matrix(y_total,Y_pred)
    hits = len([i for i, j in zip(y_total, Y_pred) if i == j])
    att = len(y_total)
    print '\nPrecision: '+str((float(hits)/float(att))*100)+' %'
    print 'Confusion Matrix = '
    print conf_mat
    print '\n-----------------------------------------------------\n'
