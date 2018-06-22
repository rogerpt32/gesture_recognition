import classify_library
import argparse
import IDT_feature
import computeFV
import numpy as np
import subprocess, os
import sys, ffmpeg
import cv2

video = 'roger_wave1.avi'
video_dir = '../data/camera_test/videos/'
tmp_dir = './tmp/'
dtBin = './iDT-master/release/DenseTrackStab'
gmm_list = '../data/fishers/gmm_list'
class_index_f = '../data/class_index.npz'
BOLD = '\033[1m'
OKGREEN = '\033[92m'
ENDC = '\033[0m'

def check_resolution(vid):
    vcap = cv2.VideoCapture(vid) # 0=camera
    if vcap.isOpened():
        # get vcap property
        width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
        return width==320 and height==240 # 320x240 resolution is standard in this project
    else:
        return False
# To test the prediction for any set of videos using the saved classifier
# Usage: python test_predict.py video_dir
# videos name must be: nameidentifier_class.avi

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicts the class of a single video.')
    parser.add_argument("video_dir", help="Name of the input directory desired to test prediction", type=str)
    parser.add_argument('--no_pca', dest='no_pca', action='store_const',
                   const=True, default=False,
                   help='Testing without PCA reduction (default: uses PCA)')
    args = parser.parse_args()
    video_dir = args.video_dir

    index_class = np.load(class_index_f)['index_class'][()]
    class_index = np.load(class_index_f)['class_index'][()]

    gmm_list = np.load(gmm_list+".npz")['gmm_list']

    if not args.no_pca:
        pca=classify_library.load_model('../data/models/pca.sav')
        svm=classify_library.load_model('../data/models/svm.sav')
    else:
        svm=classify_library.load_model('../data/models/svm_nopca.sav')

    confusion_mat = np.zeros((len(index_class),len(index_class)))
    vids = os.listdir(video_dir)

    hits=0
    attempts=0

    for video in vids:
        if video.split('.')[1]!='avi':
            continue
        vid = os.path.join(video_dir,video)
        action = video.split('_')[1].split('.')[0]
        if not check_resolution(vid):
            resizedName = os.path.join(tmp_dir,video)
            if ffmpeg.resize(vid,resizedName):
                vid = resizedName
        command = dtBin + ' ' + vid
        p = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE)
        raw_features,_ = p.communicate()
        features = raw_features.split("\n")
        points=[]
        for point in features:
            if point!='' and point[0]!='[': #to delete info messages from DenseTrackStab
                points.append(IDT_feature.IDTFeature(point))
        video_desc = IDT_feature.vid_descriptors(points)
        fish = computeFV.create_fisher_vector_unsaved(gmm_list, video_desc)
        fish = np.array(fish).reshape(1, -1)
        if args.no_pca:
            fish_pca=fish
        else:
            fish_pca = pca.transform(fish)
        result = svm.predict(fish_pca)
        confusion_mat[class_index[action]][result[0]]+=1
        if class_index[action]==result[0]:
            hits+=1
        attempts+=1
        # print '\n' +'RESULT of '+ video +': ' + OKGREEN + BOLD + index_class[result[0]] + ENDC + '\n'
    print 'Order of classes in confusion matrix:\n'
    for k,v in index_class.iteritems():
        print k,v
    print '\nConfusion Matrix:\n'
    print confusion_mat
    print '\nPrecision: '+str((float(hits)/float(attempts))*100)+' %'
