import classify_library
import computeIDTF
import numpy as np
import subprocess, os
import sys, ffmpeg

video = 'roger_wave1.avi'
video_dir = '../data/camera_test/videos/'
tmp_dir = './tmp/'
gmm_list = '../data/fishers/gmm_list'
class_index = '../data/class_index.npz'
BOLD = '\033[1m'
OKGREEN = '\033[92m'
ENDC = '\033[0m'

# To test the prediction from the saved classifier with videos from data/camera_test/videos/
# Usage: python test_predict.py filename.avi

if __name__ == '__main__':
    video = sys.argv[1]

    vid = os.path.join(video_dir,video)
    out = os.path.join(tmp_dir, video.split('.')[0]+".fisher")
    computeIDTF.extractFV(vid, out, gmm_list)
    pca=classify_library.load_model('../data/models/pca.sav')
    svm=classify_library.load_model('../data/models/svm.sav')

    x = np.load(out+'.npz')['fish']
    x=np.array(x).reshape(1, -1)
    x_pca = pca.transform(x)
    result = svm.predict(x_pca)
    index_class = np.load(class_index)['index_class']
    print '\n' +'RESULT: ' + OKGREEN + BOLD + index_class[()][result[0]] + ENDC + '\n'

    subprocess.call('rm -rf '+tmp_dir+'*',shell=True)
