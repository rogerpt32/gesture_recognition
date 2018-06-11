import classify_library
import computeIDTF
import numpy as np
import subprocess, os
import sys, ffmpeg

video = 'roger_wave2.avi'
video_dir = '../data/camera_test/videos/'
tmp_dir = './tmp/'
dtBin = './iDT-master/release/DenseTrackStab'
gmm_list = '../data/fishers/gmm_list'

vid = os.path.join(video_dir,video)
out = os.path.join(tmp_dir, video.split('.')[0]+".fisher")
computeIDTF.extractFV(vid, out, gmm_list)
pca=classify_library.load_model('../data/models/pca.sav')
svm=classify_library.load_model('../data/models/svm.sav')

x = np.load(out+'.npz')['fish']
x=np.array(x).reshape(1, -1)
x_pca = pca.transform(x)
print svm.predict(x_pca)

subprocess.call('rm -rf '+tmp_dir+'*',shell=True)
