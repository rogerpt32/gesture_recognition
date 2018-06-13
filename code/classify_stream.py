import subprocess, os, sys, collections
import numpy as np
from yael import ynumpy
import IDT_feature
from tempfile import TemporaryFile
import argparse
import computeFV
import classify_library
import computeIDTF
import ffmpeg


"""
classify every 50 frames of a given input stream of IDTFs

Usage:
	stream_of_IDTFs | python classify_stream.py
"""

tmp_dir = './tmp/'
gmm_list = '../data/fishers/gmm_list'
class_index = '../data/class_index.npz'

frame_step = 50

BOLD = '\033[1m'
OKGREEN = '\033[92m'
ENDC = '\033[0m'

#The input is a stream of IDTFs associated with a recording camera.
if __name__ == '__main__':
   #loading everything necessary
   pca=classify_library.load_model('../data/models/pca.sav')
   svm=classify_library.load_model('../data/models/svm.sav')
   gmm_list = np.load(gmm_list+".npz")['gmm_list']
   index_class = np.load(class_index)['index_class']
   index_class = index_class[()]

   points = [] # a list of IDT features.
   frame_lim = frame_step
   for line in sys.stdin:
      if line[0]!='[': # avoid getting info message as data
         frame = int(line.split()[0])
         if frame_lim <= frame:
            frame_lim=frame_lim+frame_step
            # print frame_lim<=frame
            if points!=[]:
               video_desc = IDT_feature.vid_descriptors(points)
               fish = computeFV.create_fisher_vector_unsaved(gmm_list, video_desc)
               fish=np.array(fish).reshape(1, -1)
               fish_pca = pca.transform(fish)
               result = svm.predict(fish_pca)

               print '\n' + 'RESULT: ' + OKGREEN + BOLD + index_class[result[0]] + ENDC + '\n'

            points = []
         points.append(IDT_feature.IDTFeature(line))
