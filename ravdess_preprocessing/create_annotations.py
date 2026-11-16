# -*- coding: utf-8 -*-

import os
import re
# Data folder: RAVDESS_ROOT env var, otherwise a RAVDESS folder next to the repository.
root = os.path.abspath(os.environ.get('RAVDESS_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'RAVDESS')))

n_folds=1
folds = [[[0,1,2,3],[4,5,6,7],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]]
for fold in range(n_folds):
        fold_ids = folds[fold]
        test_ids, val_ids, train_ids = fold_ids
	
        #annotation_file = 'annotations_croppad_fold'+str(fold+1)+'.txt'
        annotation_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotations.txt')
        open(annotation_file, 'w').close()  # start fresh instead of appending duplicates
	
        actors = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))  # skip .DS_Store etc.
        for actor in actors:
            i = int(re.search(r'\d+', actor).group()) - 1  # split by actor number (Actor_01 -> 0), not folder order
            for video in os.listdir(os.path.join(root, actor)):
                if not video.endswith('.npy') or 'croppad' not in video:
                    continue
                label = str(int(video.split('-')[2]))
		     
                audio = '03' + video.split('_face')[0][2:] + '_croppad.wav'  
                if i in train_ids:
                   with open(annotation_file, 'a') as f:
                       f.write(os.path.join(root,actor, video) + ';' + os.path.join(root,actor, audio) + ';' + label + ';training' + '\n')
		

                elif i in val_ids:
                    with open(annotation_file, 'a') as f:
                        f.write(os.path.join(root, actor, video) + ';' + os.path.join(root,actor, audio) + ';'+ label + ';validation' + '\n')
		
                else:
                    with open(annotation_file, 'a') as f:
                        f.write(os.path.join(root, actor, video) + ';' + os.path.join(root,actor, audio) + ';'+ label + ';testing' + '\n')
		

