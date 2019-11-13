#Simple unit tests covering preprocess_data.py and encode_image.py

import pytest
from preprocess_data import preprocessing
import os

captions_folder='Flickr8K_Text/'


def test_preprocess():
	'''
	Wherein we test preprocess_data.preprocess(). We check whether the number of 
	test/training examples as well as the 42nd entries of both output files, 
	trainimgs.txt and testimgs.txt, agree with the truth.
	'''
	preprocessing()

	train_captions=open(os.path.join(captions_folder,'trainimgs.txt'),'r').read().split('\n')
	test_captions=open(os.path.join(captions_folder,'testimgs.txt'),'r').read().split('\n')

	assert len(train_captions)==30001#the last line yields one stray empty string
	assert len(test_captions)==5001

	assert train_captions[42]=='2638369467_8fc251595b.jpg\t<start> A smiling young girl in braids is playing ball . <end>'
	assert test_captions[42]=='280706862_14c30d734a.jpg\t<start> A black dog on a beach carrying a ball in its mouth . <end>'
	
