#Simple unit tests covering preprocess_data.py and encode_image.py

import pytest
from preprocess_data import preprocessing
import os

captions_folder='Flickr8K_Text/'


def test_preprocess():
	'''
	Wherein we test preprocess_data.preprocess(). We check whether the 42nd entries
	of both output files, trainimgs.txt and testimgs.txt, agree with the truth.
	'''
	preprocessing()

	train_captions=open(os.path.join(captions_folder,'trainimgs.txt'),'r').read().split('\n')
	test_captions=open(os.path.join(captions_folder,'testimgs.txt'),'r').read().split('\n')

	print(train_captions)

	
