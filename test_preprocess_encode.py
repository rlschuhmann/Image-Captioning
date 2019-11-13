#Simple unit tests covering preprocess_data.py, encode_image.py, vgg16.py and imagenet_utils.py

import pytest

from preprocess_data import preprocessing
from vgg16 import VGG16

import os
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

captions_folder='Flickr8K_Text/'
image_folder='Flickr8K_Data/'
test_image='3666056567_661e25f54c.jpg'
vgg16_pred_slice=np.array([0,0,0,0,6.281284,0,0,0,0,0.3895438,0,0,1.1056916,0,0,1.7175925,0,1.8917649,0,0,1.7590455,6.602196,0.43233678,0.4528695,0,3.6373415,3.5424447,2.4427955,0,0,3.206093,2.5562162,0,0,2.2278,0,3.2318673,2.3519747,0,6.4740357,0,2.8249745,0,0,9.216705,0,0,0,2.835004,2.1258173,2.2908218,1.0856905,2.6308882,0.91681635,0,0,0,0,0,0,0.40301943,0,0,0,3.2203531,0.2024988,0,0,0,0,0,0,1.2414606,0,0,0,1.2453537,0,4.912657,0,4.361135,0,0,0,0,0,0,0,2.4128122,0.68293965,0.7010477,1.3694913,1.7609701,0,0,0,0,0,0,0,])


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


def test_vgg16():
	'''
	Wherein we test vgg16.py. We create the model, load a test image, perform the
	embedding, and test the result for correct shape, correct number of nonzero
	entries, correct first 100 entries.
	'''

	model=VGG16(include_top=True,weights='imagenet')

	img=image.load_img(os.path.join(image_folder,test_image),target_size=(224,224))
	x=image.img_to_array(img)
	x=np.expand_dims(x,axis=0)
	x=preprocess_input(x)

	preds=model.predict(x)
	
	assert(x.shape==(1,224,224,3))
	assert(preds.shape==(1,4096))
	assert(len(preds[0][preds[0]!=0])==1368)
	np.testing.assert_allclose(preds[0,:100],vgg16_pred_slice,err_msg='VGG16 embedding yielded wrong embedding for test image!')

