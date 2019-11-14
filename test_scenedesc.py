import pytest
import numpy as np
import os

from SceneDesc import scenedesc

test_image_id=4242
image_folder='Flickr8K_Data/'
true_enc=np.array([0,0.41896123,0,0,0.8073355,0,0,0,0,0,0,0,3.6969965,0,0,0,1.966795,0,0,0,2.4138303,0,0,0,0,0.540326,5.7941747,0,0,0,1.9354384,0,0,0,0,0,0,0,0,0.2696853,0,0,0.33549678,0,1.3821363,0,0,0,0,0,1.5806007,0,0,0,0,0,0,0,0,0,0,0,1.1648078,0,0,0,0,0,0,0,0,0,1.9297309,2.7865145,0,0,0,0.24934426,3.30201,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
true_img_slice=np.array([[105., 166., 148.,  55.,  69., 117., 136., 150., 191., 209.],
       [105., 174., 183., 115.,  73.,  87., 127., 142., 147., 207.],
       [ 77., 162., 186., 127.,  86.,  82., 117., 129., 148., 200.],
       [ 56., 146., 167.,  76.,  73.,  99., 184., 170., 198., 190.],
       [ 59., 138., 160.,  73.,  76., 143., 218., 178., 193., 177.],
       [ 54., 135., 170.,  71.,  77.,  96., 160., 165., 196., 177.],
       [ 78., 116., 165.,  82.,  73.,  82., 104., 174., 207., 189.],
       [161.,  94., 149.,  87.,  61.,  63.,  62., 158., 193., 201.],
       [156., 130., 128.,  91.,  76.,  43.,  95., 158., 188., 215.],
       [130., 143., 149.,  74.,  73.,  64., 178., 167., 152., 207.]],)
true_X_slice=np.array([[0,0,1.3217616,0,2.9636335,0,5.2782164,0,0,0,0,0,0,1.5424823,0,0,0,2.9206674,0,2.2342544,2.7055874,0.9519915,3.3693693,1.601603,0,0,2.686882,0,0,0.8407563,0,5.28025,0,0,0,0,1.285548,3.7856739,0.40175304,4.5349374,0.26478094,0.89722455,0.03038335,0,3.1030846,0,0.43142515,0.7068827,0,2.1046154],[0,0,1.3217616,0,2.9636335,0,5.2782164,0,0,0,0,0,0,1.5424823,0,0,0,2.9206674,0,2.2342544,2.7055874,0.9519915,3.3693693,1.601603,0,0,2.686882,0,0,0.8407563,0,5.28025,0,0,0,0,1.285548,3.7856739,0.40175304,4.5349374,0.26478094,0.89722455,0.03038335,0,3.1030846,0,0.43142515,0.7068827,0,2.1046154]])

@pytest.fixture(scope='module')
def sd(request):
	return scenedesc()

@pytest.mark.slow
def test_constructor(sd):
	'''
	Wherein we instantiate one object of scenedesc and check
	that it has the correct properties
	'''
	#check sd.captions
	assert(len(sd.captions)==29999)
	assert(sd.captions[test_image_id]=='<start> The bald woman is standing smiling next to a frowning man . <end>')

	#check sd.img_id
	assert(len(sd.img_id)==29999)
	assert(sd.img_id[test_image_id]=='3110174991_a4b05f8a46.jpg')

	#check sd.image_encodings
	assert(len(sd.image_encodings))
	test_enc=sd.image_encodings[sd.img_id[test_image_id]]
	assert(len(test_enc[test_enc!=0])==961)
	np.testing.assert_allclose(test_enc[:100],true_enc)

	#check sd.no_samples, sd.vocab_size, sd.max_length
	assert(sd.no_samples==383440)
	assert(sd.vocab_size==8256)
	assert(sd.max_length==40)

	#check some properties of sd.word_index and sd.index_word:
	#length and the most relevant entries
	assert(len(sd.word_index)==8256)
	assert(len(sd.index_word)==8256)

	assert(sd.word_index['Cat']==7903)
	assert(sd.word_index['cat']==2666)
	assert(sd.word_index['kitten']==2592)

	#check sd.index_word is the inverse of sd.word_index
	for idx in sd.index_word.keys():
		assert(idx==sd.word_index[sd.index_word[idx]])

	for word in sd.word_index.keys():
		assert(word==sd.index_word[sd.word_index[word]])

def test_train_generator(sd):
	'''
	Wherein we test the data generating process. We initialise it with
	a batch size of 2 and check the first element in the usual way:
	correct shape, correct number of nonvanishing elements, checking
	a few (or all) of those elements.
	'''

	datagen=sd.data_process(batch_size=2)
	d=next(datagen)

	#correct shape
	assert(len(d)==len(d[0])==2)
	assert(d[0][0].shape==(2,4096))
	assert(d[0][1].shape==(2,40))
	assert(d[1].shape==(2,8256))

	#correct number of entries	
	assert(np.count_nonzero(d[0][0])==2290)
	assert(np.count_nonzero(d[0][1])==3)
	assert(np.count_nonzero(d[1])==2)

	#checking entries

	d01_nonvanishing=np.array([d[0][1][0,0],d[0][1][1,0],d[0][1][1,1]])
	np.testing.assert_allclose(d01_nonvanishing,np.array([4012,4012,4798]))
	d1_nonvanishing=np.array([d[1][0,4798],d[1][1,4332]])
	np.testing.assert_allclose(d1_nonvanishing,np.ones(2))

def test_model_structure():
	pass

def test_model_evaluation():
	pass

def test_training():
	'''
	Wherein we test whether we can set up and run a training generator:
	we only train on one picture for one step, and check that the 
	trainable parameters have changed.
	'''
	pass	


def test_load_image(sd):
	'''
	Wherein we test the wrapper load_image by checking one example
	'''
	img=sd.load_image(os.path.join(image_folder, sd.img_id[666]))
	assert(img.shape==(224,224,3))
	np.testing.assert_allclose(img[:10,:10,0],true_img_slice)

def test_get_word(sd):
	'''
	Wherein we test the wrapper get_word by checking one example
	'''

	assert(sd.get_word(666)=='quarter')

