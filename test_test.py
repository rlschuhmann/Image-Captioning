import pytest
from tensorflow.python.client import device_lib
import keras.backend as K

from test_mod import process_caption,generate_captions
from test import text
from SceneDesc import scenedesc
import encode_image as ei

	
test_img='Flickr8K_Data/997722733_0cb5439472.jpg'
sd=scenedesc()

try:
	model=sd.create_model(ret_model=True)
	model.load_weights('Output/Weights.h5')
	preload_model=True
	print('pre-computed weights loaded; model created')

except IOError:
	print('no pre-computed weights found; skipping testing tests')
	preload_model=False


try:
	tf_available_devices=str(device_lib.list_local_devices())
	k_gpus=K.tensorflow_backend._get_available_gpus()	

	if 'GPU' in tf_available_devices and len(k_gpus)>0:
		print('GPUs visible to keras!')
		has_gpu=True
	else:
		print('No GPUs visible to keras - skipping training tests!')
		has_gpu=False
		
except AttributeError:
	print('Could not detect whether any GPU is visible to keras - most likely due to an outdated version of keras. Skipping training tests!')
	has_gpu=False


def test_process_caption():
	'''
	Wherein we test the post-processing step in test_mod.py 
	that removes start and end tokens.
	'''
	capt=process_caption(sd,'<start> A black dog is running after a white dog in the snow . <end>')
	assert(capt=='A black dog is running after a white dog in the snow .')

@pytest.mark.skipif(not preload_model, reason='no pre-computed weights found')
def test_generate_captions():
	'''
	Wherein we test test_mod.generate_captions. Since you may
	use pre-computed weights from any source only print the 
	generated sentence to stdout and check that it is nonempty
	'''

	encoded_img=ei.encodings(ei.model_gen(),test_img)
	caption=generate_captions(sd,model,encoded_img,beam_size=3)
	print('The model generated the caption: '+caption)
	assert(len(caption)>0)

@pytest.mark.skipif(not preload_model, reason='no pre-computed weights found')
def test_text_creation():
	'''
	Wherein we test the caption generation wrapper test.text.
	'''
	print('\nThe text creation wrapper generated the caption: ')
	text(test_img)
	print('This should be the same as the caption generated above, and you should also have heard it read out.\n')
	pass

@pytest.mark.skipif(not has_gpu, reason='no gpu visible to keras')
def test_training():
	pass
