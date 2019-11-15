import pytest
from tensorflow.python.client import device_lib
import keras.backend as K
import os, subprocess, atexit

from test_mod import process_caption,generate_captions
from test import text
from train import train
from SceneDesc import scenedesc
import encode_image as ei

	
test_img=os.path.join('Flickr8K_Data','997722733_0cb5439472.jpg')
sd=scenedesc()


@pytest.fixture(scope='module')
def preload_model(request):
	try:
		model=sd.create_model(ret_model=True)
		model.load_weights(os.path.join('Output','Weights.h5'))
		preload_model=True
		print('pre-computed weights loaded; model created')

	except IOError:
		print('no pre-computed weights found; skipping testing tests')
		preload_model=False
	return model

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

@pytest.mark.needs_weights
@pytest.mark.skipif(not preload_model, reason='no pre-computed weights found')
def test_generate_captions(preload_model):
	'''
	Wherein we test test_mod.generate_captions. Since you may
	use pre-computed weights from any source only print the 
	generated sentence to stdout and check that it is nonempty
	'''
	model=preloaded_model
	encoded_img=ei.encodings(ei.model_gen(),test_img)
	caption=generate_captions(sd,model,encoded_img,beam_size=3)
	def report():
		print('The model generated the caption: '+caption)
	atexit.register(report)
	assert(len(caption)>0)

@pytest.mark.needs_weights
@pytest.mark.skipif(not preload_model, reason='no pre-computed weights found')
def test_text_creation(preload_model):
	'''
	Wherein we test the caption generation wrapper test.text.
	'''
	model=preload_model
	text(test_img)
	def report():
		print('The model generated a caption that you should hear (and read, if you run pytest with the -s flag). This caption should be the same as the caption you can read below, and you should also have heard it read out.\n')
	atexit.register(report)
	pass

@pytest.mark.skipif(not has_gpu, reason='no gpu visible to keras')
def test_training():
	'''
	Wherein we test whether we can run the training pipeline
	for a single epoch. We afterwards check that the model and
	weights files have been written to file and are nonzero.
	If pre-computed model/weights files are present, they will
	be copied to a temporary location to save them from being
	wantonly and relentlessly overwritten.
	'''

	#First, we make sure that we do not overwrite our pre-computed model and weights

	if os.path.exists(os.path.join('Output','Model.h5')):
		try:
			subprocess.call(['mv',os.path.join('Output','Model.h5'),os.path.join('Output','Model.h5.swp')])
			swapped_model=True
		except:
			print('Found pre-computed model, but could not copy it in temporary location. Aborting test.')
			proceed=False #model present, but copying failed
			swapped_model=False
	else:
		proceed=True #there is no model to overwrite
		swapped_model=False

	if os.path.exists(os.path.join('Output','Weights.h5')):
		try:
			subprocess.call(['mv',os.path.join('Output','Weights.h5'),os.path.join('Output','Weights.h5.swp')])
			swapped_weights=True
		except:
			print('Found pre-computed weights, but could not copy them in temporary location. Aborting test.')
			proceed=False #weights present, but copying failed
			swapped_weights=False
	else:
		proceed=True #there are no weights to overwrite
		swapped_weights=False

	#train for one epoch; check that new model and weight files exist and are nonempty
	if proceed:
		train(1)

		assert(os.path.exists('Output','Model.h5'))
		assert(os.path.exists('Output','Weights.h5'))
		assert(os.path.getsize(os.path.join('Output','Model.h5'))>0)
		assert(os.path.getsize(os.path.join('Output','Weights.h5'))>0)
	
	#swapping back model and weights from their temporary locations
	if swapped_model:
		try:
			subprocess.call(['mv',os.path.join('Output','Model.h5.swp'),os.path.join('Output','Model.h5')])
		except:
			print('Could not copy back model file! Possibly it has been overwritten...')

	if swapped_weights:
		try:
			subprocess.call(['mv',os.path.join('Output','Weights.h5.swp'),os.path.join('Output','Weights.h5')])
		except:
			print('Could not copy back weights file! Possibly it has been overwritten...')

