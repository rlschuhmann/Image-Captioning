import pytest
import numpy as np

from SceneDesc import scenedesc


test_image_id=4242
true_enc=np.array([0,0.41896123,0,0,0.8073355,0,0,0,0,0,0,0,3.6969965,0,0,0,1.966795,0,0,0,2.4138303,0,0,0,0,0.540326,5.7941747,0,0,0,1.9354384,0,0,0,0,0,0,0,0,0.2696853,0,0,0.33549678,0,1.3821363,0,0,0,0,0,1.5806007,0,0,0,0,0,0,0,0,0,0,0,1.1648078,0,0,0,0,0,0,0,0,0,1.9297309,2.7865145,0,0,0,0.24934426,3.30201,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])


@pytest.mark.slow
def test_constructor():
	'''
	Wherein we instantiate one object of scenedesc and check
	that it has the correct properties
	'''
	sd=scenedesc()

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
