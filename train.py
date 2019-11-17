import SceneDesc
from keras.callbacks import ModelCheckpoint

import sys

def train(epoch,retrain=False,save=False):
    sd = SceneDesc.scenedesc()
    weightfile='Output/Weights.h5'
    if retrain:
        model=sd.create_model(ret_model=True)
        print('loading weights from '+weightfile)
        model.load_weights(weightfile)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    else:
        model = sd.create_model()
    batch_size = 512

    if save:
        callback=[ModelCheckpoint(filepath=weightfile,verbose=1,save_best_only=False,save_weights_only=True)]
    else:
        callback=None

    model.fit_generator(sd.data_process(batch_size=batch_size), steps_per_epoch=sd.no_samples/batch_size, epochs=epoch, verbose=2, callbacks=callback)
    model.save('Output/Model.h5', overwrite=True)
    model.save_weights('Output/Weights.h5',overwrite=True)
 
if __name__=="__main__":
    train(int(sys.argv[1]))
