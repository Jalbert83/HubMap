import segmentation_models as sm
from segmentation_models import Unet
from tensorflow.keras import backend as K
from keras import utils as np_utils
import keras
from keras import utils as np_utils

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataSequence import SliceSequence, ImSequence



sm.set_framework('tf.keras')
sm.framework()

model = Unet(backbone_name='efficientnetb5', encoder_weights='imagenet', encoder_freeze=True)
model.compile("Adam", "binary_crossentropy", [sm.metrics.FScore()])

batch_size = 4
# size of the input square image for the model
img_size = 512
# Parameter used for an intermediate resize for SliceSequence
img_Resize = 0

epochs = 30

df = pd.read_csv('TrainSliced.csv')

# Filter the data related to the organ with which is going to be tested

#df = df[df.organ == 'spleen']

# Number of samples taken for validation

NVal = int(np.floor(len(df) * 0.2))

dfshuffle = df.iloc[np.random.permutation(len(df))]

dv = dfshuffle.iloc[-NVal:]
dt = dfshuffle.iloc[:-NVal]

## ImSequence with a resize of each image to img_size

#Train = ImSequence(batch_size, img_size, dt)
#Val = ImSequence(batch_size, img_size, dv)

## SliceSequence resizes each image to img_Resize and takes slices as stored by the script Dfslices in the Dataframe

Train = SliceSequence(batch_size, img_size, dt)
Val = SliceSequence(batch_size, img_size, dv)


call = keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min')

history = model.fit(Train, epochs=epochs, callbacks = call, validation_data=Val)


fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(10, 5))
ax0.set_title("Train Loss")
ax0.plot(epochs, history.history["loss"])
ax1.set_title("Val Loss")
ax1.plot(epochs, history.history["val_loss"])
plt.show()