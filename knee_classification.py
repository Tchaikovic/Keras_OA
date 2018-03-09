#CNN for Osteoarthritis classification
import os
import numpy 
import pandas 
import matplotlib.pyplot as plt

from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential  
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD, Adam, Adagrad
from keras.models import model_from_json  
from keras.utils import np_utils, generic_utils

from keras.preprocessing.sequence import pad_sequences
#from IPython.display import SVG
#from keras.utils.visualize_util import model_to_dot
from keras import metrics

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
numpy.random.seed(7)

# image specification
img_width, img_height = 150, 150

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])

batch_size = 32

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data_PA/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data_PA/validation',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

history = model.fit_generator(
          train_generator,
          steps_per_epoch=2000 // batch_size,
          epochs=200,
          validation_data=validation_generator,
          validation_steps=800 // batch_size)

model.save_weights('first_try_PA_binary.h5')  # always save your weights after training or during training

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

## Various solver param initialisation
#sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#adagrad = Adagrad(lr=0.0001, epsilon=1e-08, decay=0.0)


#X_train_new, y_train_new=  train_test_split(X_train, y_train,random_state=2)

#model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae', 'acc'])
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['mae', 'acc'])
#model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['acc', metrics.mae])
#model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['acc', 'fmeasure', 'precision', 'recall', 'mae'])
#SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
#model.summary() 
    
