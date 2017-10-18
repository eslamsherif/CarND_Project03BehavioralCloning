from keras.models import Sequential
from keras.layers import Lambda,Cropping2D, Dropout
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def Keras_NN(Traingenerator, Validgenerator, Variant, StepsperEpoch, ValidSteps, epcohCnt, preProcess = False):
    model = Sequential()
    if preProcess == True:
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
        model.add(Cropping2D(cropping=((70,25), (0,0))))
    else:
        #shape 160 * 320 * 3
        model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
        #shape 160 * 225 * 3
    model.add(Convolution2D(24,(5,5),strides=(2,2),activation="relu"))
    model.add(Convolution2D(36,(5,5),strides=(2,2),activation="relu"))
    model.add(Convolution2D(48,(5,5),strides=(2,2),activation="relu"))
    model.add(Convolution2D(64,(3,3),activation="relu"))
    model.add(Convolution2D(64,(3,3),activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(Traingenerator, validation_data=Validgenerator, validation_steps=ValidSteps, verbose=1, steps_per_epoch=StepsperEpoch, epochs=epcohCnt)

    if Variant == 1:
        model.save('model1.h5')
    elif Variant == 2:
        model.save('model2.h5')
    elif Variant == 3:
        model.save('model3.h5')
    elif Variant == 4:
        model.save('model4.h5')
    elif Variant == 5:
        model.save('model5.h5')
    elif Variant == 6:
        model.save('model6.h5')
    elif Variant == 7:
        model.save('model7.h5')