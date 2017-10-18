import csv
import cv2
import numpy as np
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

from HelperFncs import *
from KerasModel import *

inputCSVName = '.\..\SimDataOut\driving_log.csv'
inputTestImg = ['.\..\SimDataOut\IMG\center_2017_10_16_21_35_19_735.jpg',
                '.\..\SimDataOut\IMG\center_2017_10_16_21_36_17_180.jpg',
                '.\..\SimDataOut\IMG\center_2017_10_16_21_36_37_815.jpg',
                '.\..\SimDataOut\IMG\center_2017_10_16_21_36_54_122.jpg',
                '.\..\SimDataOut\IMG\center_2017_10_16_21_36_54_122.jpg',
                '.\..\SimDataOut\IMG\center_2017_10_16_21_37_13_663.jpg']

def ReadCSV(fileName):
    lines = []
    with open(fileName) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    
    return lines

def ProcessCSVFile(lines):
    imgspaths   = []
    steeringang = []

    for line in lines:
        #load Center Image
        imgspaths.append((line[0], False))
        steeringang.append(float(line[3]))
    
        imgspaths.append((line[1], False))
        steeringang.append(float(line[3])+0.2)
    
        imgspaths.append((line[2], False))
        steeringang.append(float(line[3])-0.2)
    
        imgspaths.append((line[0], True))
        steeringang.append(-float(line[3]))
    
    return (imgspaths, steeringang)

#Preprocess Frames to increase accuracy
def PreProcessImage1(img):
    return img

def PreProcessImage2(img):
    #sharpen the frame to get more clear edges
    Sharpned = Sharpen(img, KernelSize=3)
    return Sharpned

def PreProcessImage3(img):
    #sharpen the frame to get more clear edges
    Sharpned = Sharpen(img, KernelSize=7)
    return Sharpned

def PreProcessImage4(img):
    #sharpen the frame to get more clear edges
    Sharpned = Sharpen(img, KernelSize=13)
    return Sharpned

def PreProcessImage5(img):
    #sharpen the frame to get more clear edges
    Sharpned = Sharpen(img, KernelSize=7)
    #convert to Lab color space for less susptibility for lighting conditions,
    # also it is useful for marking pure yellow lanes which the network can use
    LABForm  = LABColorFormat(Sharpned)
    return LABForm

def PreProcessImage6(img):
    #sharpen the frame to get more clear edges
    Sharpned = Sharpen(img, KernelSize=7)
    #convert to Lab color space for less susptibility for lighting conditions,
    # also it is useful for marking pure yellow lanes which the network can use
    LABForm  = LABColorFormat(Sharpned)
    #perform adaptive histogram equalization to uniform brightness across image
    clahe    = HistogramEqualization2(LABForm, seperateChannels = True)
    return clahe

def PreProcessImage7(img):
    #convert to Lab color space for less susptibility for lighting conditions,
    # also it is useful for marking pure yellow lanes which the network can use
    LABForm  = LABColorFormat(img)
    return LABForm

def PreProcessVariant(image, Variant):
    if Variant == 1:
        return PreProcessImage1(image)
    elif Variant == 2:
        return PreProcessImage2(image)
    elif Variant == 3:
        return PreProcessImage3(image)
    elif Variant == 4:
        return PreProcessImage4(image)
    elif Variant == 5:
        return PreProcessImage5(image)
    elif Variant == 6:
        return PreProcessImage6(image)
    elif Variant == 7:
        return PreProcessImage7(image)

def PopulateTrainingBatch(ImagePaths, StrAng, BatchSize, Variant):
    assert len(ImagePaths) == len(StrAng)
    assert BatchSize < len(ImagePaths)
    
    while(True):
        images = []
        labels = []
        
        ImagePaths, StrAng = Shuffle(ImagePaths, StrAng)
        
        for i in range(len(ImagePaths)):
            img, Flip = ImagePaths[i]
            image = mpimg.imread(img)
            if Flip == True:
                image = np.fliplr(image)
            images.append(PreProcessVariant(image, Variant))
            labels.append(StrAng[i])
            if len(images) == BatchSize:
                yield(np.array(images), np.array(labels))
                images = []
                labels = []
                ImagePaths, StrAng = Shuffle(ImagePaths, StrAng)
    

def TestPreProcess():
    TestImages = []
    for im in inputTestImg:
        TestImages.append(PreProcessImage(mpimg.imread(im)))

    dummy = np.zeros(len(TestImages))
    pltImages(TestImages, dummy, nrows = 2, ncols = 3)

#TestPreProcess()

CSVLines = ReadCSV(inputCSVName)

ImagePaths, StrAng = ProcessCSVFile(CSVLines)

#inspired from http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(ImagePaths, StrAng, test_size=0.2, random_state=42)

print("Length of Training Set: " + str(len(X_train)))
print("Length of Validation Set: " + str(len(X_test)))

TrainBatchSize = 256
ValidBatchSize = 64
epochs = 5

Traingenerator1 = PopulateTrainingBatch(X_train, y_train, TrainBatchSize, 1)
Validgenerator1 = PopulateTrainingBatch(X_test, y_test, ValidBatchSize, 1)

Traingenerator2 = PopulateTrainingBatch(X_train, y_train, TrainBatchSize, 2)
Validgenerator2 = PopulateTrainingBatch(X_test, y_test, ValidBatchSize, 2)

Traingenerator3 = PopulateTrainingBatch(X_train, y_train, TrainBatchSize, 3)
Validgenerator3 = PopulateTrainingBatch(X_test, y_test, ValidBatchSize, 3)

Traingenerator4 = PopulateTrainingBatch(X_train, y_train, TrainBatchSize, 4)
Validgenerator4 = PopulateTrainingBatch(X_test, y_test, ValidBatchSize, 4)

Traingenerator5 = PopulateTrainingBatch(X_train, y_train, TrainBatchSize, 5)
Validgenerator5 = PopulateTrainingBatch(X_test, y_test, ValidBatchSize, 5)

Traingenerator6 = PopulateTrainingBatch(X_train, y_train, TrainBatchSize, 6)
Validgenerator6 = PopulateTrainingBatch(X_test, y_test, ValidBatchSize, 6)

Traingenerator7 = PopulateTrainingBatch(X_train, y_train, TrainBatchSize, 7)
Validgenerator7 = PopulateTrainingBatch(X_test, y_test, ValidBatchSize, 7)

StepsPerEpoch = int(len(X_train) / TrainBatchSize)
ValidSteps = int(len(X_test) / ValidBatchSize)

Keras_NN(Traingenerator1, Validgenerator1, 1, StepsPerEpoch, ValidSteps, epochs, True)
Keras_NN(Traingenerator2, Validgenerator2, 2, StepsPerEpoch, ValidSteps, epochs)
#Keras_NN(Traingenerator3, Validgenerator3, 3, StepsPerEpoch, ValidSteps, epochs)
#Keras_NN(Traingenerator4, Validgenerator4, 4, StepsPerEpoch, ValidSteps, epochs)
Keras_NN(Traingenerator5, Validgenerator5, 5, StepsPerEpoch, ValidSteps, epochs)
Keras_NN(Traingenerator6, Validgenerator6, 6, StepsPerEpoch, ValidSteps, epochs)
Keras_NN(Traingenerator7, Validgenerator7, 7, StepsPerEpoch, ValidSteps, epochs)