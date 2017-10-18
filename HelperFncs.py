import cv2
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle

#Color Space Conversion Functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def LABColorFormat(img):
    # LAB is chosen for it's robustness aganist changing lighting conditions.
    # This is due to Lightness being a seperate parameter 'L'.
    # LAB is also useful because the maximum B values represent pure yellow so yellow lane lines would be easier to detect.
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

#below code is inspired from https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
def HistogramEqualization(img):
    return  cv2.equalizeHist(img)

def HistogramEqualization2(img, cl=2.0, gridsize=(8,8), seperateChannels = False):
    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=gridsize)
    if seperateChannels == False:
        return clahe.apply(img)
    else:
        A,B,C = cv2.split(img)
        A = clahe.apply(A)
        B = clahe.apply(B)
        C = clahe.apply(C)
        return cv2.merge([A,B,C])

#using the suggested normalization functions lead to poor results and resulting
#images had a mean value far from 0, around 1.3 in my tests
def normalize(img, seperateChannels = False):
    if seperateChannels == False:
        return (img - 128) / 128
    else:
        A,B,C = cv2.split(img)
        A = (A - 128) / 128
        B = (B - 128) / 128
        C = (C - 128) / 128
        return cv2.merge([A,B,C])

#Code below inspired from 
#https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/
#with this normalization function the output is around 0.16
def normalize2(img, seperateChannels = False):
    if seperateChannels == False:
        return preprocessing.normalize(img)
    else:
        A,B,C = cv2.split(img)
        A = preprocessing.normalize(A)
        B = preprocessing.normalize(B)
        C = preprocessing.normalize(C)
        return cv2.merge([A,B,C])

#Code below inspired from 
#https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/
#with both normalize2 and standardize applied together the mean is 
#basically zero as required, calculated value for visualation set shown below.
def Standardize(img, seperateChannels = False):
    if seperateChannels == False:
        return preprocessing.scale(img)
    else:
        A,B,C = cv2.split(img)
        A = preprocessing.scale(A)
        B = preprocessing.scale(B)
        C = preprocessing.scale(C)
        return cv2.merge([A,B,C])

def Sharpen(img, KernelSize=3, α=2, β=-1):
    blurred = gaussian_blur(img, KernelSize)
    return weighted_img(blurred, img, α, β)

def ReshapeGrayScale(img):
    return img.reshape.reshape((32,32,1))

def pltImages(images, labels, nrows = 1, ncols = 2, fig_w = 20, fig_h = 10, isgray = False):
    #below code is inspired from https://stackoverflow.com/questions/17111525/how-to-show-multiple-images-in-one-figure
    assert len(images) == len(labels)
    assert len(images) <= (nrows * ncols)
    
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    for index in range(len(images)):
        plot = fig.add_subplot(nrows,ncols,index+1)
        plot.set_title(labels[index])
        if(isgray == False):
            plt.imshow(images[index].squeeze())
        else:
            plt.imshow(images[index].squeeze(), cmap='gray')
    
    plt.show()

def Shuffle(imgs, labels):
    imgs, labels = shuffle(imgs, labels)
    return imgs, labels