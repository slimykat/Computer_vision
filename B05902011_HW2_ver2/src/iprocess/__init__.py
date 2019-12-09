
import numpy as np
from PIL import Image

def binarize(im , threshold):

    # turn into numpy array
    try:
        img_2_ndarray = np.array(im)
    except:
        print("Error : can not convert image to np.array")
        exit(1)
        
    # threshold method, binarize to get binary image
    x,y = np.shape(img_2_ndarray) 
    binary = np.zeros([x,y],dtype = 'uint8')

    for i in range(0, x):
        for j in range(0, y):
            if(img_2_ndarray[i,j] >= threshold):
                binary[i,j] = 255
            else:
                binary[i,j] = 0
    binary_im = Image.fromarray(binary, mode = 'L')

    return binary_im

