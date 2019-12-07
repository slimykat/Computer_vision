import sys
import numpy as np
from PIL import Image
import math

def expand(im_raw, size):
    X,Y = np.shape(im_raw)
    im = im_raw.copy().astype("int")
    row = np.zeros((size, Y), dtype = "int")
    col = np.zeros((X+2, size), dtype = "int")
    im = np.concatenate((row, im), axis = 0)
    im = np.concatenate((im, row), axis = 0)
    im = np.concatenate((col, im), axis = 1)
    im = np.concatenate((im, col), axis = 1)
    return im

def Robo(im_raw, threshold):
    X,Y = np.shape(im_raw)
    img = expand(im_raw, 1)
    f1 = lambda x : (img[x[0] + 1, x[1] + 1] - img[x])**2
    f2 = lambda x : (img[x[0] + 1, x[1] - 1] - img[x])**2 
    f = lambda x : math.sqrt(f1(x) + f2((x[0], x[1]+1)))
    for i in range(X):
        for j in range(Y):
            img[i+1, j+1] = 255 * (f((i+1, j+1)) < threshold )
    return img[1:X+1, 1:Y+1]

def Prew(im_raw, threshold):
    X,Y = np.shape(im_raw)
    img = expand(im_raw, 1)
    f1 = lambda x : (sum(img[x[0] + 1, x[1] - 1 : x[1] + 2]) 
                    - sum(img[x[0] - 1, x[1] - 1 : x[1] + 2]))**2
    f2 = lambda x : (sum(img[x[0] - 1 : x[0] + 2, x[1] + 1])
                    - sum(img[x[0] - 1 : x[0] + 2, x[1] - 1]))**2
    f = lambda x : math.sqrt(f1(x) + f2(x))
    for i in range(X):
        for j in range(Y):
            img[i + 1, j + 1] = 255 * (f((i + 1, j + 1)) < threshold)
    return img[1:X+1, 1:Y+1]

def main(argv):
    img = np.array(Image.open(argv[1]))

    Image.fromarray(Robo(img, 12).astype("uint8"), mode = "L").save("Robo.png")
    Image.fromarray(Prew(img, 12).astype("uint8"), mode = "L").save("Prew.png")

if __name__ == '__main__':
    if(len(sys.argv) == 2):
        main(sys.argv)
    else:
        print("usage:")
        print("python3 iprocess.py lena.bmp")
