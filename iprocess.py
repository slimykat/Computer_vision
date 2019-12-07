import sys
import numpy as np
from PIL import Image
import math

def Robo(im, threshold):
    img = im.copy().astype("int")
    #expanding
    X,Y = np.shape(img)
    row = np.zeros((1, Y), dtype = "uint8")
    col = np.zeros((X+2, 1), dtype = "uint8")
    img = np.concatenate((row, img), axis = 0)
    img = np.concatenate((img, row), axis = 0)
    img = np.concatenate((img, col), axis = 1)
    img = np.concatenate((col, img), axis = 1)

    print(img)
    
    f1 = lambda x : (img[x[0] + 1, x[1] + 1] - img[x])**2
    f2 = lambda x : (img[x[0] + 1, x[1] - 1] - img[x])**2 
    f = lambda x : math.sqrt(f1(x) + f2( (x[0], x[1]+1) ))
    for i in range(X):
        for j in range(Y):
            if(f((i+1,j+1)) >= threshold):
                img[i+1, j+1] = 0
            else:
                img[i+1, j+1] = 255
    return img[1:X+1, 1:Y+1]

def main(argv):
    img = np.array(Image.open(argv[1]))

    Image.fromarray(Robo(img, 30).astype("uint8"), mode = "L").save("Robo.png")

if __name__ == '__main__':
    if(len(sys.argv) == 2):
        main(sys.argv)
    else:
        print("usage:")
        print("python3 iprocess.py lena.bmp")
