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

def Robo(im_raw, threshold, copy_img):
    X,Y = np.shape(im_raw)
    new_img = copy_img.copy()
    f1 = lambda x : (copy_img[x[0] + 1, x[1] + 1] - copy_img[x])**2
    f2 = lambda x : (copy_img[x[0] + 1, x[1] - 1] - copy_img[x])**2 
    f = lambda x : math.sqrt(f1(x) + f2((x[0], x[1]+1)))
    for i in range(X):
        for j in range(Y):
            new_img[i+1, j+1] = 255 * (f((i+1, j+1)) < threshold )
    return new_img[1:X+1, 1:Y+1]

def Prew(im_raw, threshold, copy_img):
    X,Y = np.shape(im_raw)
    new_img = copy_img.copy()
    f1 = lambda x : (sum(copy_img[x[0] + 1, x[1] - 1 : x[1] + 2]) 
                    - sum(copy_img[x[0] - 1, x[1] - 1 : x[1] + 2]))**2
    f2 = lambda x : (sum(copy_img[x[0] - 1 : x[0] + 2, x[1] + 1])
                    - sum(copy_img[x[0] - 1 : x[0] + 2, x[1] - 1]))**2
    f = lambda x : math.sqrt(f1(x) + f2(x))
    for i in range(X):
        for j in range(Y):
            new_img[i + 1, j + 1] = 255 * (f((i + 1, j + 1)) < threshold)
    return new_img[1:X+1, 1:Y+1]

def Sobe(im_raw, threshold, copy_img):
    X,Y = np.shape(im_raw)
    new_img = copy_img.copy()
    f1 = lambda x : ((copy_img[x[0]+1, x[1]-1] + 2 * copy_img[x[0]+1, x[1]] + copy_img[x[0]+1, x[1]+1])
                - (copy_img[x[0]-1, x[1]-1] + 2 * copy_img[x[0]-1, x[1]] + copy_img[x[0]-1, x[1]+1] ))
    f2 = lambda x : ((copy_img[x[0]-1, x[1]+1] + 2 * copy_img[x[0], x[1]+1] + copy_img[x[0]+1, x[1]+1]) 
                - (copy_img[x[0]-1, x[1]-1] + 2 * copy_img[x[0], x[1]-1] + copy_img[x[0]+1, x[1]-1] ))
    f = lambda x : math.sqrt(f1(x)**2 + f2(x)**2)
    for i in range(X):
        for j in range(Y):
            new_img[i + 1, j + 1] = 255 * (f((i+1, j+1)) < threshold)
    return new_img[1:X+1, 1:Y+1]

def Frei(im_raw, threshold, copy_img):
    X,Y = np.shape(im_raw)
    new_img = copy_img.copy()
    sq2 = math.sqrt(2)
    f1 = lambda x : ((copy_img[x[0]+1, x[1]-1] + sq2 * copy_img[x[0]+1, x[1]] + copy_img[x[0]+1, x[1]+1])
                - (copy_img[x[0]-1, x[1]-1] + sq2 * copy_img[x[0]-1, x[1]] + copy_img[x[0]-1, x[1]+1] ))
    f2 = lambda x : ((copy_img[x[0]-1, x[1]+1] + sq2 * copy_img[x[0], x[1]+1] + copy_img[x[0]+1, x[1]+1]) 
                - (copy_img[x[0]-1, x[1]-1] + sq2 * copy_img[x[0], x[1]-1] + copy_img[x[0]+1, x[1]-1] ))
    f = lambda x : math.sqrt(f1(x)**2 + f2(x)**2)
    for i in range(X):
        for j in range(Y):
            new_img[i + 1, j + 1] = 255 * (f((i+1, j+1)) < threshold)
    return new_img[1:X+1, 1:Y+1] 
            

def Kirs(copy_raw, threshold):
    size = 1
    k0 = np.array([[-3, -3, 5],[-3, 0, 5],[-3, -3, 5]])
    k1 = np.array([[-3, 5, 5],[-3, 0, 5],[-3, -3, -3]])
    k2 = np.array([[5, 5, 5],[-3, 0, -3],[-3, -3, -3]])
    k3 = np.array([[5, 5, -3],[5, 0, -3],[-3, -3, -3]])
    k4 = np.array([[5, -3, -3],[5, 0, -3],[5, -3, -3]])
    k5 = np.array([[-3, -3, -3],[5, 0, -3],[5, 5, -3]])
    k6 = np.array([[-3, -3, -3],[-3, 0, -3],[5, 5, 5]])
    k7 = np.array([[-3, -3, -3],[-3, 0, 5],[-3, 5, 5]])
    mask_list = [k0,k1,k2,k3,k4,k5,k6,k7]
    return max_Mask(copy_raw, size, threshold, mask_list)

def max_Mask(copy_raw, size, threshold, mask_list):
    #extended size
    X, Y = np.shape(copy_raw)
    new_img = copy_raw.copy()

    for i in range(size, X - size):
        for j in range(size, Y - size):
            grad = np.NINF
            for mask in mask_list:
                current_grad = np.sum(np.multiply(mask, copy_raw[i-1 : i+2, j-1 : j+2]))
                grad = max(grad, current_grad)
                if(grad >= threshold):
                    break
            new_img[i, j] = 255 * (grad < threshold)

    return new_img[size : X-size, size : Y-size]

def main(argv):
    img = np.array(Image.open(argv[1]))
    copy_img = expand(img, 1)
    Image.fromarray(Robo(img, 30, copy_img).astype("uint8"), mode = "L").save("Robo.png")
    Image.fromarray(Prew(img, 30, copy_img).astype("uint8"), mode = "L").save("Prew.png")
    Image.fromarray(Sobe(img, 38, copy_img).astype("uint8"), mode = "L").save("Sobe.png")
    Image.fromarray(Frei(img, 30, copy_img).astype("uint8"), mode = "L").save("Frei.png")
if __name__ == '__main__':
    if(len(sys.argv) == 2):
        main(sys.argv)
    else:
        print("usage:")
        print("python3 iprocess.py lena.bmp")
