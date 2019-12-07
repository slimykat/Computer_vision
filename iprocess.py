#import sys


def Robo(img):
    np.concatinate()
    

def main(argv):
    import numpy as np
    from PIL import Image
    import math
    img = np.array(Image.open(argv[1]))

if __name__ == '__main__':
    if(sys.argc == 2):
        main(argv)
    else:
        print("usage:")
        print("python3 iprocess.py lena.bmp")
