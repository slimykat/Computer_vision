
def main():
    #import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    # read image and turn into numpy array
    im = Image.open("lena.bmp")
    img_2_ndarray = np.array(im)

    # theshold, binarize at 128 to get binary image
    x,y = np.shape(img_2_ndarray) 
    binary = np.zeros([x,y],dtype = 'uint8')

    for i in range(0, x):
        for j in range(0, y):
            if(img_2_ndarray[i,j] >= 128):
                binary[i,j] = 255
            else:
                binary[i,j] = 0
    binary_im = Image.fromarray(binary, mode = 'L')

    # show result
    binary_im.show()

    # save image
    binary_im.save("binarized_lena.bmp")
    
if __name__ == "__main__":
   main()
