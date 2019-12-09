# python3 img_process.py lena.bmp dil_gray.png ero_gray.png open_gray.png close_gray.png

import sys

def main(argv):
	import numpy as np
	from PIL import Image
	import matplotlib.image as img
	import matplotlib.pyplot as plt
	import iprocess as ip	# my library

	try:
		im = Image.open(argv[1]).convert('L')
	except:
		print("Error : Fail to open image")
		exit(1)
	im2array = np.array(im)

	# 3-5-5-3 kernel
	# origin at the center
	kernel = [[-2,-1],[-2,0],[-2,1],
    [-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],
    [0,-2],[0,-1],[0,0],[0,1],[0,2],
    [1,-2],[1,-1],[1,0],[1,1],[1,2],
    [2,-1],[2,0],[2,1]]


	#(a) Dilation
	dil_array = ip.gray_dilation(im2array, kernel).astype('uint8')
	img = Image.fromarray(dil_array, mode = 'L')
	img.save(argv[2])
	#(b) Erosion
	ero_array = ip.gray_erosion(im2array, kernel).astype('uint8')
	img = Image.fromarray(ero_array, mode = 'L')
	img.save(argv[3])
	#(c) Opening
	open_array = ip.gray_opening(im2array, kernel).astype('uint8')
	img = Image.fromarray(open_array, mode = 'L')
	img.save(argv[4])
	#(d) Closing
	close_array = ip.gray_closing(im2array, kernel).astype('uint8')
	img = Image.fromarray(close_array, mode = 'L')
	img.save(argv[5])


if __name__ == "__main__" :
	if(len(sys.argv) == 6):
		main(sys.argv[0:])
	else:
		print("Usage:")
		print("python3 img_process.py lena.bmp dil_gray.png ero_gray.png open_gray.png close_gray.png")
		exit(2)


#	https://github.com/iamyuanchung/Fall-2015-CV/blob/master/hw4/main.py

