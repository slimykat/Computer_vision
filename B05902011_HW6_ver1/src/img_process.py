# python3 img_process.py lena.bmp

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
	b_array = ip.binarize(im2array, 128)

	# Downsampling
	Down_array = (ip.downsample(b_array, unit_size = 8).astype("uint8"))
	# yokoi
	X,Y = np.shape(Down_array)
	yokoi = ip.YokoiConnNumber(Down_array, X, Y)
	# output
	yokoi = np.array(yokoi).reshape(np.shape(Down_array))
	np.savetxt('yokoi.txt',yokoi.T, fmt = '%c', delimiter=" ")

if __name__ == "__main__" :
	if(len(sys.argv) == 2):
		main(sys.argv[0:])
	else:
		print("Usage:")
		print("python3 img_process.py lena.bmp dil_gray.png ero_gray.png open_gray.png close_gray.png")
		exit(2)


#	https://github.com/iamyuanchung/Fall-2015-CV/blob/master/hw4/main.py

