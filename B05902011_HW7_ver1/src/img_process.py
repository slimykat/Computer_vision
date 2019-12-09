# python3 img_process.py lena.bmp thined.png

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
	# thinning
	Thined = ip.thinning(Down_array).astype("uint8")
	img = Image.fromarray(Thined*255, mode ="L")
	img.save(argv[2])

if __name__ == "__main__" :
	if(len(sys.argv) == 3):
		main(sys.argv[0:])
	else:
		print("Usage:")
		print("python3 img_process.py lena.bmp thined.png")
		exit(2)


