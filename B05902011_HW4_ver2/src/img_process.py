# python3 img_process.py lena.bmp dil.png ero.png open.png close.png HnM.png

_DEBUG = True
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
	cm = plt.cm.get_cmap('cool')
	b_array = ip.binarize(im2array, 128)

	# 3-5-5-3 kernel
	# origin at the center
	kernel = [[-2,-1],[-2,0],[-2,1],
    [-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],
    [0,-2],[0,-1],[0,0],[0,1],[0,2],
    [1,-2],[1,-1],[1,0],[1,1],[1,2],
    [2,-1],[2,0],[2,1]]
	J_kernel = [[0,-1],[0,0],[1,0]]
	K_kernel = [[-1,0],[-1,1],[0,1]]


	#(a) Dilation
	dil_array = ip.dilation(b_array, kernel).astype('uint8')
	img = Image.fromarray(dil_array * 255, mode = 'L')
	img.save(argv[2])
	#(b) Erosion
	ero_array = ip.erosion(b_array, kernel).astype('uint8')
	img = Image.fromarray(ero_array * 255, mode = 'L')
	img.save(argv[3])
	#(c) Opening
	open_array = ip.opening(b_array, kernel).astype('uint8')
	img = Image.fromarray(open_array * 255, mode = 'L')
	img.save(argv[4])
	#(d) Closing
	close_array = ip.closing(b_array, kernel).astype('uint8')
	img = Image.fromarray(close_array * 255, mode = 'L')
	img.save(argv[5])
	#(e) Hit-and-miss transform
	h_and_m = ip.hit_and_miss(b_array, J_kernel, K_kernel).astype('uint8')
	img = Image.fromarray(h_and_m * 255, mode = 'L')
	img.save(argv[6])

if __name__ == "__main__" :
	if(len(sys.argv) == 7):
		main(sys.argv[0:])
	else:
		print("Error : argument error")
		print("argument = %s" % str(sys.argv))
		exit(2)


#	https://github.com/iamyuanchung/Fall-2015-CV/blob/master/hw4/main.py
#	yes, i did found the full solution for this task
#	but i did all the coding by hand
#	and i also improved the functions (also my previous homework function)

"""
Binarize Lena with the threshold 128 (0-127,128-255).
Please use the octogonal 3-5-5-5-3 kernel.
Please use the "L" shaped kernel (same as the text book) to detect the upper-right corner for hit-and-miss transform.
Please process the white pixels (operating on white pixels).
5 images should be included in your report: Dilation, Erosion, Opening, Closing, and Hit-and-Miss.
"""

