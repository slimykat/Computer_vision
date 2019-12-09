
# >python3 img_process.py lena.bmp ud_im.bmp lr_im.bmp d_im.bmp

import sys


def main(argv):
	# import matplotlib.pyplot as plt
	import numpy as np
	from PIL import Image
	if (len(argv) == 4):
		
		# read the required image
		im = Image.open(argv[0])
		"""im = plt.imread(argv[0])"""
		
		# image process
		img_2_ndarray = np.array(im)
		up_im = Image.fromarray(img_2_ndarray[-1::-1])
		lr_im = Image.fromarray(img_2_ndarray[:,-1::-1])

		x,y = np.shape(img_2_ndarray) 
		temp = img_2_ndarray
		for i in range(0, x):
		    for j in range(i+1, y):
		    	temp[i][j],temp[j][i]=temp[j][i],temp[i][j]
		d_im = Image.fromarray(temp)
		
		# write image
		up_im.save(argv[1])
		lr_im.save(argv[2])
		d_im.save(argv[3])
		
		# Test process, only for testing
		"""
		im = np.flipud(im)
		plt.subplot(2, 1, 1)
		plt.imshow(im)
		plt.subplot(2, 1, 2)
		plt.imshow(np.
		fliplr(im))
		"""
		print("process succeeded")
	else :
		print("argument error")
		print("arguments = " ,argv)


if __name__ == "__main__":
   main(sys.argv[1:])


# https://stackoverflow.com/questions/9154120/how-can-i-flip-an-image-along-the-vertical-axis-with-python
# https://medium.com/jameslearningnote/資料分析-機器學習-第2-5講-資料視覺化-matplotlib-seaborn-plotly-75cd353d6d3f