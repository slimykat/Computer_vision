# python3 img_process.py lena.bmp a.png b.png c.png

_DEBUG = True
import sys

def main(argv):
	import numpy as np
	from PIL import Image
	import matplotlib.image as img
	import matplotlib.pyplot as plt
	import iprocess as ip	# my library

	try:
		im = Image.open(argv[0]).convert('L')
	except:
		print("Error : Fail to open image")
		exit(1)
	im2array = np.array(im)
	cm = plt.cm.get_cmap('cool')

	## (a) original image and histogram

	fig1 = plt.figure(figsize=(9, 4))
	ax1 = fig1.add_subplot(1,2,1)
	ax1.imshow(im, cmap = 'gray')
	ax1.axis('off')
	ax1.title.set_text("Origin lena")

	ax2 = fig1.add_subplot(1,2,2)
	his = ip.hist(im2array)
	ax2.bar(np.array(range(256)), his, width = 1.2, color = cm(np.array(range(256))/255))
	ax2.title.set_text("Histogram")

	fig1.savefig(argv[1])

	## (b) intensity divided by 3
	d_array = (im2array / 3).astype('uint8')
	d_im = Image.fromarray(d_array)

	fig2 = plt.figure(figsize=(9, 4))
	ax1 = fig2.add_subplot(1,2,1)
	ax1.imshow(d_im.convert('RGB'), cmap = 'gray')
	ax1.axis('off')
	ax1.title.set_text("Divided lena")

	ax2 = fig2.add_subplot(1,2,2)
	his = ip.hist(d_array)
	ax2.bar(np.array(range(256)), his, width = 1.2, color = cm(np.array(range(256))/255))
	ax2.title.set_text("Histogram")

	fig2.savefig(argv[2])

	## (c) Histogram equalization
	e_array = ip.Equalization(im2array)
	e_im = Image.fromarray(e_array)

	fig3 = plt.figure(figsize=(9, 4))
	ax1 = fig3.add_subplot(1,2,1)
	ax1.imshow(e_im, cmap = 'gray')
	ax1.axis('off')
	ax1.title.set_text("Equalized lena")

	ax2 = fig3.add_subplot(1,2,2)
	his = ip.hist(e_array)
	ax2.bar(np.array(range(256)), his, width = 1.2, color = cm(np.array(range(256))/255))
	ax2.title.set_text("Histogram")
	
	fig3.savefig(argv[3])



if __name__ == "__main__" :
	if(len(sys.argv) == 5):
		main(sys.argv[1:])
	else:
		print("Error : argument error")
		print("argument = %s" % str(sys.argv))
		exit(2)

	""" draw the histogram
				x_axle = range(256)
				cm = plt.cm.get_cmap('cool')
				fig, ax = plt.subplots()
				ax.bar(x_axle, his, color = cm(np.array(x_axle)/255), width = 1, edgecolor = 'none')
	"""



#	https://jason-chen-1992.weebly.com/home/-histogram-equalization

