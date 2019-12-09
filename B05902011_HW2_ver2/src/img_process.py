# python3 img_process.py lena.bmp histogram.png b_lena.png bounding.bmp
_DEBUG1 = False
_DEBUG2 = False
import sys

def main(argv):
	import numpy as np
	from PIL import Image, ImageDraw
	from matplotlib import pyplot as plt
	try:
		im = Image.open(argv[0]).convert('L')
	except:
		print("Error : Fail to open image")
		exit(1)

	def dist(coor):
	    axles = list(set(coor))
	    count = 0
	    for i in axles:
	        count += (i * len(np.where(coor == i)[0]))
	    count /= len(coor)
	    return count

	## histogram generation ##

	# recording pixel data
	his = np.zeros(256, dtype = np.int)
	im_2_array = np.array(im)
	# print(np.shape(im_2_array))
	X,Y = np.shape(im_2_array)
	for x in range(0, X):
		for y in range(0, Y):
			his[im_2_array[x,y]] += 1

	# DEBUG
	if(_DEBUG1):
		print(his)
		print(sum(his))

		check = im.histogram()
		print(check)
		print(sum(check))

		for i in range(0, 256):
			if his[i] != check[i]:
				print("Error : histogram error at %d" % i)
				print(his[i])
				print(check[i])
				exit(1)

	# DEBUG

	# draw the histogram
	x_axle = range(256)
	cm = plt.cm.get_cmap('cool')
	fig, ax = plt.subplots()
	ax.bar(x_axle, his, color = cm(np.array(x_axle)/255), width = 1, edgecolor = 'none')
	ax.set_title("Histogram")
	ax.set_xlabel("Value")
	ax.set_ylabel("Frequency")
	# plt.show()
	# output result
	try:
		plt.savefig(argv[1])
	except:
		print("Error : fail to save histogram")


	## Binary image generation ##
	from iprocess import binarize
	b_im = binarize(im, 128)
	draw = ImageDraw.Draw(b_im)	
	b_im.save(argv[2])

	b_array = np.array(b_im)


	## bonding box generation ##

	# Top down / bottom up labeling
	# first labeling
	label = np.zeros([X,Y] , dtype = np.int)
	n_label = 1
	for x in range(0, X):
		for y in range(0, Y):
			if(b_array[x,y] == 255):
				label[x,y] = n_label
				n_label += 1
	# top-down and bottom-up labeling
	change = True
	while(change):
	    change = False
	    for x in range(0, X):
	        for y in range(0, Y):
	            if(b_array[x,y] == 255):		# 8 way strat
	                if(x == 0):
	                    x_low = 0
	                else:
	                    x_low = x-1
	                if(y == 0):
	                    y_low = 0
	                else:
	                    y_low = y-1
	                x_high = x+2
	                y_high = y+2
	                
	                neighbors = label[x_low:x_high, y_low:y_high]
	                neighbors = neighbors[neighbors != 0]
	                
	                M = min(neighbors)
	                if (M != label[x,y]):
	                    change = True
	                    label[x,y] = M

	    for x in range(X-1, -1, -1):
	        for y in range(Y-1, -1, -1):
	            if(b_array[x,y] == 255):		# 8 way strat
	                if(x == 0):
	                    x_low = 0
	                else:
	                    x_low = x-1
	                if(y == 0):
	                    y_low = 0
	                else:
	                    y_low = y-1
	                x_high = x+2
	                y_high = y+2
	                
	                neighbors = label[x_low:x_high, y_low:y_high]
	                neighbors = neighbors[neighbors != 0]
	                
	                M = min(neighbors)
	                if (M != label[x,y]):
	                    change = True
	                    label[x,y] = M
	# threshold at pixel count > 500
	p_count = np.zeros(max(set(label.flatten())) + 1)
	for x in range(X):
	    for y in range(Y):
	        if(label[x,y] != 0): 
	            p_count[label[x,y]] += 1
	group_list = []
	for group in range(len(p_count)):
	    if(p_count[group] > 500):
	        group_list.append(group)

	# DEBUG
	if(_DEBUG2):
		print(group_list)
		if(len(group_list) != 5):
			print("Error : group count error")
			exit(1)
	# DEBUG

	p_count = np.zeros(max(set(label.flatten())) + 1)
	for x in range(X):
	    for y in range(Y):
	        if(label[x,y] != 0): 
	            p_count[label[x,y]] += 1
	group_list = []
	for group in range(len(p_count)):
	    if(p_count[group] > 500):
	        group_list.append(group)

	for group in group_list:
	    coordinates = np.where(label == group)
	    draw.rectangle((max(coordinates[1]), max(coordinates[0]), min(coordinates[1]),min(coordinates[0])), outline=(255), width = 3)
	    c = dist(coordinates[1])
	    r = dist(coordinates[0])
	    draw.ellipse((c-3, r-3, c+3, r+3), fill=(200), outline=(0))
	b_im.save(argv[3])

if __name__ == "__main__" :
	if(len(sys.argv) == 5):
		main(sys.argv[1:])
	else:
		print("Error : argument error")
		print("argument = %s" % str(sys.argv))
		exit(2)


# https://stackoverflow.com/questions/54059767/problem-plotting-a-histogram-of-grayscale-image-in-python
# https://www.youtube.com/watch?v=9D2sJ8G-nvE
# https://stackoverflow.com/questions/42656585/barplot-colored-according-a-colormap
# https://note.nkmk.me/en/python-pillow-imagedraw/

