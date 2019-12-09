# python3 img_process.py lena.bmp 

import sys


def main(argv):
	import numpy as np
	from PIL import Image
	# import matplotlib.image as img
	# import matplotlib.pyplot as plt
	import iprocess as ip	# my library

	try:
		im = Image.open(argv[1]).convert('L')
	except:
		print("Error : Fail to open image")
		exit(1)
	img = np.array(im).astype("uint8")
	kernel = [[-2,-1],[-2,0],[-2,1],
	    [-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],
	    [0,-2],[0,-1],[0,0],[0,1],[0,2],
	    [1,-2],[1,-1],[1,0],[1,1],[1,2],
	    [2,-1],[2,0],[2,1]]

	# noisy images
	gaussian_10 = ip.get_gaussian_noise(img, 10)
	gaussian_30 = ip.get_gaussian_noise(img, 30)
	pepper_010 = ip.get_SaltAndPepper_noise(img, 0.1)
	pepper_005 = ip.get_SaltAndPepper_noise(img, 0.05)

	ip.save_gray_Image(gaussian_10,"gaussian_10.png")
	ip.save_gray_Image(gaussian_30,"gaussian_30.png")
	ip.save_gray_Image(pepper_010,"pepper_010.png")
	ip.save_gray_Image(pepper_005,"pepper_005.png")

	# box filter
	ip.box_filter(gaussian_30, 3).save_image("gaussian_30_box3.png")
	ip.box_filter(gaussian_30, 5).save_image("gaussian_30_box5.png")
	ip.box_filter(gaussian_10, 3).save_image("gaussian_10_box3.png")
	ip.box_filter(gaussian_10, 5).save_image("gaussian_10_box5.png")

	ip.box_filter(pepper_010, 3).save_image("pepper_010_box3.png")
	ip.box_filter(pepper_010, 5).save_image("pepper_010_box5.png")
	ip.box_filter(pepper_005, 3).save_image("pepper_005_box3.png")
	ip.box_filter(pepper_005, 5).save_image("pepper_005_box5.png")

	# medium filter
	ip.medium_filter(gaussian_30, 3).save_image("gaussian_30_med3.png")
	ip.medium_filter(gaussian_30, 5).save_image("gaussian_30_med5.png")
	ip.medium_filter(gaussian_10, 3).save_image("gaussian_10_med3.png")
	ip.medium_filter(gaussian_10, 5).save_image("gaussian_10_med5.png")

	ip.medium_filter(pepper_010, 3).save_image("pepper_010_med3.png")
	ip.medium_filter(pepper_010, 5).save_image("pepper_010_med5.png")
	ip.medium_filter(pepper_005, 3).save_image("pepper_005_med3.png")
	ip.medium_filter(pepper_005, 5).save_image("pepper_005_med5.png")

	# opening then closing
	OC1 = ip.gray_closing(ip.gray_opening(gaussian_30, kernel), kernel)
	OC2 = ip.gray_closing(ip.gray_opening(gaussian_10, kernel), kernel)
	OC3 = ip.gray_closing(ip.gray_opening(pepper_010, kernel), kernel)
	OC4 = ip.gray_closing(ip.gray_opening(pepper_005, kernel), kernel)

	ip.save_gray_Image(OC1,"gaussian_30_OC.png")
	ip.save_gray_Image(OC2,"gaussian_10_OC.png")
	ip.save_gray_Image(OC3,"pepper_010_OC.png")
	ip.save_gray_Image(OC4,"pepper_005_OC.png")

	# closing then opening
	CO1 = ip.gray_opening(ip.gray_closing(gaussian_30, kernel), kernel)
	CO2 = ip.gray_opening(ip.gray_closing(gaussian_10, kernel), kernel)
	CO3 = ip.gray_opening(ip.gray_closing(pepper_010, kernel), kernel)
	CO4 = ip.gray_opening(ip.gray_closing(pepper_005, kernel), kernel)

	ip.save_gray_Image(CO1,"gaussian_30_CO.png")
	ip.save_gray_Image(CO2,"gaussian_10_CO.png")
	ip.save_gray_Image(CO3,"pepper_010_CO.png")
	ip.save_gray_Image(CO4,"pepper_005_CO.png")


if __name__ == "__main__" :
	if(len(sys.argv) == 2):
		main(sys.argv[0:])
	else:
		print("Usage:")
		print("python3 img_process.py lena.bmp")
		exit(2)


