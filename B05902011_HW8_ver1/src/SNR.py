# python3 SNR.py <name.png>

import sys

def main(argv):
	import numpy as np
	from PIL import Image
	import iprocess as ip	# my library

	for img_name in argv[1:]:
		num = 0
		try:
			origin = np.array(Image.open(img_name).convert('L'))
			num += 1
			box3 = np.array(Image.open(img_name[:-4] + "_box3" + ".png").convert('L'))
			num += 1
			box5 = np.array(Image.open(img_name[:-4] + "_box5" + ".png").convert('L'))
			num += 1
			CO = np.array(Image.open(img_name[:-4] + "_CO" + ".png").convert('L'))
			num += 1
			med3 = np.array(Image.open(img_name[:-4] + "_med3" + ".png").convert('L'))
			num += 1
			med5 = np.array(Image.open(img_name[:-4] + "_med5" + ".png").convert('L'))
			num += 1
			OC = np.array(Image.open(img_name[:-4] + "_OC" + ".png").convert('L'))
			num += 1
		except:
			print("Error : Fail to open image")
			print(img_name + " : " + str(num))
			exit(1)

		file = open(img_name[:-4] + ".txt", 'w')
		file.write("box3 : " + str(ip.SNR(origin, box3)) + "\n")
		file.write("box5 : " + str(ip.SNR(origin, box5)) + "\n")
		file.write("med3 : " + str(ip.SNR(origin, med3)) + "\n")
		file.write("med5 : " + str(ip.SNR(origin, med5)) + "\n")
		file.write("CO : " + str(ip.SNR(origin, CO)) + "\n")
		file.write("OC : " + str(ip.SNR(origin, OC)) + "\n")

		file.close()

if __name__ == "__main__" :
	if(len(sys.argv) >= 2):
		main(sys.argv[0:])
	else:
		print("Usage:")
		print("python3 SNR.py <name.png> ...")
		exit(2)


