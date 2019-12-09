import sys

def main(argv):
    from PIL import Image
    import numpy as np
    import iprocess as ip

    img = np.array(Image.open(argv[1]))
    
    copy_img = ip.expand(img, 1)
    # modified by gradient magnitude value
    """
    Image.fromarray(ip.Robo(img, 12, copy_img).astype("uint8"), mode = "L").save("Robo.png")
    Image.fromarray(ip.Prew(img, 24, copy_img).astype("uint8"), mode = "L").save("Prew.png")
    Image.fromarray(ip.Sobe(img, 38, copy_img).astype("uint8"), mode = "L").save("Sobe.png")
    Image.fromarray(ip.Frei(img, 30, copy_img).astype("uint8"), mode = "L").save("Frei.png")
    """
    # modified by maximum mask sum
    Image.fromarray(ip.Kirs(copy_img, 135).astype("uint8"), mode = "L").save("Kirs.png")
    Image.fromarray(ip.Robi(copy_img, 43).astype("uint8"), mode = "L").save("Robi.png")
    copy_img = ip.expand(img, 2)
    Image.fromarray(ip.Neva(copy_img, 12500).astype("uint8"), mode = "L").save("Neva.png")
    
if __name__ == '__main__':
    if(len(sys.argv) == 2):
        main(sys.argv)
    else:
        print("usage:")
        print("python3 img_process.py lena.bmp")
