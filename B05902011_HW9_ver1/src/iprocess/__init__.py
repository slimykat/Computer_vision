import numpy as np
from math import log
from math import sqrt
from PIL import Image

def binarize(im , threshold):
    
    x,y = np.shape(im) 
    binary = np.zeros(im.shape,dtype = 'bool')

    for row in range(im.shape[0]):
        binary[row] = (im[row] >= threshold)

    return binary


def distance(coor):
    axis = list(set(coor))
    count = 0
    for i in axis:
        count += (i * len(np.where(coor == i)[0]))
    count /= len(coor)

    return count


def hist(im):
    his = np.zeros(256, dtype = np.int)
    X,Y = np.shape(im)
    for x in range(0, X):
        for y in range(0, Y):
            his[im[x,y]] += 1
    return his


def pixel_label_8(b_array, threshold):
    
    label = np.zeros(b_array.shape() , dtype = np.int)
    n_label = 1

    # initialize
    for x in range(0, X):
        for y in range(0, Y):
            if(b_array[x,y] > 0):
                label[x,y] = n_label
                n_label += 1

    # top-down and bottom-up labeling
    change = True
    while(change):
        change = False
        for x in range(0, X):
            for y in range(0, Y):
                if(b_array[x,y] > 0):    # 8 way strat
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
                if(b_array[x,y] > 0):    # 8 way strat
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

    # threshold
    p_count = np.zeros(max(set(label.flatten())) + 1)
    for x in range(X):
        for y in range(Y):
            if(label[x,y] != 0): 
                p_count[label[x,y]] += 1
    group_list = []
    for group in range(len(p_count)):
        if(p_count[group] > threshold):
            group_list.append(group)

    p_count = np.zeros(max(set(label.flatten())) + 1)
    for x in range(X):
        for y in range(Y):
            if(label[x,y] != 0): 
                p_count[label[x,y]] += 1
    group_list = []
    for group in range(len(p_count)):
        if(p_count[group] > threshold):
            group_list.append(group)

    return (label, group_list)

def Equalization(SrcImg):
    SrcHistogram = hist(SrcImg)
    EquaHistogram = np.zeros(256, dtype = np.int)
    sum = 0
    Cumulative = np.zeros(256, dtype = np.float32)
    Img_row_num, Img_col_num = SrcImg.shape
    SrcImg = np.array(SrcImg)
    totalPixel = Img_row_num * Img_col_num

    for i in range(256):
        # cumulative[j] = n_j / total_pixel, 0<=j<=255
        Cumulative[i] = SrcHistogram[i] / totalPixel
        # s_k = 255 * sum(cumulative[j] | 0<=j<=k), 0<=k<=255
        sum += round(Cumulative[i] * 255)
        EquaHistogram[i] = sum

    Equalized_img_array = np.zeros([Img_row_num, Img_col_num], dtype = 'uint8')
    for x in range(Img_row_num):
        for y in range(Img_col_num):
            Equalized_img_array[x, y] = EquaHistogram[SrcImg[x, y]]
            
    return Equalized_img_array

def dilation(b_array, kernel):
    b_array = b_array.astype('bool')
    dil_array = np.zeros(b_array.shape, 'bool')
    for i in range(b_array.shape[0]):
        for j in range(b_array.shape[1]):
            if(b_array[i][j] > 0):
                for points in kernel:
                    if(0 <= i + points[0] < b_array.shape[0] and 0 <= j + points[1] < b_array.shape[1]):
                        dil_array[i + points[0]][j + points[1]] = 1
    return dil_array

def erosion(b_array, kernel):
    trans_kernel = []
    for points in kernel:
        trans_kernel.append([-1*points[0], -1*points[1]])
    invert_b = np.logical_not(b_array)
    return np.logical_not(dilation(invert_b, trans_kernel))

def closing(b_array, kernel):
    return erosion(dilation(b_array, kernel), kernel)

def opening(b_array, kernel):
    return dilation(erosion(b_array, kernel), kernel)

def hit_and_miss(b_array, J_kernel, K_kernel):
    invert_b = np.logical_not(b_array)
    IN = erosion(b_array, J_kernel)
    OUT = erosion(invert_b, K_kernel)
    return np.logical_and(IN ,OUT)

def gray_dilation(img, kernel):
    window = lambda offset : (offset [0] + i , offset[1] + j)
    X, Y = np.shape(img)
    new_img = img.copy()
    pixel = lambda position : img[position] if (0 <= position[0] < X and 0 <= position[1] < Y) else 0
    for i in range(X):
        for j in range(Y):
            new_img[i,j] = max(list(map(pixel, list(map(window, kernel)))))
    return new_img

def gray_erosion(img, kernel):
    img_ero = img.copy()
    X,Y = np.shape(img)
    window = lambda offset : (offset [0] + i , offset[1] + j)
    pixel = lambda position : img[position] if (0 <= position[0] < X and 0 <= position[1] < Y) else 0
    for i in range(X):
        for j in range(Y):
            img_ero[i,j] = min(list(map(pixel, list(map(window, kernel)))))
    return img_ero

def gray_closing(img_array, kernel):
    return gray_erosion(gray_dilation(img_array, kernel), kernel)

def gray_opening(img_array, kernel):
    return gray_dilation(gray_erosion(img_array, kernel), kernel)

def downsample(b_array, unit_size):
    x,y = np.shape(b_array)
    X = int(x//unit_size)
    Y = int(y//unit_size)
    Down_array = np.zeros([X, Y], dtype = "bool")
    for i in range(int(x//unit_size)):
        for j in range(int(y//unit_size)):
            Down_array[i, j] = b_array[i*unit_size, j*unit_size]
    return Down_array

class YokoiConnNumber(object):
    # 4 connect
    def _h(cls, l):
        b,c,d,e = l
        if (b != c):
            return 's'
        if (b == c == d == e):
            return 'r'
        return 'q'

    def _f(cls, pixels, center):
        width, height = np.shape(pixels)
        center_x, center_y = center
        x = list(map(lambda position: pixels[position] if 0 <= position[1] < width and 0 <= position[0] < height else 0,
                        map(lambda position: (center_x + position[0], center_y + position[1]),
                            [(0, 0), (0, 1), (-1, 0), (0, -1), (1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]
                        )
                    ))

        a = list(map(cls._h, [
                        (x[0], x[1], x[6], x[2]),
                        (x[0], x[2], x[7], x[3]),
                        (x[0], x[3], x[8], x[4]),
                        (x[0], x[4], x[5], x[1]),
                    ]))

        return(5 if all(ai == 'r' for ai in a) else a.count('q'))


def getBorderPixels(src):
    yok = YokoiConnNumber()
    X,Y = np.shape(src)
    marked = np.zeros((X,Y), dtype = 'bool')
    for i in range(X):
        for j in range(Y):
            if (src[i, j] == 0): # ignoring black pixels
                continue
            type = yok._f(src, (i,j))
            if(type == 1):
                marked[i,j] = 1;

    return marked

def doPairRelationship(src, border):

    X,Y = np.shape(src)
    marked = np.chararray((X,Y))
    for i in range(X):
        for j in range(Y):
            if (border[i, j] == 1):
                # it is a border pixel
                offsets = list(map(lambda xy: (i + xy[0], j + xy[1]), [(0, 1), (-1, 0), (0, -1), (1, 0)]))
                s = 0
                for n in offsets:
                    try:
                        s += border[n]
                    except:
                        s += 0
                if(s >= 1):
                    marked[i, j] = 'p'
                else:
                    marked[i, j] = 'q'
            else:
                # it is not border pixel
                marked[i, j] = 'q'
                
    return marked

def thinning(img):

    def h4(b, c, d, e):
        if (b == c and (d != b or e != b)):
            return 1
        return 0

    def f(a1, a2, a3, a4, x):
        s = a1 + a2 + a3 + a4
        return 'g' if (s == 1) else x
    X, Y = np.shape(img)
    while (True):
        nothingChanged = True

        # mark interior border pixel
        borderImage = getBorderPixels(img)

        # pair relationship operator
        pairImage = doPairRelationship(img, borderImage)

        # connected shrink operator
        # the h function for 4-connected connected shrink
        # top down, left right
        for i in range(X):
            for j in range(Y):
                n = map(lambda coord: img[coord] if (0 <= coord[0] < X and 0 <= coord[1] < Y) else 0,
                    list(map(lambda xy: (i + xy[0], j + xy[1]), 
                        [(0, 0), (0, 1), (-1, 0), (0, -1), (1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]
                    )))
                n = list(n)
                a1 = h4(n[0], n[1], n[6], n[2])
                a2 = h4(n[0], n[2], n[7], n[3])
                a3 = h4(n[0], n[3], n[8], n[4])
                a4 = h4(n[0], n[4], n[5], n[1])
                tmp = f(a1, a2, a3, a4, img[i, j])
                if (tmp == 'g' and pairImage[i, j] == ('p').encode(encoding="utf-8")):
                    img[i, j] = 0
                    nothingChanged = False
        if (nothingChanged):
            break
    return img

def get_gaussian_noise(img, amplitude):
    np.random.seed(int(1000 * img[0,0] / 3))
    new_img = img.astype("int") # prevent overflow
    new_img = new_img + amplitude * np.random.normal(0, 1, np.shape(img))
    new_img = np.clip(new_img, 0, 255)
    return new_img.astype("uint8")

def get_SaltAndPepper_noise(img, threshold):
    X, Y = np.shape(img)
    img = img.astype("float") 
    new_img = img.copy()
    distribute = np.random.uniform(0, 1, (X,Y))
    for i in range(X):
        new_img[i][distribute[i] <= threshold] = 0
        new_img[i][distribute[i] >= (1 - threshold)] = 255
    return new_img.astype("uint8")

class box_filter():
    def __init__ (self, img, box_size):
        box_size = int(box_size/2)
        X,Y = np.shape(img)
        self.result = img.copy()
        for i in range(X):
            for j in range(Y):
                up = i - box_size if(i >= box_size) else 0
                down = i + box_size if(i + box_size < X) else X - 1
                left = j - box_size if(j >= box_size) else 0
                right = j + box_size if(j + box_size < Y) else Y - 1
                self.result[i, j] = np.mean(img[up:down+1, left:right+1])
    def save_image(self, name):
        save_gray_Image(self.result, name)

class medium_filter():
    def __init__ (self, img, box_size):
        box_size = int(box_size/2)
        X, Y = np.shape(img)
        self.result = img.copy()
        for i in range(X):
            for j in range(Y):
                up = i - box_size if(i >= box_size) else 0
                down = i + box_size if(i + box_size < X) else X - 1
                left = j - box_size if(j >= box_size) else 0
                right = j + box_size if(j + box_size < Y) else Y - 1
                sort = np.sort(img[up:down+1, left:right+1], axis = None)
                self.result[i, j] = sort[int(sort.size / 2)]
    def save_image(self, name):
        save_gray_Image(self.result, name)

def save_gray_Image(img, name):
    Image.fromarray(img.astype("uint8"), mode = "L").save(name)
def save_bianry_Image(img, name):
    Image.fromarray(255*img.astype("uint8"), mode = "L").save(name)

def SNR(origin_img, noice_img):
    normalize_origin = origin_img / 255
    normalize_noice = noice_img / 255
    dim = normalize_origin.size
    
    mu = np.sum(normalize_origin, axis=None) / dim
    VS = np.sum(np.square(normalize_origin - mu), axis=None) / dim
    delta = normalize_noice - normalize_origin
    muN = np.sum(delta, axis=None) / dim
    VN = np.sum(np.square(delta - muN), axis=None) / dim
    return 20*log(sqrt(VS / VN), 10)

def expand(im_raw, size):
    X,Y = np.shape(im_raw)
    im = im_raw.copy().astype("int")
    row = np.zeros((size, Y), dtype = "int")
    col = np.zeros((X+2*size, size), dtype = "int")
    im = np.concatenate((row, im), axis = 0)
    im = np.concatenate((im, row), axis = 0)
    im = np.concatenate((col, im), axis = 1)
    im = np.concatenate((im, col), axis = 1)
    return im

def Robo(im_raw, threshold, copy_img):
    X,Y = np.shape(im_raw)
    new_img = copy_img.copy()
    f1 = lambda x : (copy_img[x[0] + 1, x[1] + 1] - copy_img[x])**2
    f2 = lambda x : (copy_img[x[0] + 1, x[1] - 1] - copy_img[x])**2 
    f = lambda x : sqrt(f1(x) + f2((x[0], x[1]+1)))
    for i in range(X):
        for j in range(Y):
            new_img[i+1, j+1] = 255 * (f((i+1, j+1)) < threshold )
    return new_img[1:X+1, 1:Y+1]

def Prew(im_raw, threshold, copy_img):
    X,Y = np.shape(im_raw)
    new_img = copy_img.copy()
    f1 = lambda x : (sum(copy_img[x[0] + 1, x[1] - 1 : x[1] + 2]) 
                    - sum(copy_img[x[0] - 1, x[1] - 1 : x[1] + 2]))**2
    f2 = lambda x : (sum(copy_img[x[0] - 1 : x[0] + 2, x[1] + 1])
                    - sum(copy_img[x[0] - 1 : x[0] + 2, x[1] - 1]))**2
    f = lambda x : sqrt(f1(x) + f2(x))
    for i in range(X):
        for j in range(Y):
            new_img[i + 1, j + 1] = 255 * (f((i + 1, j + 1)) < threshold)
    return new_img[1:X+1, 1:Y+1]

def Sobe(im_raw, threshold, copy_img):
    X,Y = np.shape(im_raw)
    new_img = copy_img.copy()
    f1 = lambda x : ((copy_img[x[0]+1, x[1]-1] + 2 * copy_img[x[0]+1, x[1]] + copy_img[x[0]+1, x[1]+1])
                - (copy_img[x[0]-1, x[1]-1] + 2 * copy_img[x[0]-1, x[1]] + copy_img[x[0]-1, x[1]+1] ))
    f2 = lambda x : ((copy_img[x[0]-1, x[1]+1] + 2 * copy_img[x[0], x[1]+1] + copy_img[x[0]+1, x[1]+1]) 
                - (copy_img[x[0]-1, x[1]-1] + 2 * copy_img[x[0], x[1]-1] + copy_img[x[0]+1, x[1]-1] ))
    f = lambda x : sqrt(f1(x)**2 + f2(x)**2)
    for i in range(X):
        for j in range(Y):
            new_img[i + 1, j + 1] = 255 * (f((i+1, j+1)) < threshold)
    return new_img[1:X+1, 1:Y+1]

def Frei(im_raw, threshold, copy_img):
    X,Y = np.shape(im_raw)
    new_img = copy_img.copy()
    sq2 = sqrt(2)
    f1 = lambda x : ((copy_img[x[0]+1, x[1]-1] + sq2 * copy_img[x[0]+1, x[1]] + copy_img[x[0]+1, x[1]+1])
                - (copy_img[x[0]-1, x[1]-1] + sq2 * copy_img[x[0]-1, x[1]] + copy_img[x[0]-1, x[1]+1] ))
    f2 = lambda x : ((copy_img[x[0]-1, x[1]+1] + sq2 * copy_img[x[0], x[1]+1] + copy_img[x[0]+1, x[1]+1]) 
                - (copy_img[x[0]-1, x[1]-1] + sq2 * copy_img[x[0], x[1]-1] + copy_img[x[0]+1, x[1]-1] ))
    f = lambda x : sqrt(f1(x)**2 + f2(x)**2)
    for i in range(X):
        for j in range(Y):
            new_img[i + 1, j + 1] = 255 * (f((i+1, j+1)) < threshold)
    return new_img[1:X+1, 1:Y+1] 
            

def Kirs(copy_raw, threshold):
    size = 1
    k0 = np.array([[-3, -3, 5],[-3, 0, 5],[-3, -3, 5]])
    k1 = np.array([[-3, 5, 5],[-3, 0, 5],[-3, -3, -3]])
    k2 = np.array([[5, 5, 5],[-3, 0, -3],[-3, -3, -3]])
    k3 = np.array([[5, 5, -3],[5, 0, -3],[-3, -3, -3]])
    k4 = np.array([[5, -3, -3],[5, 0, -3],[5, -3, -3]])
    k5 = np.array([[-3, -3, -3],[5, 0, -3],[5, 5, -3]])
    k6 = np.array([[-3, -3, -3],[-3, 0, -3],[5, 5, 5]])
    k7 = np.array([[-3, -3, -3],[-3, 0, 5],[-3, 5, 5]])
    mask_list = [k0,k1,k2,k3,k4,k5,k6,k7]
    return max_Mask(copy_raw, size, threshold, mask_list)

def Robi(copy_raw, threshold):
    size = 1
    k0 = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    k1 = np.array([[0,1,2], [-1,0,1], [-2,-1,0]])
    k2 = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    k3 = np.array([[2,1,0], [1,0,-1], [0,-1,-2]])
    k4 = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
    k5 = np.array([[0,-1,-2], [1,0,-1], [2,1,0]])
    k6 = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    k7 = np.array([[-2,-1,0], [-1,0,1], [0,1,2]])
    mask_list = [k0,k1,k2,k3,k4,k5,k6,k7]
    return max_Mask(copy_raw, size, threshold, mask_list)

def Neva(copy_raw, threshold):
    size = 2
    k0 = np.array([[100,100,100,100,100],[100,100,100,100,100],[0,0,0,0,0],[-100,-100,-100,-100,-100],[-100,-100,-100,-100,-100]])
    k1 = np.array([[100,100,100,100,100],[100,100,100,78,-32],[100,92,0,-92,-100],[32,-78,-100,-100,-100],[-100,-100,-100,-100,-100]])
    k2 = np.array([[100,100,100,32,-100],[100,100,92,-78,-100],[100,100,0,-100,-100],[100,78,-92,-100,-100],[100,-32,-100,-100,-100]])
    k3 = np.array([[-100,-100,0,100,100],[-100,-100,0,100,100],[-100,-100,0,100,100],[-100,-100,0,100,100],[-100,-100,0,100,100]])
    k4 = np.array([[-100,32,100,100,100],[-100,-78,92,100,100],[-100,-100,0,100,100],[-100,-100,-92,78,100],[-100,-100,-100,-32,100]])
    k5 = np.array([[100,100,100,100,100],[-32,78,100,100,100],[-100,-92,0,92,100],[-100,-100,-100,-78,32],[-100,-100,-100,-100,-100]])
    mask_list = [k0,k1,k2,k3,k4,k5]
    return max_Mask(copy_raw, size, threshold, mask_list)

def max_Mask(copy_raw, size, threshold, mask_list):
    #extended size
    X, Y = np.shape(copy_raw)
    new_img = copy_raw.copy()

    for i in range(size, X - size):
        for j in range(size, Y - size):
            grad = np.NINF
            for mask in mask_list:
                current_grad = np.sum(np.multiply(mask, copy_raw[i-size : i+1+size, j-size : j+1+size]))
                grad = max(grad, current_grad)
                if(grad >= threshold):
                    break
            new_img[i, j] = 255 * (grad < threshold)

    return new_img[size : X-size, size : Y-size]

    