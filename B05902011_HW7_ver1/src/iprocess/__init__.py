import numpy as np


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

def gray_dilation(img_array, kernel):
    img_dil = np.zeros(img_array.shape, dtype = 'uint8')
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            max_value = 0
            for points in kernel:
                p, q = points
                if (0 <= (i + p) <= (img_array.shape[0] - 1) and 0 <= (j + q) <= (img_array.shape[1] - 1)):
                    max_value = max(max_value, img_array[i + p, j + q])
            img_dil[i, j] = max_value
    return img_dil

def gray_erosion(img_array, kernel):
    img_ero = np.zeros(img_array.shape, dtype = 'uint8')
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            min_value = 255
            valid = True
            for points in kernel:
                p, q = points
                if (0 <= (i + p) <= (img_array.shape[0] - 1) and 0 <= (j + q) <= (img_array.shape[1] - 1)):
                    if(img_array[i + p, j + q] == 0) :
                        valid = False
                        break
                    else:
                        min_value = min(min_value, img_array[i + p, j + q])
                else:
                    valid = False
                    break
            if(valid):
                img_ero[i,j] = min_value
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


# https://github.com/sycLin/computer-vision-homework/

