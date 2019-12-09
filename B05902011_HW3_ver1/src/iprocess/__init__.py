import numpy as np


def binarize(im , threshold):
    
    # threshold method, binarize to get binary image
    x,y = np.shape(im) 
    binary = np.zeros([x,y],dtype = 'uint8')

    for i in range(0, x):
        for j in range(0, y):
            if(im[i,j] >= threshold):
                binary[i,j] = 255
            else:
                binary[i,j] = 0

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
            if(b_array[x,y] == 255):
                label[x,y] = n_label
                n_label += 1

    # top-down and bottom-up labeling
    change = True
    while(change):
        change = False
        for x in range(0, X):
            for y in range(0, Y):
                if(b_array[x,y] == 255):        # 8 way strat
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
                if(b_array[x,y] == 255):        # 8 way strat
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
