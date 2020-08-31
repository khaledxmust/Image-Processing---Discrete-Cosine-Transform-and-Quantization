import numpy as np
from PIL import Image
from copy import deepcopy
from matplotlib import pyplot as plt

path = 'lena_color.tiff'
image_tiff = Image.open(path)
rgb = np.array(image_tiff)

def RGB2YUV( rgb ): #RGB 2 YUV
    
    m = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                  [-0.14714119, -0.28886916,  0.43601035 ],
                  [ 0.61497538, -0.51496512, -0.10001026 ]])

    yuv = np.dot(rgb, m.T)
    return yuv
yuv = RGB2YUV( rgb ) 

Y = yuv[:, :, 0] #Y - Channel   
#%% Split Image to 8x8 Blocks

height, width = Y.shape
sliced = [] # new list for 8x8 sliced image 
currY = 0 #current Y index
n = 8 #dividing 8x8 blocks
for i in range(n, height+1, n):
    currX = 0 #current X index
    for j in range(n, width+1, n):
        sliced.append(Y[currY:i,currX:j]-np.ones((8,8))*128) #Normalizing 128 from all pixels
        currX = j
    currY = i

T = [np.float32(Y) for Y in sliced]

#%% DCT algorithm

D = deepcopy(sliced)
csum = 0.0
for w in range(4096):
    for i in range(8):
        for j in range(8):
            for x in range(8):
                for y in range(8): #Main algorithm below
                    csum += T[w][x][y]*np.cos((2*x+1)*i*np.pi/(2*n))*np.cos((2*y+1)*j*np.pi/(2*n))
            if (i==0 and j==0): k = 0.5
            elif ((i==0 and j) or (i and j==0)): k = 1/np.sqrt(2)
            else: k = 1
            D[w][i][j] = 2*csum*k/n
            csum = 0


#%% Quantization Matrix
            
xQuanmatrix = np.array([[16,11,10,16, 24, 40, 51, 61],
                       [12,12,14,19, 26, 58, 60, 55],
                       [14,13,16,24, 40, 57, 69,56 ],
                       [14,17,22,29 ,51, 87, 80, 62],
                       [18,22,37,56, 68,109,103, 77],
                       [24,35,55,64,81, 104,113, 92],
                       [49,64,78,87,103,121,120,101],
                       [72,92,95,98,112,100,103, 99]])

#%% Quantization
    
C = D / xQuanmatrix
for i in range(4096):
    for j in range(8):
        for x in range(8):
            C[i][j][x] = int(round(C[i][j][x]))

#%% Inverse Quantization & IDCT algorithm

R = C * xQuanmatrix
N = deepcopy(sliced)

csum = 0;
for w in range(4096):
    for i in range(8):
        for j in range(8):
            for x in range(8):
                for y in range(8):
                    k = 0.5
                    csum += k*R[w][x][y]*np.cos((2*i+1)*x*np.pi/(2*n))*np.cos((2*j+1)*y*np.pi/(2*n))
            N[w][i][j] = csum*2/n
            csum = 0

#%% ReBuilding the Image

row = 0
rowNcol = []
for j in range(int(width/n), len(N)+1, int(width/n)):
    N[row:j] = N[row:j]+np.ones((8,8))*128
    rowNcol.append(np.hstack((N[row:j])))
    row = j
YC = np.vstack((rowNcol))

#%% Plotting

output = [Y,YC]
titles = ['Original', 'Compressed']

for i in range(2):
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.title(titles[i])
    if i == 0:
        plt.imshow(output[i], cmap = 'gray')
    else:
        plt.imshow(output[i], cmap = 'gray')
plt.show()

#%% Computing PSNR

def PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

print('PSNR between two images is :',PSNR(Y,YC))



