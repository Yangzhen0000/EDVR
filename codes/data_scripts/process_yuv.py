import math
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import cv2

def zeropadding(filename, srcbitdepth, dstbitdepth):
    if srcbitdepth == 10:
        image = cv2.imread(filename, 3)
        print(image.min(), image.max())
        image = image // np.power(2, srcbitdepth-dstbitdepth) * np.power(2, srcbitdepth-dstbitdepth)
        print(image.min(), image.max())
    newname = filename[:-4] + "_zp.png" 
    cv2.imwrite(newname, image)

def readyuv420(filename, bitdepth, height, width, startfrm, save=True, show=False):
    fp = open(filename, 'rb')
    
    framesize = height * width * 3 // 2
    h_h = height // 2
    h_w = width // 2

    fp.seek(0, 2) # locate the end
    ps = fp.tell() # return current location
    numfrm = ps // framesize
    
    bytesPerPixel = math.ceil(bitdepth / 8)
    seekPixels = startfrm * framesize
    fp.seek(bytesPerPixel * seekPixels)
    
    bytes2num = partial(int.from_bytes, byteorder='little', signed=False)

    if bitdepth == 8:
        Y = np.zeros((numfrm, height, width), np.uint8)
        U = np.zeros((numfrm, h_h, h_w), np.uint8)
        V = np.zeros((numfrm, h_h, h_w), np.uint8)
    elif bitdepth == 10:
        Y = np.zeros((numfrm, height, width), np.uint16)
        U = np.zeros((numfrm, h_h, h_w), np.uint16)
        V = np.zeros((numfrm, h_h, h_w), np.uint16)
    #for i in range(numfrm - startfrm):
    for i in range(1):
        for m in range(height):
            for n in range(width):
                if bitdepth == 8:
                    Y[i, m, n] = np.uint8(bytes2num(fp.read(1)))
                elif bitdepth == 10:
                    Y[i, m, n] = np.uint16(bytes2num(fp.read(2)))
        for m in range(h_h):
            for n in range(h_w):
                if bitdepth == 8:
                    U[i, m, n] = np.uint8(bytes2num(fp.read(1)))
                elif bitdepth == 10:
                    U[i, m, n] = np.uint16(bytes2num(fp.read(2)))
        for m in range(h_h):
            for n in range(h_w):
                if bitdepth == 8:
                    V[i, m, n] = np.uint8(bytes2num(fp.read(1)))
                elif bitdepth == 10:
                    V[i, m, n] = np.uint16(bytes2num(fp.read(2)))
        plt.ion()
        if show:
            print(i)
            plt.subplot(131)
            plt.imshow(Y[i, :, :], cmap='gray')
            plt.subplot(132)
            plt.imshow(U[i, :, :], cmap='gray')
            plt.subplot(133)
            plt.imshow(V[i, :, :], cmap='gray')
            plt.show()
            plt.pause(1)
            #plt.pause(0.001)
        if save:
            #img = np.concatenate((Yt.reshape(-1), Ut.reshape(-1), Vt.reshape(-1)))
            #img = img.reshape(height * 3 // 2, width)
            #rgb_img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB_I420)
            cv2.imwrite("yuv2rgb/%03d.png"%(i+1), rgb_img)
            print("Extract frame %d" % (i+1))
    fp.close()
    print("job done!")
    return None

if __name__ == '__main__':
    readyuv420("Bosphorus_3840x2160_120fps_420_10bit_YUV.yuv", 10, 2160, 3840, 0, False, True)
