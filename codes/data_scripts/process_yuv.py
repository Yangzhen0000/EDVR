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

def scale_bitdepth(img, srcbitdepth, dstbitdepth):
    if dstbitdepth >= srcbitdepth:
        return img*np.power(2, dstbitdepth-srcbitdepth)
    else:
        return img // np.power(2, srcbitdepth-dstbitdepth)

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
    for i in range(numfrm - startfrm):
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
        #plt.ion()
        #plt.ioff()
        if show:
            print(i)
            plt.subplot(131)
            plt.imshow(Y[i, :, :], cmap='gray')
            plt.subplot(132)
            plt.imshow(U[i, :, :], cmap='gray')
            plt.subplot(133)
            plt.imshow(V[i, :, :], cmap='gray')
            plt.show()
            #plt.pause(1)
            #plt.pause(0.001)
        if save:
            if bitdepth == 10:
                y = scale_bitdepth(Y[i, :, :], 10, 16)  # scale is necessary for display, otherwise image is dark
                u = scale_bitdepth(U[i, :, :], 10, 16)
                cv2.imwrite("../data/yuv2rgb/%03d_Y.png" % (i+1), y)
                cv2.imwrite("../data/yuv2rgb/%03d_U.png" % (i+1), u)
                yuv444 = yuv420to444(Y[i, :, :], U[i, :, :], V[i, :, :])  # CV_U16 only supported by YUV444 to BGR
                yuv = scale_bitdepth(yuv444, 10, 16)  # scale bit-depth for display in the YUV space
                rgb_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                cv2.imwrite("../data/yuv2rgb/%03d.png"%(i+1), rgb_img)
                print("Extract frame %d" % (i+1))
            elif bitdepth == 8:
                pass
                #img = np.concatenate((Yt.reshape(-1), Ut.reshape(-1), Vt.reshape(-1)))
                #img = img.reshape(height * 3 // 2, width)
                #rgb_img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB_I420)
    fp.close()
    print("job done!")
    return None

def yuv420to444(y420, u420, v420):
    height, width = y420.shape
    yuv444 = np.zeros(shape=(height, width, 3), dtype=y420.dtype)
    for m in range(height):
        for n in range(width):
            yuv444[m, n, 0] = y420[m, n]
    for m in range(0, height, 2):
        for n in range(0, width, 2):
            yuv444[m, n, 1] = u420[m//2, n//2]
            yuv444[m, n+1, 1] = u420[m//2, n//2]
            yuv444[m+1, n, 1] = u420[m//2, n//2]
            yuv444[m+1, n+1, 1] = u420[m//2, n//2]

            yuv444[m, n, 2] = v420[m//2, n//2]
            yuv444[m, n+1, 2] = v420[m//2, n//2]
            yuv444[m+1, n, 2] = v420[m//2, n//2]
            yuv444[m+1, n+1, 2] = v420[m//2, n//2]
    return yuv444


if __name__ == '__main__':
    readyuv420("../data/Bosphorus_3840x2160_120fps_420_10bit_YUV.yuv", 10, 2160, 3840, 0, True, True)
