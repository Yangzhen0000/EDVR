"""A multi-thread tool to crop large images to sub-images for faster IO."""
import os
import os.path as osp
import sys
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils.util import ProgressBar  # noqa: E402
import data.util as data_util  # noqa: E402


def main():
    mode = 'pair'  # single (one input folder) | pair (extract corresponding GT and LR pairs)
    opt = {}
    opt['n_thread'] = 1
    opt['compression_level'] = 3  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.
    if mode == 'single':
        opt['input_folder'] = '../../datasets/DIV2K/DIV2K_train_HR'
        opt['save_folder'] = '../../datasets/DIV2K/DIV2K800_sub'
        opt['crop_sz'] = 480  # the size of each sub-image
        opt['step'] = 240  # step of the sliding crop window
        opt['thres_sz'] = 48  # size threshold
        extract_single(opt)
    elif mode == 'pair':
        GT_folder = '../../datasets/SDR4k/train/SDR_10BIT'  # '../../datasets/DIV2K/DIV2K_train_HR'
        LR_folder = '../../datasets/SDR4k/train/SDR_4BIT'   # '../../datasets/DIV2K/DIV2K_train_LR_bicubic/X4'
        save_GT_folder = '../../datasets/SDR4k/train/SDR_10BIT_sub'  # '../../datasets/DIV2K/DIV2K800_sub'
        save_LR_folder = '../../datasets/SDR4k/train/SDR_4BIT_sub'   # '../../datasets/DIV2K/DIV2K800_sub_bicLRx4'
        scale_ratio = 1  # 4
        crop_sz = 480  # the size of each sub-image (GT)
        step = 240  # step of the sliding crop window (GT)
        thres_sz = 48  # size threshold
        ########################################################################
        # check that all the GT and LR images have correct scale ratio
        # img_GT_list = data_util._get_paths_from_images(GT_folder)
        # img_LR_list = data_util._get_paths_from_images(LR_folder)
        # assert len(img_GT_list) == len(img_LR_list), 'different length of GT_folder and LR_folder.'
        # for path_GT, path_LR in zip(img_GT_list, img_LR_list):
        #     img_GT = Image.open(path_GT)
        #     img_LR = Image.open(path_LR)
        #     w_GT, h_GT = img_GT.size
        #     w_LR, h_LR = img_LR.size
        #     assert w_GT / w_LR == scale_ratio, 'GT width [{:d}] is not {:d}X as LR weight [{:d}] for {:s}.'.format(  # noqa: E501
        #         w_GT, scale_ratio, w_LR, path_GT)
        #     assert w_GT / w_LR == scale_ratio, 'GT width [{:d}] is not {:d}X as LR weight [{:d}] for {:s}.'.format(  # noqa: E501
        #         w_GT, scale_ratio, w_LR, path_GT)
        # # check crop size, step and threshold size
        # assert crop_sz % scale_ratio == 0, 'crop size is not {:d}X multiplication.'.format(
        #     scale_ratio)
        # assert step % scale_ratio == 0, 'step is not {:d}X multiplication.'.format(scale_ratio)
        # assert thres_sz % scale_ratio == 0, 'thres_sz is not {:d}X multiplication.'.format(
        #     scale_ratio)
        print('process GT...')
        opt['input_folder'] = GT_folder
        opt['save_folder'] = save_GT_folder
        opt['crop_sz'] = crop_sz
        opt['step'] = step
        opt['thres_sz'] = thres_sz
        extract_single(opt)  # all GT videos
        print('process LR...')
        opt['input_folder'] = LR_folder
        opt['save_folder'] = save_LR_folder
        opt['crop_sz'] = crop_sz // scale_ratio
        opt['step'] = step // scale_ratio
        opt['thres_sz'] = thres_sz // scale_ratio
        extract_single(opt)  # all LQ videos
        assert len(data_util._get_paths_from_images(save_GT_folder)) == len(
            data_util._get_paths_from_images(
                save_LR_folder)), 'different length of save_GT_folder and save_LR_folder.'
    else:
        raise ValueError('Wrong mode.')


def rm_border(img_file):
    image = cv2.imread(img_file)

    b = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
    binary_image = b[1]
    binary_image = cv2.cvtColor(binary_image,cv2.COLOR_BGR2GRAY)
    # rint(binary_image.shape)

    x = binary_image.shape[0]
    y = binary_image.shape[1]
    edges_x = []
    edges_y = []

    for i in range(x):
        for j in range(y):
            if binary_image[i][j] == 255:
                edges_x.append(i)
                edges_y.append(j)

    left = min(edges_x)
    right = max(edges_x)

    bottom = min(edges_y)
    top = max(edges_y)

    # print(left, right, bottom, top)
    return left, right, bottom, top


def extract_single(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        # print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)
    
    # img_list = data_util._get_paths_from_images(input_folder)
    video_list, _ = data_util.get_video_paths(input_folder)
    # def update(arg):
    #     pbar.update(arg)

    pbar = ProgressBar(len(video_list))

    # pool = Pool(opt['n_thread'])
    # for path in img_list:
    #     pool.apply_async(worker, args=(path, opt), callback=update)
    # pool.close()
    # pool.join()
    # print('All subprocesses done.')
    crop_sz = opt['crop_sz']
    step = opt['step']
    thres_sz = opt['thres_sz']
    for video_path in video_list:
        video_name = osp.basename(video_path)
        left, right, bottom, top = rm_border(osp.join(video_path, "001.png"))  # cut for all images in one video
        
        img_list = osp.join(video_path, os.listdir(video_path))
        for img_path in sorted(img_list):
            img_name = osp.basename(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = img[left:right, bottom:top, :]

            n_channels = len(img.shape)
            if n_channels == 2:
                h, w = img.shape
            elif n_channels == 3:
               h, w, c = img.shape
            else:
                raise ValueError('Wrong image shape - {}'.format(n_channels))

            h_space = np.arange(0, h - crop_sz + 1, step)
            if h - (h_space[-1] + crop_sz) > thres_sz:
               h_space = np.append(h_space, h - crop_sz)
            w_space = np.arange(0, w - crop_sz + 1, step)
            if w - (w_space[-1] + crop_sz) > thres_sz:
                w_space = np.append(w_space, w - crop_sz)

            index = 0
            save_folder = osp.join(opt['save_folder'], video_name, osp.splitext(img_name)[0])
            # print("save_folder", save_folder)
            if not osp.exists(save_folder):
                os.makedirs(save_folder)
                # print('mkdir [{:s}] ...'.format(save_folder))
            else:
                print('Folder [{:s}] already exists. Exit...'.format(save_folder))
                sys.exit(1)
            for x in h_space:
                for y in w_space:
                    index += 1
                    if n_channels == 2:
                        crop_img = img[x:x + crop_sz, y:y + crop_sz]
                    else:
                        crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
                    crop_img = np.ascontiguousarray(crop_img)

                    cv2.imwrite(
                        osp.join(save_folder, 'p{:03d}.png'.format(index)), crop_img,
                        [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
                    # print('Write patch {:s}'.format(osp.join(save_folder, 'p{:03d}.png'.format(index))))
            print('Processing {:s} ...'.format(img_name))
        pbar.update(1)


def worker(path, opt):
    crop_sz = opt['crop_sz']
    step = opt['step']
    thres_sz = opt['thres_sz']
    img_name = osp.basename(path)
    video_name = osp.basename(osp.dirname(path))
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    save_folder = osp.join(opt['save_folder'], video_name, osp.splitext(img_name)[0])
    print("save_folder", save_folder)
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)
    for x in h_space:
        for y in w_space:
            index += 1
            # if n_channels == 2:
            #     crop_img = img[x:x + crop_sz, y:y + crop_sz]
            # else:
            #     crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            # crop_img = np.ascontiguousarray(crop_img)

            # cv2.imwrite(
            #     osp.join(save_folder, 'p{:03d}.png'.format(index)), crop_img,
            #     [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
            print('Write patch {:s}'.format(osp.join(save_folder, 'p{:03d}.png'.format(index))))
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
