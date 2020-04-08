import os
import glob

dataset_path = '..\\..\\datasets\\SDR_10BIT_sub'
video_path_list = glob.glob(os.path.join(dataset_path, '*'))

for video_path in video_path_list:
    print('Processing video', video_path)
    frame_path_list = glob.glob(os.path.join(video_path, '*'))
    # print(frame_path_list)
    patch_name_list = sorted(os.listdir(frame_path_list[0]))
    # print(patch_name_list)
    video_path, video_name = os.path.split(video_path)
    # print(video_path, video_name)
    for patch_name in patch_name_list:
        save_name = '{}_{}'.format(video_name, os.path.splitext(patch_name)[0])
        save_dir = os.path.join(video_path, save_name)
        if os.path.exists(save_dir):
            continue
        else:
            # os.makedirs(save_dir)
            print("Making dir {}".format(save_dir))
        for frame_path in frame_path_list:
            print('mv {} {}'.format(os.path.join(frame_path, patch_name), save_dir))
