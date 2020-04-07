#!/bin/sh
# export PATH="/home/medialab/workspace/hdd/zhen/ffmpeg:$PATH"
# echo $PATH
folder="/home/medialab/workspace/hdd/zhen/dataset/HDR_4k"
for file in ${folder}/*.mp4;
do
    echo $file
    # echo "/home/medialab/workspace/hdd/zhen/EDVR/datasets/SDR4k/"${file##*/}
    ffmpeg -i "$file" -vf zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p10le,zscale=s=3840x2160 -c:v libx265 -preset slow -crf 18 -c:a copy "/home/medialab/workspace/hdd/zhen/EDVR/datasets/SDR4k/"${file##*/}
    echo "Successfully convert "${file##*/}
done
