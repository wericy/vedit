#!/bin/bash
#320
#870
#ffmpeg -i "Video 2548 - test.wmv" -b:v 10000 -an -vf "crop=320:720:0:0" "out1.avi"
total_width=1280
total_height=720
v1_width=$1
v2_width=$(($2 - $1))
v3_width=$(($total_width - $2 ))
start_v1=0
start_v2=$1
start_v3=$2
ffmpeg -i "Video 2548 - test.wmv" -filter_complex \
"[0:v]crop=$v1_width:$total_height:0:0[out1];[0:v]crop=$v2_width:$total_height:$start_v2:0[out2];[0:v]crop=$v3_width:$total_height:$start_v3:0[out3]" \
-map [out1] -b:v 800000 out1.avi \
-map [out2] -b:v 800000 out2.avi \
-map [out3] -b:v 800000 out3.avi