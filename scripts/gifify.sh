#!/usr/bin/env bash
# cd to scripts directory to run this script it will create gif from the frame plots
cd /home/alien/catkin_ws/src/cloud_map/scripts/
convert -delay 200 -loop 0 -adjoin /home/alien/catkin_ws/src/cloud_map/scripts/framesA/A_*_joint.png /home/alien/catkin_ws/src/cloud_map/scripts/gifs/out_A_joint.gif
convert -delay 200 -loop 0 -adjoin /home/alien/catkin_ws/src/cloud_map/scripts/framesA/A_to_B*.png /home/alien/catkin_ws/src/cloud_map/scripts/gifs/out_A_to_B.gif
convert -delay 200 -loop 0 -adjoin /home/alien/catkin_ws/src/cloud_map/scripts/framesA/A_to_C*.png /home/alien/catkin_ws/src/cloud_map/scripts/gifs/out_A_to_C.gif
convert -delay 200 -loop 0 -adjoin /home/alien/catkin_ws/src/cloud_map/scripts/framesA/A_from_B*.png /home/alien/catkin_ws/src/cloud_map/scripts/gifs/out_A_from_B.gif
convert -delay 200 -loop 0 -adjoin /home/alien/catkin_ws/src/cloud_map/scripts/framesA/A_from_C*.png /home/alien/catkin_ws/src/cloud_map/scripts/gifs/out_A_from_C.gif
convert -delay 200 -loop 0 -adjoin /home/alien/catkin_ws/src/cloud_map/scripts/framesB/B_*_joint.png /home/alien/catkin_ws/src/cloud_map/scripts/gifs/out_B_joint.gif
convert -delay 200 -loop 0 -adjoin /home/alien/catkin_ws/src/cloud_map/scripts/framesB/B_to_A*.png /home/alien/catkin_ws/src/cloud_map/scripts/gifs/out_B_to_A.gif
convert -delay 200 -loop 0 -adjoin /home/alien/catkin_ws/src/cloud_map/scripts/framesB/B_to_C*.png /home/alien/catkin_ws/src/cloud_map/scripts/gifs/out_B_to_C.gif
convert -delay 200 -loop 0 -adjoin /home/alien/catkin_ws/src/cloud_map/scripts/framesB/B_from_A*.png /home/alien/catkin_ws/src/cloud_map/scripts/gifs/out_B_from_A.gif
convert -delay 200 -loop 0 -adjoin /home/alien/catkin_ws/src/cloud_map/scripts/framesB/B_from_C*.png /home/alien/catkin_ws/src/cloud_map/scripts/gifs/out_B_from_C.gif
convert -delay 200 -loop 0 -adjoin /home/alien/catkin_ws/src/cloud_map/scripts/framesC/C_*_joint.png /home/alien/catkin_ws/src/cloud_map/scripts/gifs/out_C_joint.gif
convert -delay 200 -loop 0 -adjoin /home/alien/catkin_ws/src/cloud_map/scripts/framesC/C_to_A*.png /home/alien/catkin_ws/src/cloud_map/scripts/gifs/out_C_to_A.gif
convert -delay 200 -loop 0 -adjoin /home/alien/catkin_ws/src/cloud_map/scripts/framesC/C_to_B*.png /home/alien/catkin_ws/src/cloud_map/scripts/gifs/out_C_to_B.gif
convert -delay 200 -loop 0 -adjoin /home/alien/catkin_ws/src/cloud_map/scripts/framesC/C_from_A*.png /home/alien/catkin_ws/src/cloud_map/scripts/gifs/out_B_from_A.gif
convert -delay 200 -loop 0 -adjoin /home/alien/catkin_ws/src/cloud_map/scripts/framesC/C_from_B*.png /home/alien/catkin_ws/src/cloud_map/scripts/gifs/out_C_from_B.gif
# video
avconv -pattern_type glob -i "/home/alien/catkin_ws/src/dummy_cloud_map/scripts/framesB/B_*_joint.png" -r 120 /home/alien/catkin_ws/src/dummy_cloud_map/scripts/gifs/out_B_joint.mkv
