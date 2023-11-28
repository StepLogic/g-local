#!/usr/bin/env bash
source source /opt/ros/humble/setup.bash
source /catkin_ws/devel/setup.bash
if [ $# -gt 0 ];then
    # If we passed a command, run it
    exec "$@"
else
    /bin/bash
fi