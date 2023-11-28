import queue

import numpy as np

from ros_realsense import Subscriber, pose_msg_to_dict
import rclpy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Imu, Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from queue import Queue
from utils.pango_viewer import Viewer
from utils.transformations import convert_quaternion_to_rotation_matrix, \
    convert_rotation_and_translation_to_transformation
from cv_bridge import CvBridge
import glob
import os
from sptm import SPTM

bridge = CvBridge()
keyframes = []
keyframe_coordinates = []

from PIL import Image as PILImage

def number(filename):
    return int(filename[10:-4])
def main(args=None):
    rclpy.init(args=args)
    episodes = []

    for filename in sorted(glob.glob('keyframes/*.jpg'),key=number):
        episodes.append(filename)
    # np.save('keyframe_coordinates/a.npy', np.array(keyframe_coordinates, dtype=object), allow_pickle=True)
    b = np.load('keyframe_coordinates/a.npy', allow_pickle=True)
    sptm = SPTM()
    sptm.set_shortcuts_cache_file("shortcut")
    sptm.build_graph(episodes[:10], b[:10])


main()
