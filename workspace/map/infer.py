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

from sptm import SPTM

bridge = CvBridge()
keyframes = []
keyframe_coordinates = []


class SPTMNode(Node):
    def __init__(self, name="MSCKF"):
        super().__init__(name)
        self.wait_count = 0
        self.pose_queue = Queue()
        self.img_queue = Queue()
        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.on_image,
            10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/ov_msckf/poseimu",
            self.on_pose,
            10)
        self.viewer = Viewer()
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.update_viewer)

    def on_image(self, msg):
        img_msg = bridge.imgmsg_to_cv2(msg)

        self.img_queue.put(img_msg)

    def on_pose(self, msg):
        self.pose_queue.put(msg.pose)

    def update_viewer(self):
        if not self.pose_queue.qsize() == 0 and not self.img_queue.qsize() == 0:
            pose = self.pose_queue.get()
            i = pose_msg_to_dict(pose)
            # for i in poses:
            q = list(i.get("orientation").values())
            q = np.array([q[-1], *q[1:]])
            p = np.array(list(i.get("position").values())) * 1e-4
            r = convert_quaternion_to_rotation_matrix(q)
            pose = convert_rotation_and_translation_to_transformation(r, p)

            self.viewer.update_pose(pose)
            img = self.img_queue.get()
            self.viewer.update_image(img)
            keyframes.append(img)
            keyframe_coordinates.append(p)
        else:
            print("+++++Waiting for Image+++++++")
            self.wait_count += 1
            if self.wait_count > 10:
                # self.viewer.close()
                raise SystemExit
            # executor.cancel()


from PIL import Image as PILImage


def main(args=None):
    rclpy.init(args=args)
    sptm = SPTMNode()
    image_paths = []
    try:
        rclpy.spin(sptm)
    except SystemExit:  # <--- process the exception
        rclpy.shutdown()
    for i, j in zip(keyframes, range(len(keyframes))):
        im = PILImage.fromarray(i)
        image_paths.append(f"keyframes/{j}.jpg")
        im.save(f"keyframes/{j}.jpg")
    np.save('keyframe_coordinates/a.npy', np.array(keyframe_coordinates, dtype=object), allow_pickle=True)
main()
