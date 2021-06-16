#!/usr/bin/env python2.7
import sys, cv2, time, rospy
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt8
from cv_bridge import CvBridge

import Collision_Detection.src.CollisionDetector as CD


class ROS_Detector:
    def __init__(self):
        self.bridge = CvBridge()
        self.Detector = CD.CollisionDetector()
        self.timer = 0


    def broadcast_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        is_collision = self.Detector.process_frame(cv_image)
        if self.timer > 0: 
            self.timer -= 1
        else:
            if is_collision:
                self.publisher.publish(1)
                self.timer = 150
        

    def start_node(self):
        rospy.init_node('image_pub')
        rospy.loginfo('image_pub node started')
        self.publisher = rospy.Publisher("/tello/flip", UInt8, queue_size=10)
        rospy.Subscriber("/camera/image_raw",Image,self.broadcast_callback)        
        rospy.spin()


# bridge = CvBridge()
# Detector = CD.CollisionDetector()
# publisher = None

if __name__ == '__main__':
    try:
        RD = ROS_Detector()
        RD.start_node()
    except rospy.ROSInterruptException:
        pass
