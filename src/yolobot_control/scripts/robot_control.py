#!/usr/bin/python3
import os
import sys

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header, String, Int32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy, Image, CompressedImage
from cv_bridge import CvBridge
import cv2

import threading

vel_msg = Twist()

class Commander(Node):
    def __init__(self):
        super().__init__('commander')
        self.publisher_ = self.create_publisher(Twist, '/yolobot/cmd_vel', 10)
        self.timer_period = 0.02
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        self.publisher_.publish(vel_msg)

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/rgb_cam/image_raw',
            self.camera_callback,
            10)
        self.subscription

    def camera_callback(self, data):
        bridge = CvBridge()
        try:
            cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # Kırmızı rengi algılamak için HSV renk uzayına dönüştürme
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Kırmızı renge karşılık gelen HSV aralığını tanımlama
        lower_red = (0, 100, 100)
        upper_red = (10, 255, 255)

        # Kırmızı renk maskeleme işlemi
        mask = cv2.inRange(hsv_image, lower_red, upper_red)

        # Maskeden elde edilen alanın piksel sayısını hesaplama
        red_pixels = cv2.countNonZero(mask)

        # Eğer kırmızı pikseller varsa geri gitme hareketi yap
        if red_pixels > 0:
            vel_msg.linear.x = -0.2  # Geri gitme hızı ayarlanabilir
        else:
            vel_msg.linear.x = 0.2  # İleri gitme hızı ayarlanabilir

        # Kamerayı görselleştirme
        cv2.imshow("Camera", cv_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    rclpy.init(args=None)
    
    commander = Commander()
    camera_subscriber = CameraSubscriber()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(commander)
    executor.add_node(camera_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        rclpy.spin(commander)
    except KeyboardInterrupt:
        pass
    
    rclpy.shutdown()
    executor_thread.join()



