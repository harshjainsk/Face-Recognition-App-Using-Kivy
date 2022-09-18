# import dependencies

# """This is for the application layout"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# """ this is for the ux components """

from kivy.uix.image import Image  # used for real-time webcam feed
from kivy.uix.button import Button
from kivy.uix.label import Label


# """used for other functionalities"""
from kivy.clock import Clock  # used for continous real-time swap
from kivy.graphics.texture import Texture
from kivy.logger import Logger


# import other dependencies
import tensorflow as tf
import cv2
import numpy as np
import os
from layers import L1Dist