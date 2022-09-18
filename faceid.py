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

"""Build app and layout"""


class CamApp(App):

    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint=(1, 0.8))
        self.button = Button(text='Verify', size_hint=(1, 0.1))
        self.verification = Label(text='Verification uninitalised', size_hint=(1, 0.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification)

        # setup video capture
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    # Run continuously to update the webcam feed
    def update(self, *args):
        ret, frame = self.capture.read()

        # cut down frame to 250px*250px
        frame = frame[120:120 + 250, 200:200 + 250, :]

        # flip horizontally and convert the image to texture

        """
            Here we are converting our raw OpenCV image array
            to a texture for rendering in our app. Then, we 
            set out image equal to that texture
        """
        buf = cv2.flip(frame, 0).tostring()  # flips the image horizontally
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture


if __name__ == '__main__':
    CamApp().run()
