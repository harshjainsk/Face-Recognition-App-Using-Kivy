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
        self.img1 = Image(size_hint=(1, 0.8))
        self.button = Button(text='Verify', size_hint=(1, 0.1))
        self.verification = Label(text='Verification uninitalised', size_hint=(1, 0.1))

        # Add items to layout
        layout = BoxLayout(orientation = 'vertical')
        layout.add_widget(self.img1)
        layout.add_widget(self.button)
        layout.add_widget(self.verification)

        return layout


if __name__ == '__main__':
    CamApp().run()
