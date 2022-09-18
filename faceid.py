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
        self.button = Button(text='Verify', on_press=self.verify, size_hint=(1, 0.1))
        self.verification_label = Label(text='Verification uninitalised', size_hint=(1, 0.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load the siamese model
        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist})

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

    # adding preprocess function to scale and resize
    def preprocess(self, file_path):
        """
            In this function, we first read in the image and then load the image.
            After this we resize the image into 100px * 100px * 3 color channels.
            Last line helps in scaling.
        """
        byte_image = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_image)  # loading of image
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img

    # Adding verify function
    def verify(self, *args):
        """
            detection_threshold means the value above which a prediction is considered positive
            verification_threshold is proportion of positive predictions by total positive samples
        """

        # defining detection and verification threshold
        detection_threshold = 0.7
        verification_threshold = 0.7

        # Capture image for input
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]
        cv2.imwrite(SAVE_PATH, frame)

        # Build results array
        results = []

        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            # Make Predictions
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        # Detection Threshold: Metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)

        # Verification Threshold: Proportion of positive predictions / total positive samples
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        # set verification text
        self.verification_label.text = 'Verified' if verified == True else 'Unverfied'


        # log information into console
        Logger.info(verified)
        Logger.info(verification)
        Logger.info(np.sum(np.array(results) > 0.5))

        return results, verified


if __name__ == '__main__':
    CamApp().run()
