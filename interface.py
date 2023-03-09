from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import keras.models as models
import pyttsx3

#flask instance
app = Flask(__name__)

#load model
model = models.load_model("best_model.h5")

#preprocess input image
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    image = np.expand_dims(image, axis=0)
    return image

#convert to speech
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()

#create route for home page
@app.route("/")
def home():
    return render_template("index.html")

