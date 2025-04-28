# Facial-Recognition-Project
ML application that Identifies people based off of their face

# How to run 
# In google collab 

from google.colab import files
uploaded = files.upload()  # Use the file selector to upload Facial-Recognition-Project.zip

!unzip -q Facial-Recognition-Project.zip

!python Facial-Recognition-Project/main.py

For Running in Pycharm 
for cv2 import need to import opencv-python instead in the python interpreter
every other import can be imported as pycharm default by hovering over it and install
