import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import os

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Capture")
        self.root.geometry("800x600")

        self.video_source = 0  # Default webcam
        self.vid = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)

        self.label = Label(root)
        self.label.pack()

        self.capture_button = Button(root, text="Capture Image", command=self.capture_image)
        self.capture_button.pack()

        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_frame(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.label.config(image=self.photo)
        self.root.after(10, self.update_frame)

    def capture_image(self):
        ret, frame = self.vid.read()
        if ret:
            # Get the path to the Downloads folder
            downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')

            # Create filename with full path
            filename = os.path.join(downloads_path, "captured_image.jpg")
            cv2.imwrite(filename, frame)

    def on_closing(self):
        self.vid.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()