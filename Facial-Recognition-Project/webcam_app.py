import cv2
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
import os
from torchvision import transforms
import torch


class WebcamApp:
    def __init__(self, root, model, device, class_names, image_size=(128, 128)):
        self.root = root
        self.root.title("Facial Recognition System")
        self.root.geometry("1000x800")

        # Model components
        self.model = model
        self.device = device
        self.class_names = class_names
        self.image_size = image_size

        # Setup transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Video capture
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # GUI Elements
        self.create_widgets()
        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # Video display frame
        self.video_frame = Frame(self.root)
        self.video_frame.pack(pady=20)

        self.label = Label(self.video_frame)
        self.label.pack()

        # Results display
        self.result_label = Label(self.root, text="", font=('Helvetica', 18))
        self.result_label.pack(pady=20)

        # Buttons frame
        self.button_frame = Frame(self.root)
        self.button_frame.pack(pady=20)

        self.capture_button = Button(self.button_frame, text="Capture & Recognize",
                                     command=self.capture_and_recognize, height=2, width=20)
        self.capture_button.pack(side=tk.LEFT, padx=10)

        self.quit_button = Button(self.button_frame, text="Exit",
                                  command=self.on_closing, height=2, width=20)
        self.quit_button.pack(side=tk.LEFT, padx=10)

    def update_frame(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = frame.copy()  # Store for processing
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.label.config(image=self.photo)
        self.root.after(10, self.update_frame)

    def capture_and_recognize(self):
        if hasattr(self, 'current_frame'):
            # Convert to PIL Image
            pil_image = Image.fromarray(self.current_frame)

            # Apply transformations
            if self.transform:
                image = self.transform(pil_image)
                image = image.unsqueeze(0).to(self.device)  # Add batch dimension

            # Run inference
            with torch.no_grad():
                outputs = self.model(image)
                _, predicted = torch.max(outputs.data, 1)
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

            # Get results
            label = self.class_names[predicted.item()]
            conf = confidence[predicted.item()].item()

            # Update GUI
            self.result_label.config(text=f"Recognized: {label}\nConfidence: {conf:.2f}%")

            # Save the captured image
            downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
            filename = os.path.join(downloads_path, "captured_recognition.jpg")
            cv2.imwrite(filename, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))

    def on_closing(self):
        self.vid.release()
        self.root.destroy()