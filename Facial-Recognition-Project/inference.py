import torch
from torchvision import transforms
from PIL import Image
import cv2

class InferenceEngine:
    def __init__(self, model, device, transform=None, class_names=None):
        self.model = model
        self.device = device
        self.model.eval()
        self.transform = transform
        self.class_names = class_names if class_names is not None else []

    def predict_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, 1)
        if self.class_names:
            return self.class_names[predicted.item()]
        return predicted.item()

    def webcam_inference(self, capture_interval=10):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to open the webcam.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            if self.transform:
                image = self.transform(image)
            image = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(image)
                _, predicted = torch.max(outputs.data, 1)
            label = self.class_names[predicted.item()] if self.class_names else str(predicted.item())
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Webcam Inference", frame)
            if cv2.waitKey(capture_interval * 1000) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
