import torch
from torchvision import transforms
from PIL import Image
import cv2

class InferenceEngine:
    """
    InferenceEngine performs inference on images using a trained PyTorch model.
    
    This class provides two main functionalities:
      1. Predict the class of a single image from a file path.
      2. Perform real-time inference using a webcam.
    
    Attributes:
        model (torch.nn.Module): The trained PyTorch model used for inference.
        device (torch.device): The device (CPU or GPU) on which the model runs.
        transform (torchvision.transforms.Compose): A transformation pipeline applied to input images.
        class_names (list): A list of class names corresponding to model output indices.
    """
    
    def __init__(self, model, device, transform=None, class_names=None):
        """
        Initializes the InferenceEngine.
        
        Args:
            model (torch.nn.Module): The trained model for inference.
            device (torch.device): Device on which to perform inference.
            transform (torchvision.transforms.Compose, optional): Preprocessing transforms for input images.
            class_names (list, optional): List of class names for interpreting model outputs.
        """
        self.model = model
        self.device = device
        self.model.eval()  # Set model to evaluation mode
        self.transform = transform
        self.class_names = class_names if class_names is not None else []

    def predict_image(self, image_path):
        """
        Predicts the class of an image provided by its file path.
        
        The method loads the image, applies any preprocessing transforms,
        and then uses the model to predict the class.
        
        Args:
            image_path (str): The path to the image file.
        
        Returns:
            str or int: The predicted class name if class_names are provided; 
                        otherwise, returns the predicted class index.
        """
        # Open the image and convert to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        # Add a batch dimension and move to device
        image = image.unsqueeze(0).to(self.device)
        
        # Perform inference without computing gradients
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, 1)
        
        # Return the corresponding class name if available
        if self.class_names:
            return self.class_names[predicted.item()]
        return predicted.item()

    def webcam_inference(self, capture_interval=10):
        """
        Performs real-time inference using the webcam.
        
        This method opens the default webcam, captures frames at specified intervals,
        processes each frame using the provided transformations, and then displays the
        frame with the predicted label overlayed. Press 'q' to quit the inference.
        
        Args:
            capture_interval (int, optional): Time in milliseconds between frame captures.
        """
        # Open the default webcam (device index 0)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to open the webcam.")
            return
        
        while True:
            # Capture a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break
            
            # Convert the captured frame from BGR (OpenCV default) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert the NumPy array to a PIL Image for transformation
            image = Image.fromarray(frame_rgb)
            if self.transform:
                image = self.transform(image)
            image = image.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Run inference on the frame
            with torch.no_grad():
                outputs = self.model(image)
                _, predicted = torch.max(outputs.data, 1)
            
            # Get the predicted label as a string if class_names are provided
            label = self.class_names[predicted.item()] if self.class_names else str(predicted.item())
            
            # Overlay the predicted label on the frame
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display the frame with the prediction
            cv2.imshow("Webcam Inference", frame)
            
            # Wait for the specified interval or until 'q' is pressed
            if cv2.waitKey(capture_interval) & 0xFF == ord('q'):
                break
        
        # Release resources and close the window
        cap.release()
        cv2.destroyAllWindows()
