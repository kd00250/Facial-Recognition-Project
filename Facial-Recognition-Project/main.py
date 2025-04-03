# Add the project folder to the Python path if needed
import sys
sys.path.append("project")

# Import your modules
from data_preprocessing import extract_and_arrange
from data_loading import get_data_loaders
from model_training import ModelTrainer
from inference import InferenceEngine

# Run the preprocessing step (if not done already)
extract_and_arrange(zip_file="Faces.zip", extracted_folder="Faces_unorganized", structured_folder="Faces")

# Load your data
train_dataset, val_dataset, train_loader, val_loader, class_names = get_data_loaders(data_dir="Faces", batch_size=32, image_size=(128,128))

# Assume you have defined your model, device, criterion, optimizer, etc.
# For example:
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your model here or load a pre-trained model
# For demonstration, here we create a simple model:
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(128*128*3, 100),
    nn.ReLU(),
    nn.Linear(100, len(class_names))
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Initialize your trainer and start training
trainer = ModelTrainer(model, train_loader, val_loader, train_dataset, val_dataset, device, criterion, optimizer, scheduler)
trainer.train(num_epochs=10, early_stop_patience=5)

# (Optional) Use inference
inference_engine = InferenceEngine(model, device, transform=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
]), class_names=class_names)
prediction = inference_engine.predict_image("path/to/your/image.jpg")
print("Predicted class:", prediction)
