# import torch
# import torch.nn as nn

# class CNN_Model(nn.Module):
#     def __init__(self, num_classes, in_channels=3):
#         super(CNN_Model, self).__init__()
#         # Convolutional Block 1: 128x128 -> 64x64
#         self.block1 = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         # Convolutional Block 2: 64x64 -> 32x32
#         self.block2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         # Convolutional Block 3: 32x32 -> 16x16
#         self.block3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         # Convolutional Block 4: 16x16 -> 8x8
#         self.block4 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         # Fully connected classifier
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256 * 8 * 8, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )
    
#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.classifier(x)
#         return x

import torch
import torch.nn as nn

class LightCNN(nn.Module):
    """
    LightCNN is a lightweight convolutional neural network designed for face recognition.
    
    The architecture consists of four convolutional blocks followed by a fully connected classifier.
    Each convolutional block applies convolution, batch normalization, ReLU activation, and max pooling,
    which progressively reduces the spatial dimensions while increasing the number of feature maps.
    
    The network expects input images of size 128x128 with 3 channels and outputs logits corresponding to the number of classes.
    
    Args:
        num_classes (int): The number of output classes.
        in_channels (int): The number of input channels (default is 3 for RGB images).
    """
    def __init__(self, num_classes, in_channels=3):
        super(LightCNN, self).__init__()
        
        # Block 1: Processes the input image (128x128) and outputs feature maps of size 64x64.
        # Uses 8 filters.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),  # Convolution: maintains spatial dimensions.
            nn.BatchNorm2d(8),                                   # Batch normalization for stability.
            nn.ReLU(inplace=True),                               # ReLU activation introduces non-linearity.
            nn.MaxPool2d(kernel_size=2, stride=2)                 # Max pooling reduces spatial dimensions by 2.
        )
        
        # Block 2: Further processes the output from Block 1 (64x64) and reduces it to 32x32.
        # Uses 16 filters.
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: Processes the feature maps (32x32) from Block 2, reducing them to 16x16.
        # Uses 32 filters.
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4: Processes the feature maps (16x16) from Block 3, reducing them to 8x8.
        # Uses 64 filters.
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier: Flattens the 8x8 feature maps from Block 4 and passes them through fully connected layers.
        # The fully connected layers compress the features to 128 neurons, apply dropout for regularization,
        # and finally output logits corresponding to the number of classes.
        self.fc = nn.Sequential(
            nn.Flatten(),                                  # Flatten the 3D feature maps into a 1D feature vector.
            nn.Linear(64 * 8 * 8, 128),                     # Fully connected layer to reduce feature size.
            nn.ReLU(inplace=True),                         # ReLU activation.
            nn.Dropout(0.5),                               # Dropout layer to reduce overfitting.
            nn.Linear(128, num_classes)                     # Output layer: one neuron per class.
        )
        
    def forward(self, x):
        """
        Defines the forward pass of the network.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, in_channels, 128, 128].
        
        Returns:
            Tensor: Output logits of shape [batch_size, num_classes].
        """
        x = self.conv1(x)  # Pass through Block 1
        x = self.conv2(x)  # Pass through Block 2
        x = self.conv3(x)  # Pass through Block 3
        x = self.conv4(x)  # Pass through Block 4
        x = self.fc(x)     # Classify the flattened features
        return x
