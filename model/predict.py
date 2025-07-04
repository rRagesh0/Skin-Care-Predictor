import torch
from PIL import Image
from torchvision import models,transforms
import torch.nn as nn
from torchvision.models import resnet18,ResNet18_Weights
#from your_model import MultiTaskSkinCNN  # Import your model class

class MultiTaskSkinCNN(nn.Module):
    def __init__(self):
        super(MultiTaskSkinCNN, self).__init__()
        # Load pre-trained ResNet18
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.shared_cnn = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last FC layer
        num_features = resnet.fc.in_features
        
        # Add dropout and task-specific heads
        self.dropout = nn.Dropout(0.5)
        self.skin_type_head = nn.Linear(num_features, 3)  # 3 classes for skin type
        self.skin_condition_head = nn.Linear(num_features, 4)  # 4 classes for skin condition

    def forward(self, x):
        features = self.shared_cnn(x).view(x.size(0), -1)  # Flatten features
        features = self.dropout(features)
        skin_type_output = self.skin_type_head(features)
        skin_condition_output = self.skin_condition_head(features)
        return skin_type_output, skin_condition_output

# Load your trained model
model = MultiTaskSkinCNN()
model.load_state_dict(torch.load('model/model_2.pth', map_location=torch.device('cuda')))
model.eval()

def predict_skin_type_condition(image_path):
    # Preprocess the image
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        skin_type_output, skin_condition_output = model(image)
        skin_type = torch.argmax(skin_type_output, dim=1).item()
        skin_condition = torch.argmax(skin_condition_output, dim=1).item()

    # Map predictions to meaningful labels
    skin_type_labels = {0: "Dry", 1: "Oily", 2: "Normal"}
    skin_condition_labels = {0: "Acne", 1: "Redness", 2: "Darkcircle", 3: "Wrinkle"}

    return skin_type_labels[skin_type], skin_condition_labels[skin_condition]
