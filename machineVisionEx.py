import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os, random
import pandas as pd

## Imports for plotting
import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set_theme('notebook', style='whitegrid')

## Progress bar
from tqdm.notebook import tqdm

import torch
torch.manual_seed(42) # Setting the seed

print("Using torch", torch.__version__)

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Create a mapping from string labels to integer labels
        self.label_to_index = {label: idx for idx, label in enumerate(self.annotations['label'].unique())}
        # Create a reverse mapping for integer labels back to string labels
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")
        label_str = self.annotations.iloc[index, 1]
        # Convert string label to integer
        label = self.label_to_index[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label

# Paths to the data
csv_file = './data/newspaper_images/ads_data/ads_upsampled_no_index.csv'  # Path to your CSV file
img_dir = './data/newspaper_images/ads_data/images'       # Directory with all the images

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224x224 without preserving aspect ratio, i.e., squishing the image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # This is the mean and std deviation of the ImageNet dataset, in view of using a pre-trained ResNet model
])

# Create the dataset
dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)

# Define the train-validation split
train_size = int(0.8 * len(dataset))  # 80% of the data for training
val_size = len(dataset) - train_size  # Remaining 20% for validation

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for train and validation datasets
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Example: Iterate through the training data
for images, labels in train_loader:
    print(f"Train - images shape: {images.shape}, labels shape: {labels.shape}")
    break

# Example: Iterate through the validation data
for images, labels in val_loader:
    print(f"Validation - images shape: {images.shape}, labels shape: {labels.shape}")
    break

# Example: Plot one image with its label
def plot_example_image(dataset, index):
    image, label = dataset[index]
    
    # Unnormalize the image (this should be done before converting to NumPy)
    image = image * torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2) + torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
    
    # Convert the image tensor to a NumPy array for plotting
    image = image.permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC
    image = image.clip(0, 1)  # Clip values to ensure they are between 0 and 1

    # Get the label string from the index
    label_str = dataset.index_to_label[label]

    # Plotting the image with its label
    plt.imshow(image)
    plt.title(f"Label: {label_str}")
    plt.axis('off')  # Turn off the axis
    plt.show()

# Plot an example image from the dataset (e.g., the first image)
plot_example_image(dataset, index=379)


# Load pre-trained ResNet model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) #implement of architecture and rest weights

# Freeze the layers 
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer to a new one, and match the number of classes
num_features = model.fc.in_features
num_classes = len(dataset.label_to_index)
model.fc = nn.Linear(num_features, num_classes)

# Move model to GPU if available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total #calculate de accuracy 

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Save the trained model
torch.save(model.state_dict(), 'resnet_model.tar')


# Get a single batch from the validation loader
model.eval()  # Set the model to evaluation mode
data_iter = iter(val_loader)
images, true_labels = next(data_iter)

# Move data to the device
images = images.to(device)
true_labels = true_labels.to(device)

# Perform a forward pass to get predictions
with torch.no_grad():
    outputs = model(images)
    probabilities = F.softmax(outputs, dim=1)

# Get the predicted label and confidence
_, predicted_labels = torch.max(outputs, 1)
confidence, _ = torch.max(probabilities, 1)

# Randomly select an index for the image to display
index = random.randint(0, images.size(0) - 1)
image = images[index].cpu().permute(1, 2, 0).numpy()  # Convert to HxWxC format for plotting
true_label = true_labels[index].item()
predicted_label = predicted_labels[index].item()
confidence_score = confidence[index].item()

# Convert the label indexes back to string labels using the dataset's index_to_label dictionary
true_label_str = val_dataset.dataset.index_to_label[true_label]
predicted_label_str = val_dataset.dataset.index_to_label[predicted_label]

# Print the true label, predicted label, and confidence
print(f"True Label: {true_label_str}")
print(f"Predicted Label: {predicted_label_str}")
print(f"Model Confidence: {confidence_score:.4f}")

# Optionally, display the image
plt.imshow(image)
plt.title(f"True: {true_label_str}, Predicted: {predicted_label_str}, Confidence: {confidence_score:.4f}")
plt.axis('off')
plt.show()
