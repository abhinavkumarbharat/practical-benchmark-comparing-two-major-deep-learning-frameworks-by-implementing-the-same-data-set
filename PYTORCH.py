import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# Set device to GPU (Required for Colab T4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load and normalize the CIFAR-10 dataset
# ToTensor() automatically scales pixel values from 0-255 down to 0.0-1.0
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 2. Build the standard CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 3 input channels (RGB), 32 filters, 3x3 window
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Flattened size: 32 filters * 15 * 15 spatial dimensions
        self.fc1 = nn.Linear(32 * 15 * 15, 64) 
        self.fc2 = nn.Linear(64, 10) # 10 output classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # Flatten layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 3. Start the clock and reset memory tracker
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

print("Training started...")
start_time = time.time()

# 4. Train the model (Explicit Training Loop)
model.train()
for epoch in range(5):
    for inputs, labels in trainloader:
        # Move batch data to the GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()               # Clear old math 
        outputs = model(inputs)             # Make guesses
        loss = criterion(outputs, labels)   # Calculate error
        loss.backward()                     # Reverse calculate gradients
        optimizer.step()                    # Update weights

# 5. Stop the clock and capture GPU memory metrics
end_time = time.time()
total_time = end_time - start_time

print("\n--- PyTorch Benchmark Results ---")
print(f"Total Training Time: {total_time:.2f} seconds")

if torch.cuda.is_available():
    # Fetch peak GPU memory usage and convert from Bytes to Megabytes
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(f"Peak GPU Memory Allocated: {peak_memory_mb:.2f} MB")
else:
    print("Error: GPU not detected. Please change Colab runtime to T4 GPU.")