from model import SimpleNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load training data
trainset = datasets.MNIST('mnist_data/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load test data
testset = datasets.MNIST('mnist_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # Common choice for classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)  # Adam optimizer

num_epochs = 1000000

for epoch in range(num_epochs):
    for inputs, labels in train_loader:  # Assuming you have a DataLoader `train_loader`
        inputs = inputs.view(inputs.shape[0], -1)  # Flatten the images
        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.view(images.shape[0], -1)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

torch.save(model.state_dict(), 'mnist_model.pth')