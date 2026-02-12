# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="978" height="828" alt="image" src="https://github.com/user-attachments/assets/cbdc9e3e-ebee-437d-94eb-5fb51fdcd3ab" />


## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:

### STEP 3:


## PROGRAM

### Name: Harshini V
### Register Number:212224040109

~~~


class PeopleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# Initialize the Model, Loss Function, and Optimizer
input_size = X_train.shape[1]
num_classes = 4

model = PeopleClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
~~~



## Dataset Information

<img width="1027" height="410" alt="image" src="https://github.com/user-attachments/assets/c16ff1e6-88ec-4656-97f7-17ce94e6baec" />


## OUTPUT



### Confusion Matrix

<img width="683" height="569" alt="image" src="https://github.com/user-attachments/assets/653ede8d-990d-44f9-9d79-ddf6d65a567c" />



### Classification Report

<img width="601" height="371" alt="image" src="https://github.com/user-attachments/assets/b3ad1ff6-6079-467a-8f53-99068b0d7b7e" />




### New Sample Data Prediction


<img width="382" height="50" alt="image" src="https://github.com/user-attachments/assets/7f0cadc3-3aff-4953-9d95-5c88d70772a7" />



## RESULT
Thus the neural network classification model was successfully developed.
