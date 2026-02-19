# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="737" height="855" alt="image" src="https://github.com/user-attachments/assets/960019a0-5198-403c-8a8d-55b3c453ea48" />



## DESIGN STEPS

### Step 1:


Identify input features and target variable (customer segment A, B, C, D). Recognize it as a multi-class classification problem.



### Step 2:

Import the dataset using pandas or similar tools. Check data shape, columns, and target labels.



### Step 3:

Detect any null or missing values in the dataset. Fill them using mean/median or remove affected rows.


### Step 4:

Convert categorical features into numerical form using encoding. Convert segments A, B, C, D into numeric labels (0–3).

### Step 5:

Divide data into training and testing sets (e.g., 80/20). Training data is used to learn, testing data evaluates performance.



### Step 7: 

Set input neurons equal to number of features. Use hidden layers and 4 output neurons with Softmax activation.


### Step 8: 

Use CrossEntropyLoss for multi-class classification. Select Adam optimizer for efficient training.


### Step 9: 

Perform forward propagation and compute loss. Apply backpropagation to update weights over multiple epochs.

### Step 10:

Calculate accuracy and generate confusion matrix. Check precision, recall, and F1-score for performance analysis.


### Step 11: Make Predictions

Use the trained model on new customer data. Predict the correct segment (A, B, C, or D).


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


<img width="675" height="548" alt="image" src="https://github.com/user-attachments/assets/49420f84-c80e-470c-8254-76e00b229eff" />




### Classification Report


<img width="597" height="447" alt="image" src="https://github.com/user-attachments/assets/1a07294b-35d7-47ed-9fb7-552877705636" />




### New Sample Data Prediction


<img width="376" height="102" alt="image" src="https://github.com/user-attachments/assets/257730da-c6ef-4411-ae92-d593b3275df6" />





## RESULT
Thus the neural network classification model was successfully developed.
