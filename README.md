# DL-EXP-5
# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

**Problem Statement:**  
Predicting stock prices is a complex task due to market volatility. Using historical closing prices, we aim to develop a Recurrent Neural Network (RNN) model that can analyze time-series data and generate future price predictions.

**Dataset:**  
The dataset consists of historical stock closing prices from `trainset.csv` and `testset.csv`. The data is normalized using MinMax scaling, and sequences of 60 past values are used as input features. The model learns patterns from training data to predict upcoming prices, helping traders and investors make informed decisions.

### Step 1:
- Data Collection & Preprocessing: Load historical stock prices, normalize using MinMaxScaler, and create sequences for time-series input.

### Step 2:
- Model Design: Build an RNN with two layers, define input/output sizes, and set activation functions.

### Step 3:
- Training Process: Train the model using MSE loss and Adam optimizer for 20 epochs with batch-size optimization.

### Step 4:
- Evaluation & Prediction: Test on unseen data, inverse transform predictions, and compare with actual prices.

### Step 5:
- Visualization & Interpretation: Plot training loss and predictions to analyze performance and potential improvements.


### Step 1:
- Data Collection & Preprocessing: Load historical stock prices, normalize using MinMaxScaler, and create sequences for time-series input.

### Step 2:
- Model Design: Build an RNN with two layers, define input/output sizes, and set activation functions.

### Step 3:
- Training Process: Train the model using MSE loss and Adam optimizer for 20 epochs with batch-size optimization.

### Step 4:
- Evaluation & Prediction: Test on unseen data, inverse transform predictions, and compare with actual prices.

### Step 5:
- Visualization & Interpretation: Plot training loss and predictions to analyze performance and potential improvements.


## Program
#### Name:POOJA PRIYA B
#### Register Number:212224230196
Include your code here
```Python 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

## Step 1: Load and Preprocess Data
# Load training and test datasets
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')

# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

x_train.shape, y_train.shape, x_test.shape, y_test.shape

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

## Step 2: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

!pip install torchinfo


from torchinfo import summary

# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
train_losses = []

model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')


# Plot training loss
print('Name: POOJA PRIYA B ')
print('Register Number: 212224230196    ')
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

## Step 4: Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name: POOJA PRIYA B              ')
print('Register Number: 212224230196    ')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')


```

## Output

### True Stock Price, Predicted Stock Price vs time


<img width="682" height="507" alt="image" src="https://github.com/user-attachments/assets/dd3d075d-21c1-431a-8b2e-7840283b2f77" />
### Predictions 

<img width="907" height="587" alt="image" src="https://github.com/user-attachments/assets/62537a6a-004c-4cb6-ba4b-a596e3151572" />


<img width="343" height="46" alt="image" src="https://github.com/user-attachments/assets/6a38ec79-9c1f-452c-9528-37e69e7f64bc" />



## Result

Thus, a Recurrent Neural Network model for stock price prediction has successfully been devoloped
