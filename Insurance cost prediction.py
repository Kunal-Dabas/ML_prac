import torch
import jovian
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split
import seaborn as sns

DATASET_URL = "https://gist.github.com/BirajCoder/5f068dfe759c1ea6bdfce9535acdb72d/raw/c84d84e3c80f93be67f6c069cbdc0195ec36acbd/insurance.csv"
DATA_FILENAME = "insurance.csv"
download_url(DATASET_URL, '.')

dataframe_raw = pd.read_csv(DATA_FILENAME)
dataframe_raw.head()

def customize_dataset(dataframe_raw, rand_str):
    dataframe = dataframe_raw.copy(deep=True)
    # drop some rows
    dataframe = dataframe.sample(int(0.95*len(dataframe)), random_state=int(ord(rand_str[0])))
    # scale input
    dataframe.bmi = dataframe.bmi * ord(rand_str[1])/100.
    # scale target
    dataframe.charges = dataframe.charges * ord(rand_str[2])/100.
    # drop column
    if ord(rand_str[3]) % 2 == 1:
        dataframe = dataframe.drop(['region'], axis=1)
    return dataframe

dataframe = customize_dataset(dataframe_raw, "daemon")
dataframe.head()

num_rows, num_columns = dataframe_raw.shape

# Input columns 
input_cols  = [col for col in dataframe_raw.columns]

# Non-numeric (categorical) columns
categorical_cols = [col for col in dataframe_raw.columns if dataframe_raw[col].dtype == 'object']

output_cols  = [col for col in dataframe.columns]

print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")
print(f"Input columns: {input_cols}")
print(f"Non-numeric (categorical) input columns: {categorical_cols}")
print(f"Output/target columns: {output_cols}")

# Write your answer here
print(f"Minimum Value of Charge: {dataframe.charges.min()}")
print(f"Maximum Value of Charge: {dataframe.charges.max()}")
print(f"Mean Value of Charge: {dataframe.charges.mean()}")

# import matplotlib.pyplot as plt
plt.title("Values of charge")
sns.displot(dataframe.charges)

#####################   Prepare the dataset for training   #####################

def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array

inputs_array, targets_array = dataframe_to_arrays(dataframe_raw)
inputs_array, targets_array


#Convert the numpy arrays inputs_array and targets_array into PyTorch tensors. Make sure that the data type is torch.float32
inputs = torch.tensor(inputs_array, dtype=torch.float32)
targets = torch.tensor(targets_array, dtype=torch.float32)
print(inputs.dtype, targets.dtype)


dataset = TensorDataset(inputs, targets)
val_percent = 0.17 # between 0.1 and 0.2
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

for xb, yb in train_loader:
    print("inputs:", xb)
    print("targets:", yb)
    break

#######################################  Create a Linear Regression Model  #######################################
input_size = len(input_cols)
output_size = len(output_cols)


class InsuranceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, xb):
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)          
        # Calcuate loss
        loss = F.mse_loss(out, targets)                         
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.mse_loss(out, targets)                      
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch+1) % 20 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))
            
model = InsuranceModel()

list(model.parameters())


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history

result = evaluate(model, val_loader) # Use the the evaluate function
print(result)


epochs = 1500
lr = 1e-5
history1 = fit(epochs, lr, model, train_loader, val_loader)


val_loss = history1[-1]['val_loss']
print(val_loss)



def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)                 # fill this
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)
    
input, target = val_ds[0]
predict_single(input, target, model)

input, target = val_ds[23]
predict_single(input, target, model)