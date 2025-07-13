import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
from sklearn.metrics import r2_score # Import r2_score
import pandas as pd # Import pandas to load data

torch.manual_seed(0)
np.random.seed(0)

input_window = 500 # number of input steps
output_window = 1 # number of prediction steps, in this model its fixed to one
block_len = input_window + output_window # for one input-output pair
train_size = 0.55
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data first to calculate the number of training samples
df = pd.read_csv('fouling.csv')
amplitude = df["Fouling factor (m2 K/kW)"].values
sampels = int(len(amplitude) * train_size)
total_training_samples = sampels - block_len + 1
batch_size = max(1, int(total_training_samples / 300))
batch_size = min(batch_size, total_training_samples)
print(f"Setting batch_size to: {batch_size}")


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=15000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        pe = pe.unsqueeze(0).transpose(0, 1) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :].repeat(1,x.shape[1],1)

class TransAm(nn.Module):
    def __init__(self,feature_size=250,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.input_embedding  = nn.Linear(1,feature_size)
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.input_embedding(src) # linear transformation before positional embedding
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
def create_inout_sequences(input_data, input_window ,output_window):
    inout_seq = []
    L = len(input_data)
    block_num =  L - block_len + 1
   
    for i in range( block_num ):
        train_seq = input_data[i : i + input_window]
        train_label = input_data[i + output_window : i + input_window + output_window]
        inout_seq.append((train_seq ,train_label))

    return torch.FloatTensor(np.array(inout_seq))

def get_data():
   
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd

    df = pd.read_csv('fouling.csv')
    amplitude = df["Fouling factor (m2 K/kW)"].values

   
    scaler = MinMaxScaler(feature_range=(0, 1))
    amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)

    sampels = int(len(amplitude) * train_size) # use a parameter to control training size
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]
    train_sequence = create_inout_sequences( train_data,input_window ,output_window)
    test_data = create_inout_sequences(test_data,input_window,output_window)
    if len(test_data) == 0:
        print("Warning: Test data sequence is empty. Cannot perform validation or future prediction.")

    return train_sequence.to(device),test_data.to(device), scaler

def get_batch(input_data, i , batch_size):

    batch_len = min(batch_size, len(input_data) - i)
    data = input_data[ i:i + batch_len ]
    input = torch.stack([item[0] for item in data]).view((input_window,batch_len,1))
    target = torch.stack([item[1] for item in data]).view((input_window,batch_len,1))
    return input, target

def train(train_data):
    model.train() # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data), batch_size)):  # Now len-1 is not necessary
        # data and target are the same shape with (input_window,batch_len,1)
        data, targets = get_batch(train_data, i , batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def calculate_r2(truth, prediction):
    """Calculates the R2 score."""
    from sklearn.metrics import r2_score
    return r2_score(truth, prediction)

def plot_and_loss(eval_model, data_source,epoch):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(len(data_source)):  # Now len-1 is not necessary
            data, target = get_batch(data_source, i , 1) # one-step forecast
            output = eval_model(data)
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    
    len(test_result)

    import os
    if not os.path.exists('graph'):
        os.makedirs('graph')

    pyplot.plot(test_result,color="red", label="Prediction")
    pyplot.plot(truth[:500],color="blue", label="Truth")
    # pyplot.plot(test_result-truth,color="green") # Removed the green line
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.legend() # Added legend
    pyplot.savefig('graph/transformer-epoch%d.png'%epoch)
    pyplot.show() # Display the plot
    pyplot.close()

    # Calculate and print R2 score
    print(f"Shape of truth: {truth.shape}, Type of truth: {truth.dtype}")
    print(f"Shape of test_result: {test_result.shape}, Type of test_result: {test_result.dtype}")
    r2 = calculate_r2(truth, test_result)
    print(f"Epoch {epoch} Validation R2 Score: {r2:.6f}")

    return total_loss / i

# predict the next n steps based on the input data
def predict_future(eval_model, data_source,steps):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    data, _ = get_batch(data_source , 0 , 1)
    with torch.no_grad():
        for i in range(0, steps):
            output = eval_model(data[-input_window:])
            data = torch.cat((data, output[-1:])) 

    data = data.cpu().view(-1)

    import os
    if not os.path.exists('graph'):
        os.makedirs('graph')

    pyplot.plot(data,color="red")
    pyplot.plot(data[:input_window],color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.title(f'Prediction with input window size: {input_window}')
    pyplot.savefig('graph/transformer-future.png')
    pyplot.show()
    pyplot.close()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source), eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data_source)

train_data, val_data, scaler = get_data()
model = TransAm().to(device)

criterion = nn.MSELoss()
lr = 0.005
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

best_val_loss = float("inf")
epochs = 10 # The number of epochs
best_model = None

input_windows_to_test = [500]
import pandas as pd

for current_input_window in input_windows_to_test:
    print(f"\nTraining and predicting with input_window = {current_input_window}")

    # Update input window and block length
    input_window = current_input_window
    block_len = input_window + output_window # output_window is already defined
    df = pd.read_csv('/content/drive/My Drive/fouling.csv')
    amplitude = df["Fouling factor (m2 K/kW)"].values
    sampels = int(len(amplitude) * train_size)
    total_training_samples = sampels - block_len + 1
    # Ensure batch size is at least 1, and not larger than the number of training samples
    batch_size = max(1, int(total_training_samples / 300))
    batch_size = min(batch_size, total_training_samples)
    print(f"Setting batch_size to: {batch_size}")

    train_data, val_data, scaler = get_data()

    # Re-initialize model, optimizer, and scheduler
    model = TransAm().to(device)
    criterion = nn.MSELoss()
    lr = 0.005
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    best_val_loss = float("inf")
    epochs = 10 # Keep epochs at 10 as previously requested
    best_model = None

    # Train model
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data)
        if len(val_data) > 0:
            val_loss = plot_and_loss(model, val_data, epoch)
            print('-' * 89)
            print('| End of epoch {:3d} | Time: {:5.2f}s | Validation Loss {:5.5f} | '
                  'Validation ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss, math.exp(val_loss)))
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model

        scheduler.step()

    # Predict and plot future
    if len(val_data) > 0:
      predict_future(model, val_data, 2400)
    else:
      print("Cannot predict future as validation data is empty.")

print("\nFinished testing all input window values.")
