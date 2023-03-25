import time
import torch
import torch.nn as nn
from tqdm import tqdm
def train(model, dataloader, optimizer, criterion, max_gradient_norm):
    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(dataloader):
        batch_start = time.time()
        
        premises = batch['premise'].to(device)
        premises_length = batch['premise_length'].to(device)
        