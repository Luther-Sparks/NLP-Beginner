import time
import torch
import torch.nn as nn
from model import ESIM

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import argparse
from tqdm import tqdm

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from datasets import Dataset, load_dataset

import os

def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)

def data_process(data_iter, batch_size):
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(data_iter, tokenizer))
    vocab.set_default_index(vocab['<unk>'])
    def collate_batch(batch):
        premises, hypotheses, labels = [], [], []
        for premise, hypothesis, label in batch:
            premises.append(torch.tensor([vocab[token] for token in tokenizer(premise)]))
            hypotheses.append(torch.tensor([vocab[token] for token in tokenizer(hypothesis)]))
            labels.append(label)
        premises = torch.cat(premises)
        hypotheses = torch.cat(hypotheses)
        labels = torch.tensor(labels)
        return {'premise': premises, 'hypothese': hypotheses, 'label': labels}
    return DataLoader(data_iter, batch_size=batch_size, collate_fn=collate_batch), vocab
    

def correct_predictions(output_probabilities, targets):
    """Compute the number of predictions that match some target classes in the output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


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
        premises_length = torch.Tensor([len(seq) for seq in batch['premise']]).to(device)
        hypotheses = batch['hypothese'].to(device)
        hypotheses_length = torch.Tensor([len(seq) for seq in batch['hypothese']]).to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits, probs = model(premises, premises_length, hypotheses, hypotheses_length)
        loss = criterion(logits, labels)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1),
                              running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)
        
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_preds / len(dataloader.dataset)
    
    return epoch_time, epoch_loss, epoch_acc

def validate(model, dataloader, criterion):
    model.eval()
    device = model.device
    
    epoch_start = time.time()
    running_loss = 0.0
    running_acc = 0.0
    
    # Deactivate autograd for evaluation
    with torch.no_grad():
        for batch in dataloader:
            premises = batch['premise'].to(device)
            premises_length = torch.Tensor([len(seq) for seq in batch['premise']]).to(device)
            hypotheses = batch['hypothese'].to(device)
            hypotheses_length = torch.Tensor([len(seq) for seq in batch['hypothese']]).to(device)
            labels = batch['label'].to(device)
            
            logits, probs = model(
                premises, 
                premises_length,
                hypotheses,
                hypotheses_length
            )
            
            loss = criterion(logits, labels)
            running_loss += loss.item()
            running_acc += correct_predictions(probs, labels)
    
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader.dataset)
    
    return epoch_time, epoch_loss, epoch_acc

if __name__ is '__main__':
    parser = argparse.ArgumentParser(description='Train a model on the SNLI dataset.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--hidden_size', type=int, default=300, help='Hidden size.')
    parser.add_argument('--embedding_size', type=int, default=300, help='Embedding size.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--label_num', type=int, default=3, help='Number of labels.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--patience', type=int, default=0, help='Patience for early stopping.')
    parser.add_argument('--max_gradient_norm', type=float, default=5.0, help='Max gradient norm.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint path.')
    parser.add_argument('--target_dir', type=str, default='./record/', help='Target directory for saving the model.')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    embedding_size = args.embedding_size
    dropout = args.dropout
    label_num = args.label_num
    epochs = args.epochs
    lr = args.lr
    patience = args.patience
    max_gradient_norm = args.max_gradient_norm
    checkpoint_path = args.checkpoint_path
    target_dir = args.target_dir
    
    
    print('Using device:', device)
    print(20 * '=', "Preparing for training", 20 * '=')
    
    # --------------------------- Data Loading --------------------------- #
    snli = load_dataset('snli')
    train_data = Dataset(snli['train'])
    num_train = int(len(train_data) * 0.8)
    split_train_, split_valid_ = random_split(train_data, [num_train, len(train_data) - num_train])
    train_loader, vocab = data_process(split_train_, batch_size)
    valid_loader, _ = data_process(split_valid_, batch_size)
    
    
    # --------------------------- Model Training --------------------------- #
    print(20 * '=', "Training", 20 * '=')
    model = ESIM(hidden_size, embedding_size, len(vocab), label_num, dropout, device).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    
    best_score = 0.0
    start_epoch = 1
    
        # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []

    # Continuing training from a checkpoint if one was given as argument.
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
        
    _, valid_loss, valid_acc = validate(model, valid_loader, criterion)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%".format(valid_loss, (valid_acc*100)))
    patience_counter = 0
    for epoch in range(start_epoch, epochs+1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model,
                                                       train_loader,
                                                       optimizer,
                                                       criterion,
                                                       epoch,
                                                       max_gradient_norm)

        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = validate(model,
                                                          valid_loader,
                                                          criterion)

        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)

        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            # Save the best model. The optimizer is not saved to avoid having
            # a checkpoint file that is too heavy to be shared. To resume
            # training from the best model, use the 'esim_*.pth.tar'
            # checkpoints instead.
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "best.pth.tar"))

        # Save the model at each epoch.
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                   os.path.join(target_dir, "esim_{}.pth.tar".format(epoch)))

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break
    
    
    
    
    