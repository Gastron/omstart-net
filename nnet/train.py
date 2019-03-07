#Based on https://github.com/pytorch/examples/blob/master/word_language_model/main.py
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import model
import data

parser = argparse.ArgumentParser(description='PyTorch LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/tokenized/wappulehti.all.tokens',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

eval_batch_size = 10
train_dataset = data.InMemoryTextDataSet(args.data)
val_dataset = data.InMemoryTextDataSet(args.data)
train_dataset = data.InMemoryTextDataSet(args.data)

#to be used this way:
#train_gen = torch.utils.data.DataLoader(train_dataset, 
#        batch_size = args.batch_size, 
#        shuffle = True,
#        collate_fn = data.pad_and_sort_batch)

###############################################################################
# Build the model
###############################################################################

model = model.LSTMLM(train_dataset.vocab_size,  #One for the padding
        args.emsize, 
        args.nhid, 
        args.nlayers).to(device)

criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

###############################################################################
# Training code
###############################################################################

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    data_gen = torch.utils.data.DataLoader(data_source, 
        batch_size = eval_batch_size, 
        shuffle = False,
        collate_fn = data.pad_and_sort_batch)
    with torch.no_grad():
        for inputs, targets, seq_lengths in data_gen:
            inputs, targets, seq_lengths = inputs.to(device), targets.to(device), seq_lengths.to(device)
            output, hidden = model(inputs, seq_lengths)
            loss = criterion(output.permute(0,2,1), targets)
            total_loss += loss.item()
    return total_loss / (len(data_source) / eval_batch_size) 

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    train_gen = torch.utils.data.DataLoader(train_dataset, 
        batch_size = args.batch_size, 
        shuffle = True,
        collate_fn = data.pad_and_sort_batch)
    for batch_i, (inputs, targets, seq_lengths) in enumerate(train_gen):
        inputs, targets, seq_lengths = inputs.to(device), targets.to(device), seq_lengths.to(device)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        output, hidden = model(inputs, seq_lengths)
        loss = criterion(output.permute(0,2,1), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch_i % args.log_interval == 0 and batch_i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batch_ies | lr {:02.2f} | ms/batch_i {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch_i, len(train_dataset), lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0.
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_dataset)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss or True:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
