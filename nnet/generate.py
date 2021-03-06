###############################################################################
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

import time

parser = argparse.ArgumentParser(description='Char LSTM LM')

# Model parameters.
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
seed = args.seed if args.seed is not None else time.time()
torch.manual_seed(seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

input = torch.LongTensor([[2]]).to(device)
hidden = model.get_hidden_init(batch_size=1)
seq_lengths = torch.LongTensor([1])

with open(args.outf, 'w') as outf:
    outf.write("2 ")
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            output, hidden = model(input, seq_lengths, hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_weights[0:1] = 0.
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            if word_idx == 0:
                continue
            if word_idx == 1:
                outf.write("1\n2 ")
                input = torch.LongTensor([[2]]).to(device)
                hidden = model.get_hidden_init(batch_size=1)
            else:
                outf.write(str(word_idx.item()))
                outf.write(" ")
            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
