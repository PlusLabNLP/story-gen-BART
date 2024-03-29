import argparse, pickle, os, sys, csv
import re

import numpy as np
from collections import Counter
from collections import defaultdict
from itertools import chain
from datetime import datetime
import time

from scipy.special import expit
import torch
from torch.utils.data import DataLoader, IterableDataset
from torch._six import container_abcs, string_classes, int_classes
from torch.nn.utils.rnn import pad_sequence
#from torchtext import data
#from torchtext import vocab
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

# path = os.path.realpath(__file__)
# path = path[:path.rindex('/')+1]
# sys.path.insert(0, os.path.join(path, '../utils/'))
# sys.path.insert(0, os.path.join(path, '../word_rep/'))

from fairseq.models.bart import BARTModel
from fairseq.data.data_utils import collate_tokens, Corpus


from cnn_context_classifier import CNNContextClassifier
#from pool_ending_classifier import PoolEndingClassifier
#from reprnn import RepRNN

os.environ['CUDA_VISIBLE_DEVICES']="2"


class CustomIterableDataset(IterableDataset):

    def __init__(self, filename, encoder, max_tokens=1024, saved_data=None):
        self.filename = filename
        self.fields = ["context", "generated", "gold", "label"]
        self.encoder = encoder
        self.max_tokens = max_tokens
        if not saved_data:
            self.data = self.preprocess()
        else:
            self.data = saved_data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)
        
    def preprocess(self):
        data = []
        fin = open(self.filename, newline='')
        file_itr = csv.DictReader(fin, delimiter='\t', fieldnames=self.fields)
        for line in file_itr:
            for field in self.fields[:-1]:
                line[field] = self.encoder.encode(line[field])[:self.max_tokens]
            line[self.fields[-1]] = float(line[self.fields[-1]])
            data.append(line)
        return data


np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

#def my_collate(batch):


def my_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        #print(batch)
        #max_len = max(lambda x: x.size()[0], batch)
        #batch = pad_lengths(batch)
        return collate_tokens(batch, pad_idx=1)
        #return pad_sequence(batch, padding_value=1)
        #return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return my_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.half)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: my_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(my_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [my_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def validation_loss(model, valid_iter, ranking_loss, margin_ranking_loss, cuda):
    model.eval()
    #valid_iter.init_epoch()
    v_correct, v_total = 0, 0
    ones = 0
    for k, batch in enumerate(valid_iter):
        if cuda:
            #print(context, generated, gold)
            for key in batch:
                t = batch[key]
                if type(t) == tuple:
                    batch[key] = tuple([item.cuda() for item in t])
                else:
                    batch[key] = t.cuda()

        #if k % 100 == 0:
        #    print(k)
        b_size = batch["context"].size()[0]

        decision_negative = model(batch["context"].transpose(0,1),
                                  batch["generated"].transpose(0,1))
        decision_positive = model(batch["context"].transpose(0,1),
                                  batch["gold"].transpose(0,1))

        if ranking_loss or margin_ranking_loss:
            dec = decision_positive - decision_negative
        else:
            # Evaluate predictions on gold
            dec = decision_positive

        decis = dec.data.cpu().numpy()
        predicts = np.round(expit(decis))
        v_correct += np.sum(np.equal(predicts, np.ones(b_size)))
        v_total += b_size
        ones += np.sum(predicts)
    valid_acc = v_correct / v_total
    print('Valid: %f' % valid_acc)
    print('%d ones %d zeros' % (ones, v_total - ones))
    return valid_acc

parser = argparse.ArgumentParser()
# Data
parser.add_argument('data_dir', type=str, help='path to data directory')
parser.add_argument('--save_to', type=str, default='', help='path to save model to')
parser.add_argument('--load_model', type=str, default='', help='existing model file to load')
parser.add_argument('--load_corpus', type=str, help='existing processed corpus to load')
parser.add_argument('--max_seq_len', type=int, default=200, help='max tokens to truncate data to for training')

# Model
parser.add_argument('--decider_type', type=str, default='cnncontext',
                    help='Decider classifier type [cnncontext, poolending, reprnn]')
# Run Parameters
parser.add_argument('--batch_size',
                    type=int,
                    default=64,
                    help='number of examples to process in parallel')
parser.add_argument('--num_epochs',
                    type=int,
                    default=15,
                    help='number of times to run through training set')
parser.add_argument('--stop_threshold',
                    type=float,
                    default=0.99,
                    help='Early stopping threshold on validation accuracy')
parser.add_argument('--lr',
                    type=float,
                    default=0.01,
                    help='learning rate for optimizer')
parser.add_argument('--adam',
                    action='store_true',
                    help='train with adam optimizer')
parser.add_argument('--train_prefixes',
                    action='store_true',
                    help='train on all ending prefixes')
parser.add_argument('--valid_only',
                    action='store_true',
                    help='use only validation set')
parser.add_argument('--event_type', type=str, choices=[None, "random", "intra", "intraV", "inter"],
                    default=None, help='type of event shuffle used (for naming conventions). See choices enumeration in create_classifier_dataset.py')
parser.add_argument('--ranking_loss',
                    action='store_true',
                    help='train based on ranking loss')
parser.add_argument('--margin_ranking_loss',
                    action='store_true',
                    help='train based on margin ranking loss')
# Model Parameters
parser.add_argument('--embedding_dim',
                    type=int,
                    default=300,
                    help='length of word embedding vectors')
parser.add_argument('--hidden_dim',
                    type=int,
                    default=300,
                    help='length of hidden state vectors')
parser.add_argument('--filter_size',
                    type=int,
                    default=3,
                    help='convolutional filter size')
parser.add_argument('--dropout_rate',
                    type=float,
                    default=0.5,
                    help='dropout rate')
parser.add_argument('--fix_embeddings',
                    action='store_true',
                    help='fix word embeddings')
parser.add_argument('--cuda', action='store_true')
# Output Parameters
parser.add_argument('--valid_every',
                    type=int,
                    default=128,
                    help='batch interval for running validation')
parser.add_argument('--p',
                    action='store_true',
                    help='use this flag to print samples of the data')
args = parser.parse_args()

print("Args:",args)

print('Loading BART')
bart = BARTModel.from_pretrained(
    'checkpoint/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='full'
)
bart.eval()
if args.cuda:
    bart.half()
    bart.cuda()

print("Loading Data")


valid_name = 'valid.txt.title+plot.tsv'
if args.valid_only:
    train_name = valid_name
else:
    if args.event_type:
        train_name = 'disc_train.txt.title+plot.' + args.event_type + '.tsv'
        valid_name = 'valid.txt.title+plot.' + args.event_type + '.tsv'
    else:
        train_name = 'disc_train.txt.title+plot.tsv'

print("Using {} as the training data".format(train_name))

if not args.load_corpus:
    print('Reading the data')

    train = CustomIterableDataset(os.path.join(args.data_dir, train_name), bart, max_tokens=args.max_seq_len)
    val = CustomIterableDataset(os.path.join(args.data_dir, valid_name), bart, max_tokens=args.max_seq_len)    
    corpus = Corpus(train, val, args.max_seq_len)
    save_name = "corpus.{}.{}.data".format("_".join(args.data_dir.split("/")), "_".join(str(datetime.now()).split()))
    torch.save(corpus, save_name)


else:
    print('Loading in preprocessed corpus: {}'.format(args.load_corpus))
    corpus = torch.load(args.load_corpus)
    train = CustomIterableDataset(corpus.train_name, None, max_tokens=corpus.max_seq_len, saved_data=corpus.train)
    val = CustomIterableDataset(corpus.val_name, None, max_tokens=corpus.max_seq_len, saved_data=corpus.val)

num_batch = int(len(train)/args.batch_size)

train_iter = DataLoader(train, batch_size=args.batch_size, collate_fn=my_collate) #, shuffle=True)                                                                      
valid_iter = DataLoader(val, batch_size=args.batch_size, collate_fn=my_collate) #, shuffle=True) 

itos=None #itos = TEXT.vocab.itos if args.p else None # TODO just remove this entirely

print('Initializing the model')

if args.decider_type == 'cnncontext':
    model = CNNContextClassifier( args.hidden_dim, args.filter_size, args.dropout_rate, bart)

# Have not implemented these in BART yet
# elif args.decider_type == 'poolending':
#     model = PoolEndingClassifier(len(TEXT.vocab), args.embedding_dim,
#             args.hidden_dim,
#             embed_mat=TEXT.vocab.vectors,
#             fix_embeddings=args.fix_embeddings).cuda()
# elif args.decider_type == 'reprnn':
#     model = RepRNN(len(TEXT.vocab), args.embedding_dim,
#             args.hidden_dim,
#             embed_mat=TEXT.vocab.vectors).cuda()
else:
  assert False, 'Invalid model type.'

if args.load_model != '':
    model.load_state_dict(torch.load(args.load_model))

if args.cuda:
    model = model.cuda()

loss_function = nn.BCEWithLogitsLoss()
margin_loss_function = nn.MarginRankingLoss()

parameters = filter(lambda p: p.requires_grad, model.parameters())
if args.adam:
    optimizer = optim.Adam(parameters, lr=args.lr)
else:
    optimizer = optim.SGD(parameters, lr=args.lr)

#if args.load_model != '':
#    print('Evaluating model')
#    val_accuracy = validation_loss(model, valid_iter, args.ranking_loss, args.margin_ranking_loss,
#                                   args.cuda)


early_stop = False
best_accuracy = 0
print('{} batches per epoch, {} times validation is run per epoch'.format(num_batch, int(num_batch/args.valid_every)))
for epoch in range(args.num_epochs):
    if early_stop:
        print('Early stopping to prevent overfitting')
        break
    print('Starting epoch %d' % epoch)
    #train_iter.init_epoch()
    correct, total = 0, 0
    total_loss = 0
    start_time = time.time()
    # Everything is loaded in Batch first x seq len by DataLoader, but then later transposed for the Classifier forward function
    for b, batch in enumerate(train_iter):
        model.train()
        model.zero_grad()
        batch_size = batch["context"].size()[0]

        #print(type(context), type(generated))
        if args.cuda:
            #print(context, generated, gold)
            for key in batch:
                t = batch[key]
                if type(t) == tuple:
                    batch[key] = tuple([item.cuda() for item in t])
                else:
                    batch[key] = t.cuda()

        def compute_loss(context, generated, gold):  #TODO move this
    
            decision_negative = model(context, generated, itos=itos)
            decision_positive = model(context, gold, itos=itos)
            if args.ranking_loss or args.margin_ranking_loss:
                decision = decision_positive - decision_negative
            else:
                decision = decision_positive
           
            pos_labels = torch.ones(batch_size, dtype=torch.half)
            neg_labels = torch.zeros(batch_size, dtype=torch.half)
            if args.cuda:
                pos_labels = pos_labels.cuda()
                neg_labels = neg_labels.cuda()
            if args.ranking_loss:
                x_loss = loss_function(
                  decision_positive - decision_negative,
                  pos_labels)
            elif args.margin_ranking_loss:
                # 1: positive ranked higher than negative
                #print(decision_positive.shape)
                x_loss = margin_loss_function(
                  decision_positive, decision_negative,
                  pos_labels)
            else:
                x_loss = loss_function(decision_positive,
                  pos_labels)
                x_loss += loss_function(decision_negative,
                                        neg_labels)

            return x_loss, decision

        loss = None
        if args.train_prefixes:
            end_seq_len = max(batch["generated"].size()[1],
                              batch["gold"].size()[1])
            loss = 0
            #length_range = chain(range(min(10, end_seq_len)),
            #                     range(10, end_seq_len, 5))
            length_range = chain(range(0, min(10, end_seq_len-1), 2),
                                 range(10, min(end_seq_len, 30), 5),
                                 iter([end_seq_len-1]))

            for i in length_range:
                gen_len = min(i + 1, batch["generated"].size()[1])
                gold_len = min(i + 1, batch["gold"].size()[1])
                # prefix_loss, decision = compute_loss(batch.context[0],
                #                                      (batch.generated[0][:gen_len, :].view(gen_len, -1),
                #                                       autograd.Variable(torch.ones(batch_size) * i).cuda()),
                #                                      (batch.gold[0][:gold_len, :].view(gold_len, -1),
                #                                       autograd.Variable(torch.ones(batch_size) * i).cuda()))
                these_labels = torch.HalfTensor(batch_size).fill_(i)
                these_labels = these_labels.cuda() if args.cuda else these_labels
                # starts a seq_len, batch, then transpose to batch x seq_len
                prefix_loss, decision = compute_loss(batch["context"].transpose(0,1),
                                                     (batch["generated"].transpose(0,1)[:gen_len, :].view(gen_len, -1),
                                                      these_labels),
                                                     (batch["gold"].transpose(0,1)[:gold_len, :].view(gold_len, -1),
                                                      these_labels))
                loss += prefix_loss
        else:
            loss, decision = compute_loss(batch["context"].transpose(0,1), batch["generated"].transpose(0,1), batch["gold"].transpose(0,1))
        #breakpoint()
        loss.backward()
        # print(loss)
        total_loss += loss.data.item()
        optimizer.step()

        correct += np.sum(np.equal(
            np.round(expit(decision.data.cpu().numpy())),
            np.ones(batch_size)))
        total += batch_size

        if b % args.valid_every == 0:
            val_start_time = time.time()
            valid_accuracy = validation_loss(model, valid_iter, args.ranking_loss,
                                             args.margin_ranking_loss, args.cuda)
            if epoch > 0 and valid_accuracy > best_accuracy:  # to prevent getting the best by chance early in training and never saving again
                best_accuracy = valid_accuracy
                print('Saving model')
                with open(args.save_to, 'wb') as f:
                    torch.save(model.state_dict(), f)
            elapsed = (time.time() - val_start_time)
            print('Seconds to test validation: {:.2f}'.format(elapsed))
            if valid_accuracy > args.stop_threshold: # early stopping
                early_stop = True
                break
    elapsed = (time.time() - start_time)
    print('Train: %f' % (correct / total))
    print('Loss: %f' % (total_loss / total))
    print('Seconds per epoch: {:.2f}'.format(elapsed))

