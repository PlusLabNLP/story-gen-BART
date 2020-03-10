import argparse, pickle, os, sys, csv
import re

import numpy as np
from collections import Counter
from collections import defaultdict
from itertools import chain


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

from cnn_context_classifier import CNNContextClassifier
#from pool_ending_classifier import PoolEndingClassifier
#from reprnn import RepRNN

class CustomIterableDataset(IterableDataset):

    def __init__(self, filename, encoder):
        self.filename = filename
        self.fields = ["context", "generated", "gold", "label"]
        self.encoder = encoder
        self.data = self.preprocess()

    def __iter__(self):
        return iter(self.data)

    def preprocess(self): # TODO note that this means the max sequence length is 120
        data = []
        fin = open(self.filename, newline='')
        file_itr = csv.DictReader(fin, delimiter='\t', fieldnames=self.fields)
        for line in file_itr:
            for field in self.fields[:-1]:
                line[field] = self.encoder.encode(line[field])
            line[self.fields[-1]] = float(line[self.fields[-1]])
            data.append(line)
        return data


np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


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
        return pad_sequence(batch)
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
        return torch.tensor(batch, dtype=torch.float64)
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

parser = argparse.ArgumentParser()
# Data
parser.add_argument('data_dir', type=str, help='path to data directory')
parser.add_argument('--save_to', type=str, default='', help='path to save model to')
parser.add_argument('--load_model', type=str, default='', help='existing model file to load')
parser.add_argument('--dic', type=str, default='dic.pickle',
                    help='lm dic to use as vocabulary')
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
    data_name_or_path='temp'
)

print("Loading Data")


# TEXT = data.Field(sequential=True, use_vocab=False, include_lengths=True)
# TEXT.numericalize =
#
# # LABEL = data.Field(sequential=False, use_vocab=False, tensor_type=torch.FloatTensor, postprocessing=data.Pipeline(lambda x, y: float(x)))
# LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float, postprocessing=data.Pipeline(lambda x, y: float(x)))
#

valid_name = 'valid.txt.tiny.tsv'
if args.valid_only:
    train_name = valid_name
else:
    if args.event_type:
        train_name = 'disc_train.txt.' + args.event_type + '.tsv'
    else:
        train_name = 'disc_train.txt.tiny.tsv'

print("Using {} as the training data".format(train_name))

print('Reading the data')

train = CustomIterableDataset(os.path.join(args.data_dir, train_name), bart)
val = CustomIterableDataset(os.path.join(args.data_dir, valid_name), bart)

# TODO save and cache the preprocessed datasets cause that shit is slow
train_iter = DataLoader(train, batch_size=args.batch_size, collate_fn=my_collate) #, shuffle=True)
valid_iter = DataLoader(val, batch_size=args.batch_size, collate_fn=my_collate) #, shuffle=True)


# train, valid = data.TabularDataset.splits(
#     path=args.data_dir,
#     train=train_name, validation='valid.txt.tsv',
#     format='tsv',
#     fields=[
#         ('context', TEXT),
#         ('generated', TEXT),
#         ('gold', TEXT),
#         ])
# if args.p:
#     print('Example Train data:\nContext: {}\nGold: {}\nGenerated: {}'.format(
#         ' '.join(train[0].context), ' '.join(train[0].gold), ' '.join(train[0].generated)))
#     print('Example Valid data:\nContext: {}\nGold: {}\nGenerated: {}'.format(
#         ' '.join(valid[0].context), ' '.join(valid[0].gold), ' '.join(valid[0].generated)))

#TEXT.build_vocab(train)
# Read in the LM dictionary.
#print('Building the dictionary')
#with open(args.dic, 'rb') as dic_file:
#    dictionary = pickle.load(dic_file)

#dictionary.idx2word.insert(0,"[UNK]")
#for item in dictionary.word2idx:
#    dictionary.word2idx[item] += 1

# Reconstruct the dictionary in torchtext.
#counter = Counter({'[UNK]': 0, '</s>':0})
#TEXT.vocab = vocab.Vocab(counter, specials=['[UNK]', '</s>'])
#TEXT.vocab.itos = dictionary.idx2word
#TEXT.vocab.stoi = defaultdict(vocab._default_unk_index, dictionary.word2idx)
#print(TEXT.vocab.stoi["<EOL>"])
#print(TEXT.vocab.itos[0])

#TEXT.vocab.load_vectors('glove.6B.%dd' % args.embedding_dim)
# TEXT.vocab.load_vectors('glove.6B')
itos=None #itos = TEXT.vocab.itos if args.p else None # TODO just remove this entirely
#print('Vocab size %d' % len(TEXT.vocab))

#train_iter = data.Iterator(dataset=train, batch_size=args.batch_size,
#        sort_key=lambda x: len(x.context), sort=True, repeat=False)
#valid_iter = data.Iterator(dataset=valid, batch_size=args.batch_size, sort_key=lambda x: len(x.context), sort=True, repeat=False)

print('Initializing the model')

if args.load_model != '':
    with open(args.load_model, 'rb') as f:
        model = torch.load(f)

elif args.decider_type == 'cnncontext':
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

if args.cuda:
    model = model.cuda()

loss_function = nn.BCEWithLogitsLoss()
margin_loss_function = nn.MarginRankingLoss()

parameters = filter(lambda p: p.requires_grad, model.parameters())
if args.adam:
    optimizer = optim.Adam(parameters, lr=args.lr)
else:
    optimizer = optim.SGD(parameters, lr=args.lr)

if args.load_model != '':
    print('Evaluating model')
    model.eval()
    #valid_iter.init_epoch()
    v_correct, v_total = 0, 0
    ones = 0
    for k, batch in enumerate(valid_iter):
        #if k % 100 == 0:
        #    print(k)
        batch_size = batch.context[0].size()[1]

        decision_negative = model(batch.context[0],
            batch.generated, itos=itos)
        decision_positive = model(batch.context[0],
            batch.gold, itos=itos)

        if args.ranking_loss or args.margin_ranking_loss:
            decision = decision_positive - decision_negative
        else:
            # Evaluate predictions on gold
            decision = decision_positive

        decis = decision.data.cpu().numpy()
        predicts = np.round(expit(decis))
        v_correct += np.sum(np.equal(predicts, np.ones(batch_size)))
        v_total += batch_size
        ones += np.sum(predicts)
    print('Valid: %f' % (v_correct / v_total))
    print('%d ones %d zeros' % (ones, v_total - ones))


early_stop = False
best_accuracy = 0
for epoch in range(args.num_epochs):
    if early_stop:
        break
    print('Starting epoch %d' % epoch)
    #train_iter.init_epoch()
    correct, total = 0, 0
    total_loss = 0
    for b, batch in enumerate(train_iter):
        model.train()
        model.zero_grad()
        batch_size = batch["context"].size()[1]

        def compute_loss(context, generated, gold):  #TODO move this
            decision_negative = model(context, generated, itos=itos)
            decision_positive = model(context, gold, itos=itos)
            if args.ranking_loss or args.margin_ranking_loss:
                decision = decision_positive - decision_negative
            else:
                decision = decision_positive

            x_loss = None
            if args.ranking_loss:
                x_loss = loss_function(
                  decision_positive - decision_negative,
                  autograd.Variable(torch.ones(batch_size)))
            elif args.margin_ranking_loss:
                # 1: positive ranked higher than negative
                #print(decision_positive.shape)
                x_loss = margin_loss_function(
                  decision_positive, decision_negative,
                  autograd.Variable(torch.ones(batch_size)))
            else:
                x_loss = loss_function(decision_positive,
                  autograd.Variable(torch.ones(batch_size)))
                x_loss += loss_function(decision_negative,
                  autograd.Variable(torch.zeros(batch_size)))

            return x_loss, decision

        loss = None
        if args.train_prefixes:
            end_seq_len = max(batch["generated"].size()[0],
                              batch["gold"].size()[0])
            loss = 0
            #length_range = chain(range(min(10, end_seq_len)),
            #                     range(10, end_seq_len, 5))
            length_range = chain(range(0, min(10, end_seq_len-1), 2),
                                 range(10, min(end_seq_len, 30), 5),
                                 iter([end_seq_len-1]))

            for i in length_range:
                gen_len = min(i + 1, batch["generated"].size()[0])
                gold_len = min(i + 1, batch["gold"].size()[0])
                # prefix_loss, decision = compute_loss(batch.context[0],
                #                                      (batch.generated[0][:gen_len, :].view(gen_len, -1),
                #                                       autograd.Variable(torch.ones(batch_size) * i).cuda()),
                #                                      (batch.gold[0][:gold_len, :].view(gold_len, -1),
                #                                       autograd.Variable(torch.ones(batch_size) * i).cuda()))
                prefix_loss, decision = compute_loss(batch["context"],
                                                     (batch["generated"][:gen_len, :].view(gen_len, -1),
                                                      torch.ones(batch_size) * i),
                                                     (batch["gold"][:gold_len, :].view(gold_len, -1),
                                                      torch.ones(batch_size) * i))
                loss += prefix_loss
        else:
            loss, decision = compute_loss(batch["context"], batch["generated"], batch["gold"])

        loss.backward()
        # print(loss)
        total_loss += loss.data.item()
        optimizer.step()

        correct += np.sum(np.equal(
            np.round(expit(decision.data.cpu().numpy())),
            np.ones(batch_size)))
        total += batch_size

        if b % args.valid_every == 0:
            model.eval()
            #valid_iter.init_epoch()
            v_correct, v_total = 0, 0
            ones = 0
            for k, batch in enumerate(valid_iter):
                #if k % 100 == 0:
                #    print(k)
                # batch_size = batch.context[0].size()[1]
                #
                # decision_negative = model(batch.context[0],
                #     batch.generated, itos=itos)
                # decision_positive = model(batch.context[0],
                #     batch.gold, itos=itos)
                batch_size = batch.context[0].size()[1]

                # bg = batch.generated
                # item = torch.cat(bg, dim=-1).to('cuda')
                # batch.generated = torch.chunk(item, chunks=2, dim=-1)

                temp = []

                for bg_item in batch.generated:
                    temp.append( bg_item)
                batch.generated = tuple(temp)

                temp2 = []
                for bg_item in batch.gold:
                    if bg_item is not None:
                        temp2.append(bg_item)
                batch.gold = tuple(temp2)

                def convert_nested_tuple_cuda(obj):
                    if isinstance(obj, tuple):
                        return convert_nested_tuple_cuda(obj)
                    elif isinstance(obj, torch.Tensor):
                        return obj.cuda()
                    # else:
                    #     print('Jinrui Debug')
                    return list(convert_nested_tuple_cuda(o) for o in obj)



                # print("bacth generated", batch.generated)
                #
                # for item in batch.generated:
                #     print("batch generated item: ", item)


                decision_negative = model(batch.context[0],
                        batch.generated, itos=itos)
                decision_positive = model(batch.context[0],
                    batch.gold, itos=itos)

                if args.ranking_loss or args.margin_ranking_loss:
                    decision = decision_positive - decision_negative
                else:
                    # Evaluate predictions on gold
                    decision = decision_positive

                decis = decision.data.cpu().numpy()
                predicts = np.round(expit(decis))
                v_correct += np.sum(np.equal(predicts, np.ones(batch_size)))
                v_total += batch_size
                ones += np.sum(predicts)
            valid_accuracy = v_correct / v_total
            print('Valid: %f' % (valid_accuracy))
            print('%d ones %d zeros' % (ones, v_total - ones))

            if epoch > 1 and valid_accuracy > best_accuracy:  # to prevent getting the best by chance early in training and never saving again
                best_accuracy = valid_accuracy
                print('Saving model')
                with open(args.save_to, 'wb') as f:
                    torch.save(model, f)

            if valid_accuracy > args.stop_threshold: # early stopping
                early_stop = True
                break

    print('Train: %f' % (correct / total))
    print('Loss: %f' % (total_loss / total))
