import argparse

import sys
import os
import copy
import torch
from fairseq.models.bart import BARTModel
from fairseq.models.roberta import RobertaModel
from fairseq.StaticCoefficientModel import CoefTrainer
from utils import load_scorers


parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=str, default='./temp/valid.txt.title+plot.tiny', help='input file')
parser.add_argument('--outfile', type=str, default='./temp/valid.txt.title+plot.tiny.out', help='output file')
parser.add_argument('--scorers', type=str, default='WP_scorers.tsv', help='tsv with discriminator info')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--dedup', action='store_true')
parser.add_argument('--ranking_loss', action='store_true')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--split', type=str, default="<EOT>", help='first character to split on for context and continuation')

args = parser.parse_args()
print("Args: ", args, file=sys.stderr)

os.environ['CUDA_VISIBLE_DEVICES']="0"

use_cuda = torch.cuda.is_available()

### load BART model
bart = BARTModel.from_pretrained(
    'checkpoint_full/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='full'
)


bart.eval()

if use_cuda:
    bart.cuda() # remove this line if not running with cuda
    bart.half() # doesn't work with CPU



### load discriminators
scorers = []
coefs, scorer_info = load_scorers(args.scorers) # will need to deal with the config only if learning
for info in scorer_info:
    if len(info) > 3:
        print("too many fields (3 req): {}".format(info))
    model_dir, checkpoint_name, data_path = info

    roberta = RobertaModel.from_pretrained(
    model_dir,
    checkpoint_file=checkpoint_name,
    data_name_or_path=data_path)

    roberta.eval()
    if use_cuda:
        roberta.cuda()
        roberta.half()

    scorers.append(roberta)

# learning coefficients
coef_trainer = CoefTrainer(len(scorers), args.ranking_loss, args.lr)



count = 0
bsz = args.batch_size

with open(args.infile, 'r') as source, open(args.outfile, 'w') as fout:
    sline = source.readline().strip()
    slines, cont_lines = [], []
    # TODO consider adding epochs here to re-loop over data
    for sline in source:
        if count % bsz == 0 and count:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, sampling=True, sampling_topk=5, lenpen=2.0,
                                               max_len_b=250, min_len=55, no_repeat_ngram_size=3,
                                               rescore=True, coefs=coefs, scorers=scorers,
                                               learn=True, dedup=args.dedup, gold_tokens=cont_lines,
                                               coef_trainer=coef_trainer)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis.replace('\n', '') + '\n')
                fout.flush()
            slines, cont_lines = [], []

        sline, cont_line = sline.strip().split(args.split, 1)
        slines.append(sline)
        cont_lines.append(cont_line)
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, sampling=True,  sampling_topk=5, lenpen=2.0, max_len_b=250, min_len=55, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis.replace('\n','') + '\n')
            fout.flush()
