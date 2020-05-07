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
parser.add_argument('--learn_every_token', action='store_true', help='causes the model to learn on each token generated rather than wait for each sample to finish')
parser.add_argument('--ranking_loss', action='store_true')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--split', type=str, default="<EOT>", help='first character to split on for context and continuation')
parser.add_argument('--save_every', type=int, default=50, help='number of batches to save after')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs to loop over train data')


args = parser.parse_args()
print("Args: ", args, file=sys.stderr)

os.environ['CUDA_VISIBLE_DEVICES']="2"

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
coefs, scorer_info, scorer_config = load_scorers(args.scorers)
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
coef_trainer = CoefTrainer(len(scorers), args.ranking_loss, args.lr, coefs)

count, batch = 0, 0
bsz = args.batch_size
avg, a_n = 0, 0  # used for tracking writing coefs

with open(args.infile, 'r') as fin, open(args.outfile, 'w') as fout:
    sline = fin.readline().strip()
    slines, cont_lines = [], []
    for epoch in range(args.epochs):
        for sline in fin:
            if count % bsz == 0 and count:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(slines, sampling=True, sampling_topk=5, lenpen=2.0,
                                                   max_len_b=250, min_len=55, no_repeat_ngram_size=3,
                                                   rescore=True, coefs=coefs, scorers=scorers,
                                                   learn=True, dedup=args.dedup, gold_tokens=cont_lines,
                                                   coef_trainer=coef_trainer,
                                                   learn_every_token=args.learn_every_token)

                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis.replace('\n', '') + '\n')
                    fout.flush()
                slines, cont_lines = [], []
                batch += 1

            sline, cont_line = sline.strip().split(args.split, 1)
            slines.append(sline)
            cont_lines.append(cont_line)

            if count == 0:
                print("Example Data:\nContext: {}\n Continuation: {}".format(sline, cont_line))

            count += 1

            if batch and batch % args.save_every == 0:
                # avg and a_n init to None and 0. a_n seems to just track the saves
                with open(args.scorers, 'w') as out:
                    if avg is None:
                        avg = coef_trainer.weight_model.coefs.weight.data.cpu().squeeze().clone()
                    else:
                        avg += coef_trainer.weight_model.coefs.weight.data.cpu().squeeze()
                    a_n += 1
                    for s, coef in enumerate(avg.numpy() / a_n):
                        scorer_config[s][0] = str(coef)
                        out.write('%s\n' % '\t'.join(scorer_config[s]))
                    print("Writing coefficients: ", avg / a_n, file=sys.stderr)
        # if slines != []:
        #     hypotheses_batch = bart.sample(slines, sampling=True,  sampling_topk=5, lenpen=2.0, max_len_b=250, min_len=55, no_repeat_ngram_size=3)
        #     for hypothesis in hypotheses_batch:
        #         fout.write(hypothesis.replace('\n','') + '\n')
        #         fout.flush()
