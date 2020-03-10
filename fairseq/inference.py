import argparse

import sys
import os
import torch
from fairseq.models.bart import BARTModel
from utils import load_scorers

parser = argparse.ArgumentParser()
parser.add_argument('--apply-disc', action='store_true', help='whether to use discriminators to rescore')
parser.add_argument('--scorers', type=str, default='checkpoint/WP_scorers.tsv', help='tsv with discriminator info')

args = parser.parse_args()
print("Args: ", args, file=sys.stderr)

os.environ['CUDA_VISIBLE_DEVICES']="2, 3"

### load discriminators if using
scorer_config, scorers, coefs = [], [], []
if args.apply_disc:
    scorer_config, scorers, coefs = load_scorers(args.scorers) # will need to deal with the config only if learning

### load BART model
bart = BARTModel.from_pretrained(
    'checkpoint/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='temp'
)

#bart.cuda() # remove this line if not running with cuda
bart.eval()
#bart.half() # doesn't work with CPU
count = 1
bsz = 1 #28

with open('./temp/val.source.tiny') as source, open('temp/val.hypo.tiny', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, sampling=True, sampling_topk=5 ,lenpen=2.0,
                                               max_len_b=250, min_len=55, no_repeat_ngram_size=3,
                                               rescore=args.apply_disc, coefs=coefs, scorers=scorers, learn=False)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis.replace('\n','') + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, sampling=True,  sampling_topk=5, lenpen=2.0, max_len_b=250, min_len=55, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis.replace('\n','') + '\n')
            fout.flush()
