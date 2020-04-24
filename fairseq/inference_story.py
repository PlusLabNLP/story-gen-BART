import argparse

import sys
import os
import copy
import torch
from fairseq.models.bart import BARTModel
from fairseq.models.roberta import RobertaModel
from utils import load_scorers


parser = argparse.ArgumentParser()
parser.add_argument('src')
parser.add_argument('out')
parser.add_argument('--batch_size', type=int, default=28)

args = parser.parse_args()
print("Args: ", args, file=sys.stderr)

os.environ['CUDA_VISIBLE_DEVICES']="3"

### load BART model                                                                                                                                                                 
bart = BARTModel.from_pretrained(
    'checkpoint-fullstory/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='fullstory'
)


bart.cuda() # remove this line if not running with cuda                                                                                               
bart.eval()
bart.half() # doesn't work with CPU   


count = 1
bsz = args.batch_size

with open(args.src) as source, open(args.out, 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, sampling=True, sampling_topk=5 ,lenpen=2.0,
                                               max_len_b=250, min_len=55, no_repeat_ngram_size=3,
                                               rescore=False)

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
