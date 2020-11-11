import argparse
from pathlib import Path

import sys
import os
import copy
import torch
from fairseq.models.bart import BARTModel
from fairseq.models.roberta import RobertaModel
from utils import load_scorers

parser = argparse.ArgumentParser(add_help=False)
parser.description = "Generates stories from plots"

required = parser.add_argument_group('Required arguments')
optional = parser.add_argument_group('Optional arguments')

# Add back help
optional.add_argument(
    '-h',
    '--help',
    action='help',
    default=argparse.SUPPRESS,
    help='Show this help message and exit'
)

required.add_argument('src', type=Path, help='Input file (e.g., val.source)')
required.add_argument('out', type=Path, help='Output file (e.g., val.hypo)')
optional.add_argument('--batch_size', type=int, default=1)
optional.add_argument('--banned_tok', nargs='+', default=["[", " [", "UN", " UN", "\n", " \n"], help="tokens to prevent generating")
optional.add_argument('--max_len', type=int, default=250, help="Max length of generation in BPE tok")
required.add_argument('--bart_dir', type=Path, help="Path to directory with BART checkpoint", required=True)
optional.add_argument('--checkpoint', type=str, default='checkpoint_best.pt',
                    help="Which checkpoint file to use for BART")

args = parser.parse_args()
print("Args: ", args, file=sys.stderr)

# Load BART model
bart_path = args.bart_dir
bart_checkpoint = args.checkpoint
bart = BARTModel.from_pretrained(
    bart_path,
    checkpoint_file=bart_checkpoint,
    data_name_or_path='fullstory'
)

bart.eval()
use_cuda = torch.cuda.is_available()
if use_cuda:
    bart.cuda()
    bart.half()

count = 1
bsz = args.batch_size
pad_toks = {0,2}
banned_ids = []

if args.banned_tok:
    banned_tok_tensors = [bart.encode(t) for t in args.banned_tok]
    banned_ids = list(set([i.data.item() for t in banned_tok_tensors for i in t]) - pad_toks)
    print("Banning token ids: {}".format(banned_ids))

with open(args.src) as source, open(args.out, 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, sampling=True, sampling_topk=5 ,lenpen=2.0,
                                               max_len_b=args.max_len, min_len=55, no_repeat_ngram_size=3,
                                               rescore=False, banned_toks=banned_ids)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis.replace('\n','') + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        with torch.no_grad():
            bart.sample(slines, sampling=True, sampling_topk=5 ,lenpen=2.0,
                                               max_len_b=args.max_len, min_len=55, no_repeat_ngram_size=3,
                                               rescore=False, banned_toks=banned_ids)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis.replace('\n','') + '\n')
            fout.flush()
