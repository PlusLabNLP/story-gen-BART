import argparse
from pathlib import Path
import sys
import time

import torch

from fairseq.models.bart import BARTModel
from fairseq.models.roberta import RobertaModel
from utils import load_scorers

parser = argparse.ArgumentParser(add_help=False)
parser.description = "Generates plots from prompts"

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
required.add_argument('--infile', type=Path, help='Input file (e.g., val.source)', required=True)
required.add_argument('--outfile', type=Path, help='Output file (e.g., val.hypo)', required=True)
optional.add_argument('--apply_disc', action='store_true',
                      help='Whether to use discriminators to rescore')
optional.add_argument('--scorers', type=str, default='checkpoint/WP_scorers.tsv', help='TSV with discriminator info')
optional.add_argument('--batch_size', type=int, default=1)
optional.add_argument('--dedup', action='store_true')
optional.add_argument('--banned_tok', nargs='+', default=["[", " [", "UN", " UN"], help="tokens to prevent generating")
optional.add_argument('--max_len', type=int, default=250, help="Max length of generation in BPE tok")
required.add_argument('--bart_dir', type=Path, help="Path to directory with BART checkpoint", required=True)
optional.add_argument('--checkpoint', type=str, default='checkpoint_best.pt',
                    help="Which checkpoint file to use for BART")

args = parser.parse_args()
print("Args: ", args, file=sys.stderr)

use_cuda = torch.cuda.is_available()
# Load BART model
bart_path = args.bart_dir
bart_checkpoint = args.checkpoint
bart = BARTModel.from_pretrained(
    bart_path,
    checkpoint_file=bart_checkpoint,
    data_name_or_path='full'
)


bart.eval()

if use_cuda:
    bart.cuda()
    bart.half()

### load discriminators
scorers, coefs = [], []
if args.apply_disc:
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


count = 1
bsz = args.batch_size
pad_toks = {0,2}
banned_verbs, banned_ids = [], []

if args.dedup:
    verb_strings = ["<V>", " <V>"]
    verb_tensors = [bart.encode(v) for v in verb_strings]
    banned_verbs = list(set([i.data.item() for t in verb_tensors for i in t]) - pad_toks)

if args.banned_tok:
    banned_tok_tensors = [bart.encode(t) for t in args.banned_tok]
    banned_ids = list(set([i.data.item() for t in banned_tok_tensors for i in t]) - pad_toks)


with open(args.infile, 'r') as fin, open(args.outfile, 'w') as fout:
    sline = fin.readline().strip()
    slines = [sline]
    print("Example Data: {}".format(sline.strip()))
    for sline in fin:
        if count % bsz == 0 and count:
            start_time = time.time()
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, sampling=True, sampling_topk=5, lenpen=2.0,
                                               max_len_b=args.max_len, min_len=55, no_repeat_ngram_size=3,
                                               rescore=args.apply_disc,
                                               coefs=coefs, scorers=scorers, dedup=args.dedup, 
                                               banned_toks=banned_ids, verb_idxs=banned_verbs)
            elapsed = time.time() - start_time
            print("Seconds per batch: {}".format(elapsed))
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis.replace('\n', '') + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        with torch.no_grad():
            hypotheses_batch = bart.sample(slines, sampling=True, sampling_topk=5, lenpen=2.0,
                                               max_len_b=args.max_len, min_len=55, no_repeat_ngram_size=3,
                                               rescore=args.apply_disc,
                                               coefs=coefs, scorers=scorers, dedup=args.dedup,
                                               banned_toks=banned_ids, verb_idxs=banned_verbs, learn=False)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis.replace('\n','') + '\n')
            fout.flush()