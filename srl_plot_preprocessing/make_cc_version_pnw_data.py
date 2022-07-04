"""
script that splits data in a given directory (when appropriately named based on earlier use of
split_data.py).
--sent_sym is used to split the data into context and continuation.
If just want to split the data once, and a delimiter occurs once per line, then you can just put
whatever other symbol you like in for sent_sym, and it will work (though the delimiter will end up
in continuation but not in context unless you specify to keep it on the context with the flag).
"""

import argparse, os, random, sys
import re

import numpy as np

parser = argparse.ArgumentParser('Split text data into context and continuation')
parser.add_argument('data_dir', type=str,
                    help='directory with data splits in it')
parser.add_argument('--out_dir', type=str,
                    help='if given, write files to a different directory than the read directory')
parser.add_argument('--len_context', type=int, default=1,
                    help='number of sentences in context')
parser.add_argument('--len_continuation', type=int, default=None,
                    help='number of sentences in continuation. If none given, uses all')
parser.add_argument('--doc_level', action='store_true',
                    help='use this flag if each line in the dataset is a document')
parser.add_argument('--sent_sym', type=str, default='</s>',
                    help = 'the sentence delimiter to use')
parser.add_argument('--keep_split_context', action='store_true',
                    help='by default the first split character is added to the continuation. Use this flag to keep it on context.')
args = parser.parse_args()
print(args, file=sys.stderr)

def no_rep_shuffle(l):
    if len(l) <= 1: #since then can't shuffle
        return l
    l = list(zip(l, range(len(l))))
    nu_l = l[:]
    while True:
        random.shuffle(nu_l)
        for x, y in zip(l, nu_l):
            if x == y:
                break
        else:
            break
    return next(zip(*nu_l))


def make_shuffled_keywords(str_list, remove_char="<EOL>"):
    """ takes list of strings, and characters to remove, and returns shuffled version with remove_chars at end """
    remove = re.compile("<EOL>")
    clean_str = [remove.sub("", text) for text in str_list]
    internally_shuffled = [" ".join(no_rep_shuffle(phrases.strip().split())) for phrases in clean_str]
    shuffle_str = (' %s ' % args.sent_sym).join(no_rep_shuffle(internally_shuffled))
    return shuffle_str.strip() + " " + remove_char

filenames = ['disc_train.txt', 'valid.txt', 'test.txt']


for filename in filenames:
    incomplete_lines = 0
    print("Working on {}".format(filename))
    contexts, continuations, intra_shuffled_continuations = [], [], []
    with open(os.path.join(args.data_dir, filename), 'r') as lines:
        if args.doc_level:
            # TODO make this section work (wasn't working in L2w, but also wasn't used)
            assert(args.sent_sym is not None)
            for line in lines:
                sents = line.strip().split(args.sent_sym)
                context = (' %s ' % args.sent_sym).join(sents[:args.len_context])
                continuation = (' %s ' % args.sent_sym).join(sents[args.len_context:(args.len_context+args.len_continuation)])
                contexts.append(context)
                continuations.append(continuation)
        else:
            for i, line in enumerate(lines):
                sentences = line.strip().split(args.sent_sym)
                if len(sentences) <= args.len_context:
                    incomplete_lines += 1
                    continue
                else:
                    context = sentences[:args.len_context]
                    if args.len_continuation:
                        continuation = sentences[args.len_context:args.len_continuation]
                    else:
                        continuation = sentences[args.len_context:]
                    #print(sentences)
                    context = (' %s ' % args.sent_sym).join(context)
                    shuffled_continuation = make_shuffled_keywords(continuation)
                    continuation = (' %s ' % args.sent_sym).join(continuation)

                    if args.keep_split_context:
                        context = context + args.sent_sym
                    else:
                        continuation = args.sent_sym + continuation
                        shuffled_continuation = args.sent_sym + shuffled_continuation
                contexts.append(context)
                continuations.append(continuation)
                intra_shuffled_continuations.append(shuffled_continuation)

                #if i % 100 == 0:
                #    print("finished {} lines".format(i), file=sys.stderr)
        print("{} lines were too short for sufficient context and were skipped".format(incomplete_lines),
              file=sys.stderr)

    if args.out_dir:
        output = args.out_dir
    else:
        output = args.data_dir

    with open(os.path.join(output, filename + '.context'), 'w') as out:
         out.write('\n'.join(contexts))
    with open(os.path.join(output, filename + '.true_continuation'), 'w') as out:
         out.write('\n'.join(continuations))
    with open(os.path.join(output, filename + '.shuffled_continuation'), 'w') as out:
         out.write('\n'.join(no_rep_shuffle(continuations)))
    with open(os.path.join(output, filename + '.all_shuffled_continuation'), 'w') as out:
        out.write('\n'.join(intra_shuffled_continuations))
