import argparse
import os
import random
import re
from mosestokenizer import MosesDetokenizer
from nltk import sent_tokenize


special_chars = re.compile(r"</s>|ent\s\d+")
p = re.compile(r"<P>")
unk = re.compile(r"\[\w{1,5}\]")

def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-d', dest='input_dir')
    p.add_argument('-f', dest='files', nargs='+')
    p.add_argument('--detokenize', action='store_true')
    p.add_argument('--remove_partial', action='store_true',
                   help="if True removes partial sentences that were truncated in generation")
    p.add_argument('--sent_sym', default='</s>', type=str, help='if removing partial sentences, '
                                                           'delimiting symbol')
    p.add_argument('--shuffle', action='store_true', help='create a shuffled version for coherence eval')
    p.add_argument('--needs_sent_tokenize', action='store_true', help='for removing partial sentences or '
                                                                'shuffling for stories without sentence symbols')
    return p.parse_args()


def strip_chars(line: str):
    line = special_chars.sub("", line)
    cleanline = re.sub("\s+", " ", line)
    return cleanline


def make_human_readable(files: list, detokenize: bool,
                        remove_partial_sent: bool=True, sent_sym: str="</s>",
                        shuffle: bool=False, needs_sent_tokenize: bool=False):
    if detokenize:
        detokenizer = MosesDetokenizer("en")
    print("Postprocessing on {} files...".format(len(files)))
    for file in files:
        print("Working on: {}".format(file))
        outfile = file+".human_readable" if not shuffle else file+".shuffle"
        with open(file, "r") as fin, open(outfile, "w") as fout:
            for line in fin:
                if needs_sent_tokenize:
                    line = sent_sym.join(sent_tokenize(line))
                if remove_partial_sent:
                    line = line[:line.rfind(sent_sym)]
                if shuffle:
                    split_line = line.strip().split(sent_sym)
                    random.shuffle(split_line)
                    line = sent_sym.join(split_line)
                cleanline = strip_chars(line)
                if detokenize:
                    cleanline = detokenizer(cleanline.strip().split())
                fout.write("{}\n".format(cleanline))


if __name__ == "__main__":
    args = setup_argparse()

    if args.input_dir:
        with os.scandir(args.input_dir) as source_dir:
            files = sorted([file.path for file in source_dir if
                            file.is_file() and not file.name.startswith('.')])
    else:
        files = args.files

    make_human_readable(files, detokenize=args.detokenize,
                        remove_partial_sent=args.remove_partial, sent_sym=args.sent_sym,
                        shuffle=args.shuffle, needs_sent_tokenize=args.needs_sent_tokenize)


