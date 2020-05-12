import argparse
import os
import re
from mosestokenizer import MosesDetokenizer


special_chars = re.compile(r"</s>|ent\s\d+")
p = re.compile(r"<P>")
unk = re.compile(r"\[\w{1,5}\]")

def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-d', dest='input_dir')
    p.add_argument('-f', dest='files', nargs='+')
    p.add_argument('--detokenize', action='store_true')
    return p.parse_args()


def strip_chars(line: str):
    line = special_chars.sub("", line)
    cleanline = re.sub("\s+", " ", line)
    return cleanline


def make_human_readable(files: list, detokenize: bool):
    if detokenize:
        detokenizer = MosesDetokenizer("en")
    print("Postprocessing on {} files...".format(len(files)))
    for file in files:
        print("Working on: {}".format(file))
        with open(file, "r") as fin, open(file+".human_readable", "w") as fout:
            for line in fin:
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

    make_human_readable(files, detokenize=args.detokenize)



