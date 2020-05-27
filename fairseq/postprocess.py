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
    p.add_argument('--concat_titles', action='store_true')
    p.add_argument('--titles', type=str, help="path to title file if concat titles")
    p.add_argument('--remove_partial', action='store_true',
                   help="if True removes partial sentences that were truncated in generation")
    p.add_argument('--sent_sym', default='</s>', type=str, help='if removing partial sentences, '
                                                           'delimiting symbol')
    return p.parse_args()


def strip_chars(line: str):
    line = special_chars.sub("", line)
    cleanline = re.sub("\s+", " ", line)
    return cleanline


def make_human_readable(files: list, detokenize: bool,
                        remove_partial_sent: bool=True, sent_sym: str="</s>"):
    if detokenize:
        detokenizer = MosesDetokenizer("en")
    print("Postprocessing on {} files...".format(len(files)))
    for file in files:
        print("Working on: {}".format(file))
        with open(file, "r") as fin, open(file+".human_readable", "w") as fout:
            for line in fin:
                if remove_partial_sent:
                    line = line[:line.rfind(sent_sym)]
                cleanline = strip_chars(line)
                if detokenize:
                    cleanline = detokenizer(cleanline.strip().split())
                fout.write("{}\n".format(cleanline))


def concat_title_plot(title_file, plot_files):
    with open(title_file, "r") as t_fin:
        all_titles = t_fin.readlines()
    for file in plot_files:
        with open(file, "r") as fin, open(file+".title+plot", "w") as fout:
            for i, line in enumerate(fin):
                fout.write("{} <EOT> {}\n".format(all_titles[i].strip(), line.strip()))


if __name__ == "__main__":
    args = setup_argparse()

    if args.input_dir:
        with os.scandir(args.input_dir) as source_dir:
            files = sorted([file.path for file in source_dir if
                            file.is_file() and not file.name.startswith('.')])
    else:
        files = args.files

    if args.concat_titles:
        concat_title_plot(args.titles, files)
    else:
        make_human_readable(files, detokenize=args.detokenize,
                        remove_partial_sent=args.remove_partial, sent_sym=args.sent_sym)



