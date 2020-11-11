"""
script that splits data in a given directory (when appropriately named based on earlier use of
split_data.py).
--sent_sym is used to split the data into context and continuation.
If just want to split the data once, and a delimiter occurs once per line, then you can just put
whatever other symbol you like in for sent_sym, and it will work (though the delimiter will end up
in continuation but not in context unless you specify to keep it on the context with the flag).
"""

import argparse
import logging
import os
from pathlib import Path
import random
import sys
from typing import Any, List


class ParserWithUsage(argparse.ArgumentParser):
    """ A custom parser that writes error messages followed by command line usage documentation."""

    def error(self, message) -> None:
        """
        Prints error message and help.
        :param message: error message to print
        """
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def no_rep_shuffle(l: List[Any]) -> List[Any]:
    """
    Shuffle a list such that no element stays fixed.
    Args:
        l: a list to shuffle

    Returns:
        Shuffled version of the input.
    """
    if len(l) <= 1:  # Can't shuffle an empty list, returning the original list
        return l
    l = list(zip(l, range(len(l))))
    nu_l = l[:]
    # Shuffle the list until no object is at the same location as in the original list.
    while True:
        random.shuffle(nu_l)
        for x, y in zip(l, nu_l):
            if x == y:
                break
        else:
            break
    # Return only the first element of the unzip, i.e. the shuffled list
    return next(zip(*nu_l))


def make_shuffled_keywords(text: List[str], sent_sym: str, remove_char: str = "<EOL>") -> str:
    """
    Takes list of strings, and characters to remove, and returns shuffled version with
    remove_chars at end.

    Each string is first shuffled internally (the token order is random). Then, the order of the
    strings themselves is shuffled such that the output contains the shuffled strings with their
    tokens also shuffled.

    Args:
        text: text to shuffle (list of sentences)
        sent_sym: sentence delimiter
        remove_char: character to remove from the string

    """
    import re
    # Create a regular expression to remove remove_char
    remove = re.compile(remove_char)
    # Remove the character from the each input string
    clean_str = [remove.sub("", text) for text in text]
    # Shuffle the tokens or each string
    internally_shuffled = [" ".join(no_rep_shuffle(phrases.strip().split())) for phrases in
                           clean_str]
    # Now shuffle the order of the strings
    shuffle_str = f" {sent_sym} ".join(no_rep_shuffle(internally_shuffled))
    return shuffle_str.strip() + " " + remove_char


def main():
    """
    Main method
    """
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO,
                        datefmt='%m/%d/%Y %H:%M:%S')
    parser = ParserWithUsage('Split text data into context and continuation')
    parser.add_argument('data_dir', type=str,
                        help='directory with data splits in it')
    parser.add_argument('--filenames', nargs='+', type=str, help='names of files to work on',
                        default=['disc_train.txt'])
    parser.add_argument('--out_dir', type=str,
                        help='if given, write files to a different directory than the read '
                             'directory')
    parser.add_argument('--len_context', type=int, default=1,
                        help='number of sentences in context')
    parser.add_argument('--len_continuation', type=int, default=None,
                        help='number of sentences in continuation. If none given, uses all')
    parser.add_argument('--doc_level', action='store_true',
                        help='use this flag if each line in the dataset is a document')
    parser.add_argument('--sent_sym', type=str, default='</s>',
                        help='the sentence delimiter to use')
    parser.add_argument('--keep_split_context', action='store_true',
                        help='by default the first split character is added to the continuation. '
                             'Use this flag to keep it on context.')
    args = parser.parse_args()
    logging.info("STARTED")
    logging.info(args)

    for filename in args.filenames:
        incomplete_lines = 0
        logging.info("Working on {}".format(filename))
        contexts, continuations, intra_shuffled_continuations = [], [], []
        with open(os.path.join(args.data_dir, filename), 'r') as lines:
            if args.doc_level:
                pass
                # # TODO make this section work (wasn't working in L2w, but also wasn't used)
                # assert(args.sent_sym is not None)
                # for line in lines:
                #     sents = line.strip().split(args.sent_sym)
                #     context = (' %s ' % args.sent_sym).join(sents[:args.len_context])
                #     continuation = (' %s ' % args.sent_sym).join(sents[args.len_context:(
                #     args.len_context+args.len_continuation)])
                #     contexts.append(context)
                #     continuations.append(continuation)
            else:
                for i, line in enumerate(lines):
                    sentences = line.strip().split(args.sent_sym)
                    # need to add a check here in case the line starts with sentence symbols (we
                    # don't want context to be empty, which it will be otherwise)

                    # The following line locates the first sentence that is not just empty spaces.
                    shift_up = next(idx for idx, tok in enumerate(sentences) if
                                    tok.strip())  # after strip whitespace eval to False,
                    # so gets first "real" token
                    split_point = args.len_context + shift_up  # if first token is valid shift up
                    # is 0
                    if len(sentences) <= split_point:
                        # The split point is outside the document. We cannot split.
                        incomplete_lines += 1
                        continue
                    else:
                        context = sentences[:split_point]
                        if args.len_continuation:
                            continuation = sentences[split_point:args.len_continuation]
                        else:
                            continuation = sentences[split_point:]
                        # print(sentences)
                        context = f" {args.sent_sym} ".join(context)
                        shuffled_continuation = make_shuffled_keywords(continuation,
                                                                       sent_sym=args.sent_sym)
                        continuation = f" {args.sent_sym} ".join(continuation)

                        if args.keep_split_context:
                            context = context + args.sent_sym
                        else:
                            continuation = args.sent_sym + continuation
                            shuffled_continuation = args.sent_sym + " " + shuffled_continuation
                    contexts.append(context)
                    continuations.append(continuation)
                    intra_shuffled_continuations.append(shuffled_continuation)

            logging.info(
                f"{incomplete_lines} lines were too short for sufficient context and were skipped")

        if args.out_dir:
            output = Path(args.out_dir)
        else:
            output = Path(args.data_dir)

        # Write the contexts
        with open(output / f"{filename}.context", mode="w") as out:
            out.write('\n'.join(contexts))
        # Write the continuations
        with open(output / f"{filename}.true_continuation", mode="w") as out:
            out.write('\n'.join(continuations))
        # Write the sentence-shuffled continuation
        with open(output / f"{filename}.shuffled_continuation", mode="w") as out:
            out.write('\n'.join(no_rep_shuffle(continuations)))
        # Write the sentence- and token-shuffled continuation
        with open(output / f"{filename}.all_shuffled_continuation", mode="w") as out:
            out.write('\n'.join(intra_shuffled_continuations))
    logging.info("DONE")

if __name__ == "__main__":
    main()
