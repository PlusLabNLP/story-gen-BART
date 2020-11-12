import argparse
import logging
import os
from pathlib import Path
import random
import re
import sys
from typing import List, Union


def get_verbs(event):
    srl_list = re.split(r'(<.*?>)', event)
    if '<V>' not in srl_list:
        return -1, ''
    vidx = srl_list.index('<V>')
    return vidx + 1, srl_list[vidx + 1]


def replace_verbs(events, swap_pairs, verbs_n_idxs):
    for bf, aft in swap_pairs:
        srl_list = re.split(r'(<.*?>)', events[bf])
        srl_list[verbs_n_idxs[bf][0]] = verbs_n_idxs[aft][1]
        events[bf] = ''.join(srl_list)
    return events


def event_intra_shuffle(true_end: List[str], sent_delimiter: str, event_delimiter: str,
                        only_verb: bool = False) -> List[str]:
    """

    Args:
        true_end: list of documents
        sent_delimiter: symbol that delimits sentences
        event_delimiter: symbol that delimits events
        only_verb: Whether to shuffle only verbs, or verbs and their arguments.

    Returns:

    """
    shuffled_end = []
    count = 0
    for line in true_end:
        sentences = line.split(sent_delimiter)
        new_sents = []
        for sent in sentences[:-1]:
            events = sent.split(event_delimiter)
            if len(events) < 2:
                new_sents.append(events)
                continue
            if only_verb:
                verbs_n_idxs = list(map(lambda x: get_verbs(x), events))
                verb_idxs = [i for i in range(len(verbs_n_idxs)) if verbs_n_idxs[i][0] != -1]
                shuf_idxs = list(range(len(verb_idxs)))
                random.shuffle(shuf_idxs)
                swap_pairs = [(verb_idxs[i], verb_idxs[si]) for i, si in enumerate(shuf_idxs)]
                shuffled_events = replace_verbs(events, swap_pairs, verbs_n_idxs)
                '''
                if len(verb_idxs) < len(verbs_n_idxs)-2 and len(verb_idxs) > 5:
                    print (sent)
                    print (events)
                    print (verbs_n_idxs)
                    print (verb_idxs, shuf_idxs)
                    print (swap_pairs)
                    print (event_dilimiter.join(shuffled_events))
                    print (len(sent), len(event_dilimiter.join(shuffled_events)))
                    exit(0)
                for eve in events:
                    if eve.strip() == '':
                        new_sents.append([])
                        continue
                    elems = re.split(r'(<.*?>)', eve)
                    if len(elems) > 9 and '< P >' not in elems:
                        print(eve)
                        print(len(elems), sent, elems)
                        count += 1
                    vidx=find_verb(elems)
                    print(vidx)
                    print(elems[vidx+1])
                    exit(0)'''
            else:
                random.shuffle(events)
                shuffled_events = events
            new_sents.append(shuffled_events)
        shuffled_end.append(sent_delimiter.join(
            list(map(lambda x: event_delimiter.join(x), new_sents)) + sentences[-1:]))
    assert (len(shuffled_end) == len(true_end))
    return shuffled_end


def event_inter_shuffle(true_end: List[str], sent_delimiter: str) -> List:
    """
    Shuffle sentences in each document.
    Args:
        true_end: list of documents
        sent_delimiter: sentence delimiter

    Returns:
        A version of true_end where the sentences in each document are shuffled.
    """
    shuffled_end = []
    for line in true_end:
        sentences = line.split(sent_delimiter)
        shuffled_sent = random.sample(sentences[:-1], len(sentences) - 1)
        shuffled_sent += sentences[-1:]
        shuffled_end.append(sent_delimiter.join(shuffled_sent))
    assert (len(shuffled_end) == len(true_end))
    return shuffled_end


def read_txt(file_name: Union[Path, str]) -> List[str]:
    """
    Loads the content of a text file into a list.
    Args:
        file_name: name of file to open

    Returns:
        List of content, one element is one line
    """
    return open(file_name).read().split('\n')


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


def main():
    """
    Main method
    """
    shuf_random = "random"
    shuf_intra = "random"
    shuf_intrav = "intraV"
    shuf_inter = "inter"
    all_shuf_strategies = {shuf_random, shuf_inter, shuf_intra, shuf_intrav}

    adv_lm = "lm"
    adv_random = "random"
    adv_event = "event"
    all_adv = {adv_lm, adv_event, adv_random}

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO,
                        datefmt='%m/%d/%Y %H:%M:%S')

    parser = ParserWithUsage('Create training TSV file from continuation data')
    parser.add_argument('data_dir', type=Path, help='Directory with data splits and continuations.')
    parser.add_argument('out_dir', type=Path, help='Directory to output data to')
    parser.add_argument('--filenames', nargs='+', type=str, help='Names of files to work on',
                        default=['disc_train.txt'])
    parser.add_argument('--comp', type=str, required=True,
                        help=f'What adversarial example to compare to [{", ".join(all_adv)}]')
    parser.add_argument('--event_shuffle', type=str, required=False, default=shuf_inter,
                        help=f'What event shuffle strategy to use ['
                             f'{", ".join(all_shuf_strategies)}]')
    args = parser.parse_args()

    logging.info("STARTED")
    data_dir = args.data_dir
    sent_delimiter = '</s>'
    event_delimiter = '#'

    for filename in args.filenames:
        context = read_txt(data_dir / f"{filename}.context")
        true_end = read_txt(data_dir / f"{filename}.true_continuation")
        if args.comp == adv_lm:
            comp_end = read_txt(data_dir / f"{filename}.generated_continuation")
        elif args.comp == adv_random:
            comp_end = read_txt(data_dir / f"{filename}.shuffled_continuation")
        elif args.comp == adv_event:
            if args.event_shuffle == shuf_random:
                comp_end = read_txt(data_dir / f"{filename}.all_shuffled_continuation")
            elif args.event_shuffle == shuf_intra:
                comp_end = event_intra_shuffle(true_end, only_verb=False,
                                       sent_delimiter=sent_delimiter,
                                       event_delimiter=event_delimiter)
            elif args.event_shuffle == shuf_intrav:
                comp_end = event_intra_shuffle(true_end, only_verb=True,
                                       sent_delimiter=sent_delimiter,
                                       event_delimiter=event_delimiter)
            elif args.event_shuffle == shuf_inter:
                comp_end = event_inter_shuffle(true_end, sent_delimiter=sent_delimiter)
            else:
                raise ValueError
        else:
            raise ValueError

        tsv_lines = []
        randomize = False
        incomplete_lines = 0

        for cont, comp, true in zip(context, comp_end, true_end):
            tsv_line = cont.strip() + '\t'
            if randomize:
                if random.random() < 0.5:
                    tsv_line += comp.strip() + '\t' + true.strip() + '\t' + '1'
                else:
                    tsv_line += true.strip() + '\t' + comp.strip() + '\t' + '0'
            else:
                tsv_line += comp.strip() + '\t' + true.strip() + '\t' + '1'
                if not bool(comp.strip()) or not (bool(cont.strip())):
                    incomplete_lines += 1
                    continue
            tsv_lines.append(tsv_line)
        tag = args.event_shuffle if args.comp == "event" else ''  # include event type if relevant
        train_file = os.path.join(args.out_dir, os.path.splitext(filename)[0] + '.' + tag + '.tsv')
        with open(train_file, 'w') as out:
            out.write('\n'.join(tsv_lines))

        # validate that all lines have exactly 2 tabs
        invalid_lines = []
        with open(train_file, 'r') as fin:
            for i, line in enumerate(fin):
                num_examples_in_line = len(line.split("\t"))
                if num_examples_in_line != 4:
                    invalid_lines.append((i, num_examples_in_line))

        print("Lines removed due to one or more continuations being empty: {}".format(
            incomplete_lines),
            file=sys.stderr)
        print("{} lines in file have too many or too few tabs\n"
              "Lines: {}\n Num items afer tab split: {}".format(len(invalid_lines),
                                                                [item[0] for item in invalid_lines],
                                                                [item[1] for item in
                                                                 invalid_lines]),
              file=sys.stderr)

    logging.info("DONE")


if __name__ == "__main__":
    main()
