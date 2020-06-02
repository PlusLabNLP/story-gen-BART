import sys, argparse, random, os, re

parser = argparse.ArgumentParser('Split text data into context and continuation')
parser.add_argument('data_dir', type=str,
                    help='directory with data splits in it')
parser.add_argument('out_dir', type=str,
                    help='directory to output data to')
parser.add_argument('--filenames', nargs='+', type=str, help='names of files to work on',
                    default=['disc_train.txt'] )
parser.add_argument('--comp', type=str, required=True,
                    help='what adversarial example to compare to [lm, random, event]')
parser.add_argument('--event_shuffle', type=str, required=False, default='inter',
                    help='what event shuffle strategy to use [random, intra, intraV, inter]')
args = parser.parse_args()


sent_delimiter='</s>'
event_delimiter= '#'

def read_txt(fname):
    return open(fname).read().split('\n')

assert len(sys.argv) > 2, "Arguments required."


def get_verbs(event):
    srl_list = re.split(r'(<.*?>)', event)
    if '<V>' not in srl_list:
        return -1, ''
    vidx = srl_list.index('<V>')
    return vidx+1, srl_list[vidx + 1]

def replace_verbs(events, swap_pairs, verbs_n_idxs):
    for bf, aft in swap_pairs:
        srl_list = re.split(r'(<.*?>)', events[bf])
        srl_list[verbs_n_idxs[bf][0]] = verbs_n_idxs[aft][1]
        events[bf] = ''.join(srl_list)
    return events


def event_intra_shuffle(true_end, only_verb=False):
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
            #assert events[-1] == ' ', events[-1]
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
                #shuffled_events = random.sample(events[:-1], len(events)-1)
                #shuffled_events += events[-1:]
                random.shuffle(events)
                shuffled_events = events
            new_sents.append(shuffled_events)
        shuffled_end.append(sent_delimiter.join(list(map(lambda x:event_delimiter.join(x), new_sents)) + sentences[-1:]))
    assert (len(shuffled_end) == len(true_end))
    print(len(true_end))
    print(true_end[0])  # TODO these maybe are not ideal print statements since the end may not be modified? But also need to check
    print(shuffled_end[0])
    print(count)
    return shuffled_end


def event_inter_shuffle(true_end):
    shuffled_end = []
    for line in true_end:
        sentences = line.split(sent_delimiter)
        #print(sentences)
        shuffled_sent = random.sample(sentences[:-1], len(sentences)-1)
        shuffled_sent += sentences[-1:]
        #print(shuffled_sent)
        shuffled_end.append(sent_delimiter.join(shuffled_sent))
    assert (len(shuffled_end) == len(true_end))
    print(len(true_end))
    #print(true_end[0])
    #print(shuffled_end[0])
    return shuffled_end
        

for filename in args.filenames:
    context = read_txt(os.path.join(args.data_dir, filename) + '.context')
    true_end = read_txt(os.path.join(args.data_dir, filename) + '.true_continuation')
    if args.comp == 'lm':
        comp_end = read_txt(os.path.join(args.data_dir, filename) + '.generated_continuation')
    elif args.comp == 'random':
        comp_end = read_txt(os.path.join(args.data_dir, filename) + '.shuffled_continuation')
    elif args.comp == 'event':
        if args.event_shuffle == 'random':
            comp_end = read_txt(os.path.join(args.data_dir, filename) + '.all_shuffled_continuation')
        elif args.event_shuffle == 'intra':
            comp_end = event_intra_shuffle(true_end)
        elif args.event_shuffle == 'intraV':
            comp_end = event_intra_shuffle(true_end, True)
        elif args.event_shuffle == 'inter':
            comp_end = event_inter_shuffle(true_end)
        else:
            raise ValueError
    else:
        assert(False)
    
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
            if not bool(comp.strip()) or not(bool(cont.strip())):
                incomplete_lines += 1
                continue
        tsv_lines.append(tsv_line)
    tag = args.event_shuffle if args.comp == "event" else '' # include event type if relevant
    train_file = os.path.join(args.out_dir, os.path.splitext(filename)[0] + '.'+ tag +'.tsv')
    with open(train_file, 'w') as out:
        out.write('\n'.join(tsv_lines))

    # validate that all lines have exactly 2 tabs
    invalid_lines = []
    with open(train_file, 'r') as fin:
        for i, line in enumerate(fin):
            num_examples_in_line = len(line.split("\t"))
            if num_examples_in_line != 4:
                invalid_lines.append((i, num_examples_in_line))

    print("Lines removed due to one or more continuations being empty: {}".format(incomplete_lines),
          file=sys.stderr)
    print("{} lines in file have too many or too few tabs\n"
          "Lines: {}\n Num items afer tab split: {}".format(len(invalid_lines),
                                                            [item[0] for item in invalid_lines],
                                                            [item[1] for item in invalid_lines]),
          file=sys.stderr)

