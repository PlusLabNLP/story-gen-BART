import sys, argparse, random, os

parser = argparse.ArgumentParser('Split text data into context and continuation')
parser.add_argument('data_dir', type=str,
                    help='directory with data splits in it')
parser.add_argument('out_dir', type=str,
                    help='directory to output data to')
parser.add_argument('--comp', type=str, required=True,
                    help='what adversarial example to compare to [lm, random, event]')
args = parser.parse_args()

def read_txt(fname):
    return open(fname).read().split('\n')

assert len(sys.argv) > 2, "Arguments required."

filenames = ['disc_train.txt', 'valid.txt', 'test.txt']
for filename in filenames:
    context = read_txt(os.path.join(args.data_dir, filename) + '.context')
    if args.comp == 'lm':
        comp_end = read_txt(os.path.join(args.data_dir, filename) + '.generated_continuation')
    elif args.comp == 'random':
        comp_end = read_txt(os.path.join(args.data_dir, filename) + '.shuffled_continuation')
    elif args.comp == 'event':
        comp_end = read_txt(os.path.join(args.data_dir, filename) + '.all_shuffled_continuation')
    else:
        assert(False)
    true_end = read_txt(os.path.join(args.data_dir, filename) + '.true_continuation')
    
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

    train_file = os.path.join(args.out_dir, filename[:-4] + '.tsv')
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

