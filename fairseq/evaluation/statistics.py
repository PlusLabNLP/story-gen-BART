"""script that prints out misc metrics on story generations (or gold stories). Metrics:
* Vocab, Vocab:Token Ratio
* # Unique verbs
* % Diverse verbs (may not be valid as we remove really common ones)
* # unique entities per story
* length of generations
"""

import argparse
import re
import sys
from collections import Counter

from numpy import mean

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--infile', type=str, nargs="+", help='files to run metrics on')
    p.add_argument('--gen_type', type=str, choices=["srl_storyline", "story"],
                   help='the type of generation to get metrics on')
    p.add_argument('--sep', type=str, help="character to split on if only scoring part. Will only split once into two parts")
    p.add_argument('--score_section', type=str, choices=["context", "continuation"],
                   help="if splitting the input, which part to use")
    p.add_argument('--num_freq_verbs', type=int, default=5, help="the number of verbs to consider the most frequent")
    args = p.parse_args()

    print("Processing Metrics for {} files".format(len(args.infile)), file=sys.stderr)

    verbs = re.compile("(?<=<V>)[\s\w]+(?=[#<])") # finds things with <V> tag
    entity_numbers = re.compile("(?<=ent\s)\d+")

    for filepath in args.infile:
        with open(filepath, "r") as fin:
            all_tokens, verb_tokens = Counter(), Counter()
            ent_per_story, tok_per_story = [], []
            lines = fin.readlines()
            print("-"*89)
            print("Metrics for {} lines in file: {}".format(len(lines), filepath))
            for line in lines:
                if not line:
                    continue
                if args.sep:
                    line = line.split(args.sep, 1)
                    if len(line) < 2:
                        #print("Skipping", file=sys.stderr)
                        continue
                    line = line[0] if args.score_section == "context" else line[1]
                toks = line.split()
                all_tokens.update(toks)
                tok_per_story.append(len(toks))
                all_verbs = [v.strip() for v in verbs.findall(line)]
                verb_tokens.update(all_verbs)
                all_entities = set(entity_numbers.findall(line))
                ent_per_story.append((len(all_entities)))

        total_verb_toks = sum(verb_tokens.values())
        common_verbs = sum([v for k,v in verb_tokens.most_common(args.num_freq_verbs)])
        # Print out min, max, average of ent per sample, as well as overall vocab, etc
        #print(verb_tokens)
        print("Length of samples (num tokens):")
        print("Min: {:.2f} Max: {:.2f} Mean: {:.2f}".format(min(tok_per_story), max(tok_per_story),
                                                            mean(tok_per_story)))
        print("Entities per story:")
        print("Min: {:.2f} Max: {:.2f} Mean: {:.2f}".format(min(ent_per_story), max(ent_per_story),
                                                            mean(ent_per_story)))
        print("Total Vocab: {}, Vocab:Token Ratio: {}".format(len(all_tokens),
                                                              (len(all_tokens)/sum(all_tokens.values()))*100))
        print("Unique Verbs: {}, % Diverse Verbs (thresh {}): {}".format(len(verb_tokens),
                                                                          args.num_freq_verbs,
                                                                         ((total_verb_toks-common_verbs)/total_verb_toks)*100))
        print("-" * 89)