import argparse
import sys
from rouge import FilesRouge, Rouge


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', dest="hyp_path", type=str, help="path for hypothesis")
    parser.add_argument('--ref', dest="ref_path", type=str, help="path for reference file")
    parser.add_argument('--average', action='store_true', help="whether to average over file instead of report line by line")
    args = parser.parse_args()
    
    files_rouge = FilesRouge()
    batch_size = 1024
    with open(args.hyp_path, "r") as hfile, open(args.ref_path, "r") as rfile:
        all_samples = hfile.readlines()
        num_hyp = len(all_samples)
        all_samples.extend(rfile.readlines())
        num_ref = len(all_samples) - num_hyp
        if num_hyp != num_ref:
            print("Mismatch number of hypothesis and reference: Hyp {} Ref {}".format(num_hyp, num_ref))
    seq_len = max([len(sample.split()) for sample in all_samples])
    min_lim = seq_len * batch_size + 10
    recursion_limit = sys.getrecursionlimit()
    if min_lim > recursion_limit:
        print("Recursion Limit: {}".format(recursion_limit), file=sys.stderr)
        sys.setrecursionlimit(min_lim)
        print("New Recursion Limit: {}".format(sys.getrecursionlimit()), file=sys.stderr)
    scores = files_rouge.get_scores(args.hyp_path, args.ref_path, avg=args.average)
    for key in scores: 
        subdict=scores[key] 
        print("{}:".format(key)) 
        [print("{}: {:.2f}".format(key, val)) for key, val in list(subdict.items())]

