'''
A script that combines a context and continuation file in a directory with a provided separator
Command to run: python combine_context_continuation_with_sep.py context_file continuation_file out_file separator
'''

import argparse, sys

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('context_file', type=str, help='file with lines of context')
    p.add_argument('continuation_file', type=str, help='file with lines of continuation')
    p.add_argument('out_file', type=str, help='name of file to write to')
    p.add_argument('sep', type=str, help='character to separate the context from continuation')
    args = p.parse_args()

    print("Args: ", args, file=sys.stderr)
    with open(args.context_file, "r") as context, open(args.continuation_file, "r") as continuation:
        context_lines, continuation_lines = context.readlines(), continuation.readlines()
        assert(len(context_lines) == len(continuation_lines)), "context and continuation lines must be equal"
        combined_lines = ["{0} {1} {2}".format(
            context_lines[i].strip(), args.sep, continuation_lines[i].strip()) for i in range(len(context_lines))]
    with open(args.out_file, "w") as outfile:
        outfile.write("\n".join(combined_lines))


    