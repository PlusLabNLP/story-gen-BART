import json
import argparse

def change_format(input, output):
    with open(input ,"r") as fin:
        data = json.load(fin)
        all_storyline = ''
        for item in data:
            one_storyline = ''
            for SRL in item:
                if SRL == "#":
                    one_storyline += ' # '
                elif SRL == '</s>':
                    one_storyline += " </s> "
                else:
                    one_storyline += ' '.join(tag + ' ' + word for tag, word in SRL.items())
            print(one_storyline)
            all_storyline += one_storyline + '\n'
    with open(output, 'w') as fout:
        fout.write(all_storyline)

def unpack_json(input, output):
    with open(input ,"r") as fin:
        data = json.load(fin)
        all = ''
        for item in data:
            for i in item:
                all += i + '\n'
    with open(output, 'w') as fout:
        fout.write(all)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='path to input file')
    parser.add_argument('--output_file', type=str, help='path to output file')
    args = parser.parse_args()
    change_format(args.input_file, args.output_file)
    # unpack_json(args.input_file, args.output_file)

if __name__ == '__main__':
    main()