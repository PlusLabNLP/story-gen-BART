
import re
import argparse
import sys

def make_refer(gold_data, num):
    with open(gold_data, 'r') as fin:
        data = fin.readlines()
        for idx, line in enumerate(data):
            line = line.split('<EOL>')[1]
            line = re.sub(r'(<ent> [0-9]*) | (</ent> [0-9]*)','', line)
            line = line.replace('</s>', '')
            line = line.replace('<P>', '')
            data[idx] = line
        refer_data = data[:num]
    return refer_data



if __name__ == "__main__":
    refer_data = make_refer(sys.argv[1], int(sys.argv[2]))
    with open(sys.argv[3], 'w') as fout:
        fout.write(''.join(refer_data))