"""Script that calculates unigram, bigram, and trigram repetition between all stories generated.
Writes repetition to std out - lower numbers are better. Also writes raw counts to 3 files.
Also inherited and is super messy and hard coded and should be refactored."""



import sys

class NGram(object):
    def __init__(self, n):
        # n is the order of n-gram language model
        self.n = n
        self.unigram = {}
        self.bigram = {}
        self.trigram={}

    # scan a sentence, extract the ngram and update their frequence.
    # @param    sentence    list{str}
    # @return   none
    def scan(self, sentence):
        for line in sentence:
            self.ngram(line.split())
        
        if self.n == 1:
            try:
                fip = open("data.uni", "w")
            except:
                print("failed to open data.uni", file=sys.stderr)

            num = len(self.unigram)
            sum = 0
            for i in self.unigram:
                sum = sum + self.unigram[i]
                fip.write("%s %d\n" % (i, self.unigram[i]))

            proportion = (1.0 - float(num) / float(sum)) * 100
            print("unigram proportion: {:.2f} %".format(proportion))
        if self.n == 2:
            try:
                fip = open("data.bi", "w")
            except:
                print("failed to open data.bi", file=sys.stderr)
            num = len(self.bigram)
            sum = 0
            for i in self.bigram:
                sum = sum + self.bigram[i]
                fip.write("%s %d\n" % (i, self.bigram[i]))
            proportion = (1.0 -  float(num) / float(sum)) * 100
            print("bigram proportion: {:.2f} %".format(proportion))

        if self.n == 3:
            try:
                fip = open("data.tri", "w")
            except:
                print("failed to open data.bi", file=sys.stderr)
            num = len(self.trigram)
            sum = 0
            for i in self.trigram:
                sum = sum + self.trigram[i]
                fip.write("%s %d\n" % (i, self.trigram[i]))
            proportion = ((1.0 - float(num) / float(sum))) * 100
            print("trigram proportion: {:.2f} %".format(proportion))
        return proportion

    
    # @param    words       list{str}
    # @return   none
    def ngram(self, words):
        # uni-gram
        if self.n == 1:
            for word in words:
                if word not in self.unigram:
                    self.unigram[word] = 0
                self.unigram[word] = self.unigram[word] + 1

        # bi-gram
        if self.n == 2:
            for i in range(0, len(words)-1):
                stri = words[i] + words[i+1]
                if stri not in self.bigram:
                    self.bigram[stri] = 0
                self.bigram[stri] = self.bigram[stri] + 1
        # tri-gram
        if self.n == 3:
            for i in range(0, len(words)-2):
                stri = words[i] + words[i+1]+ words[i+2]
                if stri not in self.trigram:
                    self.trigram[stri] = 0
                self.trigram[stri] = self.trigram[stri] + 1

def main(human_readeable_story_file):
    with open(human_readeable_story_file, 'r') as fin:
        lines = fin.readlines()
    sentence = []
    for line in lines:
        if len(line.strip()) != 0:
            sentence.append(line.strip())
    tri = NGram(3)
    inter_tri = tri.scan(sentence)
    return inter_tri


# input file is the generated stories - the entire set of them, one per line.
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("sys.argv[1]: input file!", file=sys.stderr)
        exit()
    try:
        fip = open(sys.argv[1], "r")
    except:
        print("failed to open input file", file=sys.stderr)
    sentence = []
    for line in fip:
        if len(line.strip()) != 0:
            sentence.append(line.strip())
    uni = NGram(1)
    bi = NGram(2)
    tri=NGram(3)
    uni.scan(sentence)
    bi.scan(sentence)
    tri.scan(sentence)
    fip.close()
    print("-" * 89)
