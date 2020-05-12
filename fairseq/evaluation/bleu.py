import sys
import os
import pprint
import math
import re
c=0
r=0
def getCAndR(candidateSentence,referenceSentences):
    global c
    global r
    referenceCount=[]
    referenceLength=[]
    candidateSentence = list(candidateSentence)
    referenceSentences = list(referenceSentences)
    c+=len(candidateSentence)
    for index3 in range(0,len(referenceSentences)):
        candidateSentence = list(candidateSentence)
        referenceSentences = list(referenceSentences)
        # print(type(referenceSentences[index3]))
        # print(type(candidateSentence))
        referenceCount.append(abs(len(list(referenceSentences[index3]))-len(candidateSentence)))
        referenceLength.append(len(list(referenceSentences[index3])))
    r+=referenceLength[referenceCount.index(min(referenceCount))]
    #print c,r

def getBP():
    if c>=r:
        return 1
    else:
        return math.exp(1-r/float(c))



def getFiles(candidatePath,referencePath):
    candidatefile=candidatePath
    referencefiles=[]
    if os.path.isfile(referencePath):
        referencefiles.append(referencePath)
    else:
        referencefiles=os.listdir(referencePath)
        for i in range(0,len(referencefiles)):
            referencefiles[i]=referencePath+"/"+referencefiles[i]
    return candidatefile,referencefiles

def readFiles(candidatefile,referencefiles):
    candidateData=[]
    referencesData=[]
    with open(candidatefile) as fp:
        for line in fp:
            candidateData.append(line.strip().split(" "))
    for i in range(0,len(referencefiles)):
        temp=[]
        with open(referencefiles[i]) as fp:
            for line in fp:
                temp.append(line.strip().split(" "))
        referencesData.append(temp)
    return candidateData,referencesData

def uniGramDictionary(sentence):
    dictionary={}
    #sentence = sentence[0]
    i = 0
    sentence = list(sentence)
    # print(type(sentence))
    while i < len(sentence):
        unigram=sentence[i]
        #print "unigram:", unigram
        if unigram in dictionary:
            dictionary[unigram]+=1
        else:
            dictionary[unigram]=1
        i += 1
    return dictionary
def biGramDictionary(sentence):
    dictionary={}
    sentence = list(sentence)
    i = 0
    while i < len(sentence):
        if i+1 >= len(sentence):
            break
        bigram="".join(sentence[i])+" "+"".join(sentence[i+1])
        #print "bigram:", bigram
        if bigram in dictionary:
            dictionary[bigram]+=1
        else:
            dictionary[bigram]=1
        i += 1
    return dictionary
def triGramDictionary(sentence):
    sentence = list(sentence)
    dictionary={}
    #sentence = sentence[0]
    i = 0
    while i < len(sentence):
        if i+2 >= len(sentence):
            break
        trigram="".join(sentence[i])+" "+"".join(sentence[i+1])+" "+"".join(sentence[i+2])
        #print "trigram:", trigram
        if trigram in dictionary:
            dictionary[trigram]+=1
        else:
            dictionary[trigram]=1
        i += 1
    return dictionary
def quadrupleGramDictionary(sentence):
    sentence = list(sentence)
    dictionary={}
    #sentence = sentence[0]
    i = 0
    while i < len(sentence):
        if i+3 >= len(sentence):
            break
        quadruplegram="".join(sentence[i])+" "+"".join(sentence[i+1])+" "+"".join(sentence[i+2])+" "+"".join(sentence[i+3])
        #print "quadruplegram:", quadruplegram
        if quadruplegram in dictionary:
            dictionary[quadruplegram]+=1
        else:
            dictionary[quadruplegram]=1
        i += 1

    return dictionary
def uniGram(candidateSentence,referenceSentences):
    referenceDict=[]
    reference=[]
    #candidateSentence=candidateSentence.lower().split()
    candidateSentence=filter(None,candidateSentence)
    candidateDict = uniGramDictionary(candidateSentence)
    count=0
    for line in referenceSentences:
        #line=line.lower().split()
        line=filter(None,line)
        reference.append(line)
        referenceDict.append(uniGramDictionary(line))
    getCAndR(candidateSentence,reference)
    for word in candidateDict:
        #print "word in candidateDict:", word
        maxRefIndex=0
        for index2 in range(0,len(referenceDict)):
            if word in referenceDict[index2]:
                maxRefIndex=max(maxRefIndex,referenceDict[index2][word])
                
        count+=min(candidateDict[word],maxRefIndex)
        #print count
    sumngram=0
    for values in candidateDict.values():
        sumngram+=values
    return count,sumngram

def biGram(candidateSentence,referenceSentences):
    referenceDict=[]
    #candidateSentence=candidateSentence.lower().split()
    candidateSentence=filter(None,candidateSentence)
    candidateDict = biGramDictionary(candidateSentence)
    count=0
    for line in referenceSentences:
        #line=line.lower().split()
        line=filter(None,line)
        referenceDict.append(biGramDictionary(line))
    for word in candidateDict:
        maxRefIndex=0
        for index2 in range(0,len(referenceDict)):
            if word in referenceDict[index2]:
                maxRefIndex=max(maxRefIndex,referenceDict[index2][word])
        count+=min(candidateDict[word],maxRefIndex)
    sumngram=0
    for values in candidateDict.values():
        sumngram+=values
    return count,sumngram

def triGram(candidateSentence,referenceSentences):
    referenceDict=[]
    #candidateSentence=candidateSentence.lower().split()
    candidateSentence=filter(None,candidateSentence)
    candidateDict = triGramDictionary(candidateSentence)
    count=0
    for line in referenceSentences:
        #line=line.lower().split()
        line=filter(None,line)
        referenceDict.append(triGramDictionary(line))
    for word in candidateDict:
        maxRefIndex=0
        for index2 in range(0,len(referenceDict)):
            if word in referenceDict[index2]:
                maxRefIndex=max(maxRefIndex,referenceDict[index2][word])
                
        count+=min(candidateDict[word],maxRefIndex)
    sumngram=0
    for values in candidateDict.values():
        sumngram+=values
    return count,sumngram

def quadrupleGram(candidateSentence,referenceSentences):
    referenceDict=[]
    #candidateSentence=candidateSentence.lower().split()
    candidateSentence=filter(None,candidateSentence)
    candidateDict = quadrupleGramDictionary(candidateSentence)
    count=0
    for line in referenceSentences:
        #line=line.lower().split()
        line=filter(None,line)
        referenceDict.append(quadrupleGramDictionary(line))
    for word in candidateDict:
        maxRefIndex=0
        for index2 in range(0,len(referenceDict)):
            if word in referenceDict[index2]:
                maxRefIndex=max(maxRefIndex,referenceDict[index2][word])
        count+=min(candidateDict[word],maxRefIndex)
    sumngram=0
    for values in candidateDict.values():
        sumngram+=values
    return count,sumngram

def getModifiedPrecision(candidateData,referencesData):
    uniNum=0
    uniDen=0
    biNum=0
    biDen=0
    triNum=0
    triDen=0
    quadrupleNum=0
    quadrupleDen=0
    for index in range(0,len(candidateData)):
        referenceSentences=[]
        candidateSentence=candidateData[index]
        for index1 in range(0,len(referencesData)):
            referenceSentences.append(referencesData[index1][index])
        #print candidateSentence
        #print referenceSentences[0]
        uniClipCount,uniCount=uniGram(candidateSentence,referenceSentences)
        uniNum+=uniClipCount
        uniDen+=uniCount
        biClipCount,biCount=biGram(candidateSentence,referenceSentences)
        biNum+=biClipCount
        biDen+=biCount
        triClipCount,triCount=triGram(candidateSentence,referenceSentences)
        triNum+=triClipCount
        triDen+=triCount
        quadrupleClipCount,quadrupleCount=quadrupleGram(candidateSentence,referenceSentences)
        quadrupleNum+=quadrupleClipCount
        quadrupleDen+=quadrupleCount
    # print(uniNum,uniDen)
    # print(biNum,biDen)
    # print(triNum,triDen)
    # print(quadrupleNum,quadrupleDen)
    if uniDen > 0:
        unigram1=uniNum/float(uniDen)
    else:
        unigram1 = 0
    if biDen > 0:
        bigram1=biNum/float(biDen)
    else:
        bigram1 = 0
    if triDen > 0 :
        trigram1=triNum/float(triDen)
    else:
        trigram1 = 0
    if quadrupleDen > 0:
        quadruplegram1=quadrupleNum/float(quadrupleDen)
    else:
        quadruplegram1 = 0

    # print(unigram1,bigram1,trigram1,quadruplegram1)
    bleu1 = 0
    bleu2 = 0
    bleu3 = 0
    bleu4 = 0
    if unigram1+bigram1+trigram1+quadruplegram1 == 0:
        bleu1 = 0
        bleu2 = 0
        bleu3 = 0
        bleu4 = 0
    else:
        if unigram1 > 0:
            # print("in 1",getBP())
            bleu1=getBP()*math.exp(math.log(unigram1))
        if unigram1 > 0 and bigram1 > 0 :
            # print("in 2",getBP())
            bleu2=getBP()*math.exp(0.5*math.log(unigram1)+0.5*math.log(bigram1))
        if unigram1 > 0 and bigram1 > 0 and trigram1 > 0 :
            # print("in 3",getBP())
            bleu3=getBP()*math.exp((1/3.0)*math.log(unigram1)+(1/3.0)*math.log(bigram1)+(1/3.0)*math.log(trigram1))
        if unigram1 >0 and bigram1 >0 and trigram1 > 0 and quadruplegram1 > 0:
            # print("in 4",getBP())
            bleu4=getBP()*math.exp(0.25*math.log(unigram1)+0.25*math.log(bigram1)+0.25*math.log(trigram1)+0.25*math.log(quadruplegram1))
    
    print("blue_1: {:.2f}\nblue_2: {:.2f}\nblue_3: {:.2f}\nblue_4: {:.2f}".format( bleu1*100, bleu2*100, bleu3*100, bleu4*100))
    return bleu1, bleu2, bleu3, bleu4
    # print("-" * 89)
    # print("%s has blue score: %f\t%f\t%f\t%f\n".format(sys.argv[1], bleu1, bleu2, bleu3, bleu4))
    # fp=open('bleu_out.txt','w')
    # fp.write("%s has blue score: %f\t%f\t%f\t%f\n" %(sys.argv[1], bleu1, bleu2, bleu3, bleu4))
    # fp.close()
def main(candidatefile,referencefiles):
    candidatefile, referencefiles = getFiles(candidatefile,referencefiles)
    candidateData, referencesData = readFiles(candidatefile, referencefiles)
    bleu1, bleu2, bleu3, bleu4 = getModifiedPrecision(candidateData,referencesData)
    return bleu1*100, bleu2*100, bleu3*100, bleu4*100

if __name__ == "__main__":
    candidatefile,referencefiles = getFiles(sys.argv[1],sys.argv[2])
    candidateData,referencesData=readFiles(candidatefile,referencefiles)
    """
    for item in candidateData:
        for word in item:
            print word,
        print
    """
    print('Bleu scores %:')
    getModifiedPrecision(candidateData,referencesData)
    print("-" * 89)
