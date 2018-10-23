import numpy as np
import pandas as pd
import string
from string import digits
import os.path
import pickle

#  exclude = set(string.punctuation).union(['>>','<<'])
english_letters = set(string.ascii_lowercase)
french_letters = set(['é','à', 'è', 'ù','â', 'ê', 'î', 'ô', 'û','ç','ë', 'ï', 'ü','á','ó','ā','œ',])
spanish_letters = set(['ñ','ó','í', 'á','é','í','ú','á'])
all_letters = set.union(english_letters,french_letters,spanish_letters)

def remove_punctuation(words):
    """
    words: list of strings
    s: list of strings where each string has been stripped of punctuations
    """
    s = []
    for w in words:
        w = w.lower()
        wnew = ''.join(ch for ch in w if ch in all_letters)
        if wnew:
            s.append(wnew)
    return s

def text2words(infile,outfile):
    fout = open(outfile,'w')
    fp = open(infile,'r') 
    for line in fp:
        words = remove_punctuation(line.split())
        for w in words:
            if len(w)<=2: # don't include words shorter than length 2
                continue
            fout.write("%s\n" % w)
    fp.close()
    fout.close()

def alphabets_from_file(filename):
    """
    in : filename: name of txt file
    out: symbols : set of unique symbols used in the file
    """
    symbols = set()
    with open(filename,'r') as fp:
        for line in fp:
            symbols = symbols.union(list(line))
    return symbols

if __name__ == "__main__":
    path = 'data/'
    infiles = ['english_text.txt','french_text.txt','spanish_text.txt']
    outfiles = ['english_words.csv','french_words.csv','spanish_words.csv']
    testname = 'testing.csv'
    trainname = 'training.csv'

    DF = []
    k = 0
    for fin,fout in zip(infiles,outfiles):
        # extract words from file
        text2words(path+fin,path+fout)

        # get unique words
        df = pd.read_csv(path+fout,names=['WORDS'])
        df.drop_duplicates('WORDS',inplace=True)
        df.to_csv(path+fout,index=False)
        df['LANGUAGE'] = k
        DF.append(df)
        k += 1

    # concatenate dataframes
    dnew = pd.concat(DF,axis=0)
    dnew = dnew.sample(frac=1).reset_index(drop=True) 
    dnew.to_csv(path+'dataset.csv',index=False)
