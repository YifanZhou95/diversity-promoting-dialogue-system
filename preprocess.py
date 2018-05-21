

### reference: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb ###

import unicodedata
import re

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# a class for computing statistics of corpus

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

class CorpusStat:
    def __init__(self, name):
        self.name = name
        # self.word2ind = {'PAD':0, 'SOS':1, 'EOS':2, 'UNK':3}
        # self.ind2word = {0:'PAD', 1:'SOS', 2:'EOS', 3:'UNK'}
        self.word2cnt = {}
        self.numOfWords = 4
        
        self.pairs = []
        self.pairsInd = []
        
    def filterSen(self, pair, max_sentence, min_sentence):
        q_len = len(pair[0].split())
        r_len = len(pair[1].split())
        if q_len>=min_sentence and q_len<=max_sentence and r_len>=min_sentence and r_len<=max_sentence:
            return True
        else:
            return False
    
    # statistics
    def count(self, pairs):
        self.word2cnt = {}
        self.numOfWords = 4
        for pair in pairs:
            for sentence in pair:
                for w in sentence.split():
                    if w in self.word2cnt:
                        self.word2cnt[w] += 1
                    else:
                        self.word2cnt[w] = 1
                        self.numOfWords += 1
    
    # customize a filter
    def filterVocab(self, max_vocab):
        items_sort = sorted(self.word2cnt.items(), key=lambda x:x[1], reverse=True)
        list_word = ['PAD','SOS','EOS','UNK'] + [w[0] for w in items_sort[:max_vocab-4]]
        list_ind = list(range(max_vocab))
        self.word2ind = dict(zip(list_word, list_ind))
        self.ind2word = dict(zip(list_ind, list_word))
    
    def countUNK(self, stat, pair):
        return sum([0 if word in stat.word2ind else 1 for word in pair[0].split()+pair[1].split()])
        

# transform sentences into code

def isInVocab(stat,token):
    return token if token in stat.word2ind else 'UNK'

def encodePair(stat, pair, reverse):
    tokens_q = pair[0].split()[::-1] if reverse else pair[0].split()
    tokens_r = pair[1].split()
    pairInd0 = [stat.word2ind[isInVocab(stat,token)] for token in tokens_q+['EOS']]
    pairInd1 = [stat.word2ind[isInVocab(stat,token)] for token in tokens_r+['EOS']]
    return (pairInd0, pairInd1)


def dataPreProcess(filename, max_vocab = 1000, max_sen = 18, min_sen = 3, unk_num = 2, reverse_flag = 1, inverse_flag=0):
    lines = f = open(filename, encoding = 'utf-8', mode = "r+").read().strip().split('\n')
    dialog = [normalize_string(line) for line in lines]
    
    stat = CorpusStat('train')
    if inverse_flag:
        stat.pairs = [(dialog[ind+1],dialog[ind]) for ind in range(0, len(dialog)-1, 2)]
    else:
        stat.pairs = [(dialog[ind],dialog[ind+1]) for ind in range(0, len(dialog)-1, 2)]
    stat.pairs = [pair for pair in stat.pairs if stat.filterSen(pair, max_sen, min_sen)]
    
    stat.count(stat.pairs)
    stat.filterVocab(max_vocab)
    stat.pairs = [pair for pair in stat.pairs if stat.countUNK(stat, pair)<=unk_num]
    stat.count(stat.pairs)

    for i in range(len(stat.pairs)):
        pair = stat.pairs[i]
        pairInd = encodePair(stat, pair, reverse_flag)
        stat.pairsInd.append(pairInd+(len(pairInd[0]),len(pairInd[1])))
    return stat