# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
import torch
from process_data import *
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import LogSoftmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# vocabulary dictionary
if os.path.exists('vocabulary.dat'):
    with open("vocabulary.dat", "rb") as f:
        vocab = pickle.load(f)
else:
    vocab = {w:ind+1 for ind,w in enumerate(all_letters)}
    with open("vocabulary.dat", "wb") as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

def word2seq(words):
    """converts the given list of words to their vocab indices"""
    seq = []
    seq_len = []
    for word in words:
        seq.append([vocab[ch] for ch in word])
        seq_len.append(len(seq[-1]))
    return (seq,seq_len)

def pad_sequences(df):
    languages = torch.tensor(df['LANGUAGE'],dtype=torch.long)
    words = list(df['WORDS'])

    sequence, sequence_length = word2seq(words)
    seq_tensor = torch.zeros((len(sequence),max(sequence_length))).long()

    for idx, (seq, seq_len) in enumerate(zip(sequence,sequence_length)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    sequence_length = torch.LongTensor(sequence_length)


    # Return variables
    # DataParallel requires everything to be a Variable

    return create_variable(seq_tensor), \
        create_variable(sequence_length), \
        create_variable(languages)

def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

class WordDataset(Dataset):
    """ word dataset."""
    def __init__(self, X_data, y_data, X_len):
        self.words = X_data
        self.language = y_data
        self.words_len = X_len
        self.len = len(self.words)

    def __getitem__(self, index):
        return self.words[index,:], self.language[index], self.words_len[index]

    def __len__(self):
        return self.len

# create the neural network
class LangClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, n_layers=1):
        super(LangClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_directions = 1
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim,hidden_size,n_layers,batch_first=False)
        self.fc = nn.Linear(hidden_size,output_size)
        self.lsmax = LogSoftmax(dim=1) 

    def forward(self, input, sequence_length):
        input = input.t()
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)

        yemb = self.embedding(input)
        gru_input = pack_padded_sequence(yemb,sequence_length.data.cpu().numpy())
        self.gru.flatten_parameters()
        output, hidden = self.gru(gru_input, hidden)

        fc_output = self.fc(hidden[-1])
        yhat = self.lsmax(fc_output)

        return yhat

    def predict(self,word):
        """predicts for a single input word"""
        seq,seq_len = word2seq([word])
        seq = torch.LongTensor(seq)
        seq_len = torch.LongTensor(seq_len)
        
        yhat = self.forward(seq,seq_len).detach().numpy()[0]

        langs = ['English','French ','Spanish']
        idx = yhat.argmax()

        print('Prediction: *'+langs[idx]+'*\n')
        for k,la in enumerate(langs):
            print(la+' :'+str(np.exp(yhat[k])))


    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return create_variable(hidden)

if __name__ == "__main__":
    for xi,batch in enumerate(train_loader):
        break

    input = batch[0]
    languages = batch[1]
    sequence_length = batch[2]

    # Sort tensors by their length
    sequence_length, perm_idx = sequence_length.sort(0, descending=True)
    input = input[perm_idx]
    languages = languages[perm_idx]

    #  def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, n_layers=1):
    clf = LangClassifier(len(vocab)+1,20,10,3)
    ypred = clf.forward(input,sequence_length)
    
    criterion = torch.nn.NLLLoss()
    loss = criterion(ypred,languages)


