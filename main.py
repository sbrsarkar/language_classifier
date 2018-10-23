from utils import *
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split 

# parameters
batch_size = 1500 
lr = 5e-3
split_ratio = 0.95
embedding_dim = 50
hidden_size = 50
n_layers = 2
max_epochs = 50
model_filename = 'model.pth'

if __name__=='__main__':
    # load data
    df = pd.read_csv('data/dataset.csv')
    X_data, X_len, y_data = pad_sequences(df)

    n_train = int(split_ratio*X_data.size()[0])
    trainset = WordDataset(X_data[:n_train,:],y_data[:n_train],X_len[:n_train])
    testset = WordDataset(X_data[n_train:,:],y_data[n_train:],X_len[n_train:])

    train_loader = DataLoader(trainset,batch_size=batch_size,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(testset,shuffle=True, num_workers=4)

    # create the network
    # vocab_size, embedding_dim, hidden_size, output_size, n_layers=1):
    clf = LangClassifier(len(vocab)+1,embedding_dim,hidden_size,3,n_layers)

    # train the network
    #  criterion = torch.nn.MSELoss(reduction='elementwise_mean')
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    print('training..')
    loss_val = []
    for epoch in range(max_epochs):
        for idx,batch in enumerate(train_loader):
            input = batch[0]
            languages = batch[1]
            sequence_length = batch[2]

            # Sort tensors by their length
            sequence_length, perm_idx = sequence_length.sort(0, descending=True)
            input = input[perm_idx]
            languages = languages[perm_idx]

            ypred = clf.forward(input,sequence_length)
        
            loss = criterion(ypred,languages)
            loss_val.append(loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  

        if np.mod(epoch,1)==0:
            print('epoch={:3d}, loss={}'.format(epoch+1,loss.item()))

    # some manual tests
    print('model saved to file:',model_filename)
    torch.save({'epoch':epoch,
                'model_state_dict':clf.state_dict(),
                'loss':loss,
                'vocab': vocab
                },model_filename)
