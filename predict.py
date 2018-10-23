from main import *
import torch
import sys

# load the model
clf = LangClassifier(len(vocab)+1,embedding_dim,hidden_size,3,n_layers)
checkpoint = torch.load(model_filename)
clf.load_state_dict(checkpoint['model_state_dict'])

# predict
word = sys.argv[1]
print('Given word:',word)
clf.predict(word)
