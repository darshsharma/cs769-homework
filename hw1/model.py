import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    embeddings_dict = {}
    with open(emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_dict[word] = vector

    vocab_size = len(vocab.word2id)
    emb_matrix = np.random.normal(scale=0.6, size=(vocab_size, emb_size)).astype("float32")
    
    for word, idx in vocab.word2id.items():
        if word in embeddings_dict:
            emb_matrix[idx] = embeddings_dict[word]
    return emb_matrix
     


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()
        

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        self.embedding = nn.Embedding(len(self.vocab), self.args.emb_size, padding_idx=self.vocab.pad_id)
        layers = []
        input_size = self.args.emb_size
        output_size = self.args.hid_size
        for _ in range(self.args.hid_layer):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
            input_size = output_size
        layers.append(nn.Linear(input_size, self.tag_size))
        self.fc = nn.Sequential(*layers)
        self.dropout_fc = nn.Dropout(p=self.args.emb_drop)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        for name, params in self.named_parameters():
            if params.requires_grad:
                nn.init.uniform_(params, -0.08, 0.08)
        with torch.no_grad():
            self.embedding.weight[self.vocab.pad_id].zero_()

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        emb_matrix = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        vocab_size, emb_size = emb_matrix.shape
        print(f'Load pre-trained embeddings from {self.args.emb_file}, size={vocab_size}x{emb_size}')
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_size,
            padding_idx=self.vocab.pad_id  # ensures <pad> stays zeroed out
       )
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
        if self.vocab.pad_id is not None:
            self.embedding.weight.data[self.vocab.pad_id] = 0.0

        self.embedding.weight.requires_grad = True  # fine-tune embeddings during training
    
    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        batch_size, seq_len = x.size()
        token_mask = (x != self.vocab.pad_id).float()

        if self.training:
            p = self.args.word_drop
            rand_vals = torch.rand(batch_size, seq_len)
            mask = (rand_vals > p).float()  # use .float() if you want float 0/1
            token_mask = token_mask * mask
        # Create binary mask    
        emb = self.embedding(x)
        emb_masked = emb * token_mask.unsqueeze(-1)
        summed = emb_masked.sum(dim=1)
        count = token_mask.sum(dim=1,keepdim=True).clamp(min=1.0)
        avg = summed / count
        avg = self.dropout_fc(avg) 
        x = self.fc(avg)
        return x

        


