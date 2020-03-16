import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class CNNContextClassifier(nn.Module):

    def __init__(self, hidden_dim,
                 filter_size, dropout_rate, bart, fix_embeddings=False):
        super(CNNContextClassifier, self).__init__()

        self.vocab_size = bart.model.encoder.embed_tokens.num_embeddings
        self.embedding_dim = bart.model.encoder.embed_tokens.embedding_dim
        self.hidden_dim = hidden_dim

        self.word_embeds = bart.extract_features #nn.Embedding(vocab_size, embedding_dim)
        self.use_cuda = next(bart.parameters()).is_cuda
        # if embed_mat is not None:
        #     self.word_embeds.weight.data = embed_mat
        #     if fix_embeddings:
        #         self.word_embeds.weight.requires_grad=False
        
        self.context_conv = nn.Conv1d(self.embedding_dim, self.embedding_dim, 
          filter_size, stride=1, padding=int((filter_size-1)/2), 
          groups=self.embedding_dim).half() # else groups=1

        self.ending_conv = nn.Conv1d(self.embedding_dim, self.embedding_dim, 
          filter_size, stride=1, padding=int((filter_size-1)/2), 
          groups=self.embedding_dim).half() # else groups=1

        self.fc = nn.Linear(self.embedding_dim, 1).half()
        self.drop = nn.Dropout(dropout_rate).half()


    # vec is seq_len x batch so transpose before iterating over
    def embed_seq(self, vec): # todo make sure this makes sense in BART world
        # currently bart doesn't let you batch extract features - trying to sort out
        all_embeddings = torch.stack([self.word_embeds(tok_list).squeeze(0)
                                      for tok_list in vec.transpose(0, 1)]) # results in batch x seq_len x embedsize
        vec_tr = all_embeddings.transpose(1, 2).contiguous()
        return vec_tr # dim [batch_size, embed_dim, length]

    # Input dimensions: 
    #   context: Tensor dim [seq_len, batch_size].
    #   endings: tuple of Tensors - 
    #            (dim [end_seq_len*, batch_size or num_endings] - endings, 
    #             dim [batch_size or num_endings] - batch lengths).
    #   Training: num_endings = 1; decoding: batch_size = 1.
    def forward(self, context, endings, itos=None):
        
        ends = endings[0]
        ends_ls = endings[1]
        cont_seq_len, batch_size = context.size()

        end_seq_len = ends.size()[0]
        end = ends.view(end_seq_len, -1)
        end_batch_size = end.size()[1]
        decode_mode = (batch_size == 1 and end_batch_size > 1)
        if not decode_mode:
            assert batch_size == end_batch_size, "Batch Size {} and End Batch Size {} do not match".format(batch_size, end_batch_size)

        maxpool = nn.MaxPool1d(cont_seq_len) # define layer for context length
        print(context.size(), end.size())
        embedding = self.embed_seq(context)
        context_convol = self.context_conv(embedding)
        context_pooled = maxpool(context_convol).view(batch_size, self.embedding_dim)
         
        maxpool_end = nn.MaxPool1d(end_seq_len)
        end_conv = F.relu(self.ending_conv(self.embed_seq(end)))
        end_pooled = maxpool_end(end_conv).view(end_batch_size, self.embedding_dim)

        if decode_mode:
            context_pooled = context_pooled.expand(end_batch_size, self.embedding_dim).contiguous()
        pooled = context_pooled * end_pooled

        dropped = self.drop(pooled)
        final = self.fc(dropped).view(-1)
        return final

