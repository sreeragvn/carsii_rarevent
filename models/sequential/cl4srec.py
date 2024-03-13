import math
import random
from models.base_model import BaseModel
from models.model_utils import LSTM_interactionEncoder, TransformerEncoder, TransformerEncoder_DynamicContext
from models.model_utils import TransformerLayer, TransformerEmbedding, LSTM_contextEncoder
import numpy as np
import torch
from torch import nn
from config.configurator import configs
import pickle


class CL4SRec(BaseModel):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.
    """
    def __init__(self, data_handler):
        super(CL4SRec, self).__init__(data_handler)

        # # Todo should we embed everything to same space or different space ? how do we select the embedding size ?
        # Extract configuration parameters
        data_config = configs['data']
        model_config = configs['model']
        train_config = configs['train']
        lstm_config = configs['lstm']
        duorec_config = configs['duorec']

        self.item_num = data_config['item_num']
        self.emb_size = model_config['embedding_size']
        self.max_len = model_config['max_seq_len']
        self.mask_token = self.item_num + 1
        self.n_layers = model_config['n_layers']
        self.n_heads = model_config['n_heads']
        self.inner_size = 4 * self.emb_size
        self.dropout_rate = model_config['dropout_rate']
        self.batch_size = train_config['batch_size']
        self.lmd = model_config['lmd']
        self.tau = model_config['tau']
        self.dynamic_context_feat_num = data_config['dynamic_context_feat_num']
        self.lstm_hidden_size = lstm_config['hidden_size']
        self.lstm_num_layers = lstm_config['num_layers']
        self.inner_size = duorec_config['inner_size']
        self.hidden_dropout_prob = duorec_config['hidden_dropout_prob']
        self.attn_dropout_prob = duorec_config['attn_dropout_prob']
        self.hidden_act = duorec_config['hidden_act']
        self.layer_norm_eps = duorec_config['layer_norm_eps']
        self.initializer_range = duorec_config['initializer_range']
        self.static_context_max_token = data_config['static_context_max']
        self.static_context_num = data_config['static_context_feat_num']

        # Static Embedding
        self.static_embedding = nn.ModuleList([nn.Embedding(num_embeddings=max_val + 1, 
                                                            embedding_dim=self.emb_size) 
                                                            for max_val, _ in zip(self.static_context_max_token, 
                                                                                  range(self.static_context_num))])
        self.fc_input_size = len(self.static_embedding) * self.emb_size
        self.fc_static_dim_red = nn.Linear(self.fc_input_size, self.lstm_hidden_size)

        # interaction Encoder( # interaction_encoder options are lstm, sasrec, durorec)
        if model_config['interaction_encoder'] == 'lstm':
            self.emb_layer = nn.Embedding(self.item_num + 2, self.emb_size)
            self.interaction_encoder = LSTM_interactionEncoder(self.item_num + 2, 
                                                         self.emb_size, 
                                                         self.lstm_hidden_size, 
                                                         self.lstm_num_layers)
        elif model_config['interaction_encoder'] == 'sasrec':
            self.emb_layer = TransformerEmbedding(self.item_num + 2, 
                                                  self.emb_size, self.max_len)
            self.transformer_layers = nn.ModuleList([TransformerLayer(self.emb_size, 
                                                                      self.n_heads, 
                                                                      self.inner_size, 
                                                                      self.dropout_rate) 
                                                                      for _ in range(self.n_layers)])
        # implementation of sasrec from another source - DUORec https://github.com/RuihongQiu/DuoRec/tree/master
        elif model_config['interaction_encoder'] == 'duorec':
            self.emb_layer = nn.Embedding(self.item_num + 2, self.emb_size, padding_idx=0)
            self.position_embedding = nn.Embedding(self.max_len, self.emb_size)
            self.transformer_layers = TransformerEncoder(n_layers=self.n_layers,
                                                         n_heads=self.n_heads,
                                                         hidden_size=self.emb_size,
                                                         inner_size=self.inner_size,
                                                         hidden_dropout_prob=self.hidden_dropout_prob,
                                                         attn_dropout_prob=self.attn_dropout_prob,
                                                         hidden_act=self.hidden_act,
                                                         layer_norm_eps=self.layer_norm_eps,
                                                         eps=self.initializer_range)
            self.LayerNorm = nn.LayerNorm(self.emb_size, eps=self.initializer_range)
            self.dropout = nn.Dropout(self.attn_dropout_prob)
        else:
            print('mention the interaction encoder - sasrec, lstm or duorec')

        # dynamic Context Encoder
        if model_config['context_encoder'] == 'lstm':
            self.context_encoder = LSTM_contextEncoder(self.dynamic_context_feat_num, 
                                                       self.lstm_hidden_size, 
                                                       self.lstm_num_layers)
            input_size = 3 * self.emb_size
        elif model_config['context_encoder'] == 'transformer':
            self.context_encoder = TransformerEncoder_DynamicContext(self.dynamic_context_feat_num, # num_features_continuous
                                                                     data_config['dynamic_context_window_length'],
                                                                     hidden_dim=512, # d_model
                                                                     num_heads=8,)
            input_size = 6400 + 2 * self.emb_size

        output_size = 128 

        fc_layers = []
        for i in range(10):
            layer_output_size = int(output_size - (output_size - 1) * (i / 9))
            fc_layers.append(nn.Linear(input_size, layer_output_size))
            fc_layers.append(nn.ReLU())
            input_size = layer_output_size
        fc_layers.append(nn.Linear(layer_output_size, 1))
        # fc_layers.append(nn.Sigmoid())
        self.fc_layers = nn.Sequential(*fc_layers)

        # Loss Function
        with open(configs['train']['parameter_class_weights_path'], 'rb') as f:
            _class_w = pickle.load(f)
        print(_class_w)

        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=_class_w)

        self.mask_default = self.mask_correlated_samples(
            batch_size=self.batch_size)

        # parameters initialization
        self.apply(self._init_weights)

    def count_parameters(self):
        # Count the total number of parameters in the model
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, (nn.Linear)):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, batch_seqs,batch_context, batch_static_context, sequence_length):
        if configs['model']['interaction_encoder'] == 'lstm':
            item_embedded = self.emb_layer(batch_seqs)
            sasrec_out = self.interaction_encoder(item_embedded) ## not sasrec. just lstm
        elif  configs['model']['interaction_encoder'] == 'duorec':
            position_ids = torch.arange(batch_seqs.size(1), dtype=torch.long, device=batch_seqs.device)
            position_ids = position_ids.unsqueeze(0).expand_as(batch_seqs)
            position_embedding = self.position_embedding(position_ids)

            item_emb = self.emb_layer(batch_seqs)
            input_emb = item_emb + position_embedding
            input_emb = self.LayerNorm(input_emb)
            input_emb = self.dropout(input_emb)

            extended_attention_mask = self.get_attention_mask(batch_seqs)

            trm_output = self.transformer_layers(input_emb, extended_attention_mask, output_all_encoded_layers=True)
            output = trm_output[-1]
            sasrec_out = self.gather_indexes(output, sequence_length - 1)
        elif configs['model']['interaction_encoder'] == 'sasrec':
            mask = (batch_seqs > 0).unsqueeze(1).repeat(
                1, batch_seqs.size(1), 1).unsqueeze(1)
            # Embedding Layer:
            # Passes the input sequence batch_seqs through an embedding layer (self.emb_layer). This layer converts integer indices into dense vectors.
            x = self.emb_layer(batch_seqs)

            # Transformer Layers:
            # Iterates through a series of transformer layers (self.transformer_layers) and applies each one to the input tensor x. The transformer layers are expected to take the input tensor and the mask as arguments.
            for transformer in self.transformer_layers:
                x = transformer(x, mask)
            # Extracts the output from the last position of the sequence (-1). This is a common practice in transformer-based models, where the output corresponding to the last position is often used as a representation of the entire sequence.
            sasrec_out = x[:, -1, :]
        batch_context = batch_context.to(sasrec_out.dtype)
        batch_context = batch_context.transpose(1, 2)
        context_output = self.context_encoder(batch_context)
        static_context = []
        for i, embedding_layer in enumerate(self.static_embedding):
            static_context.append(embedding_layer(batch_static_context[:, i]))
        static_context = torch.cat(static_context, dim=1)
        static_context = self.fc_static_dim_red(static_context)
        out = torch.cat((sasrec_out, context_output, static_context), dim=1)
        output = self.fc_layers(out)
        return output

    def cal_loss(self, batch_data):
        _, batch_seqs, _, _, batch_dynamic_context, batch_static_context, sequence_length, common_item = batch_data
        seq_output = self.forward(batch_seqs, batch_dynamic_context, batch_static_context, sequence_length)
        input = seq_output.float().squeeze()
        target = common_item.float()
        loss = self.loss_func(input, target)
        cl_loss = 0
        loss_dict = {
            'rec_loss': loss.item(),
            'cl_loss': cl_loss,
        }
        return loss + cl_loss, loss_dict
    
    def full_predict(self, batch_data):
        _, batch_seqs, _, _, batch_dynamic_context, batch_static_context, sequence_length, _ = batch_data
        seq_output = self.forward(batch_seqs, batch_dynamic_context, batch_static_context, sequence_length)
        return seq_output
