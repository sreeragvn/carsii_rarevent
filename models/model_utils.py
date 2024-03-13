import math
import torch as t
from torch import nn
from torch.nn import init
# import dgl.function as fn
from config.configurator import configs
import torch.nn.functional as F
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_size, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.d_k = hidden_size // num_heads
        self.n_h = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.output_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout)

    def _cal_attention(self, query, key, value, mask=None, dropout=None):
        scores = t.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return t.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.n_h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self._cal_attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_h * self.d_k)

        return self.output_linear(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, d_ff)
        self.w_2 = nn.Linear(d_ff, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class ResidualConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, feed_forward_size, dropout_rate):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, hidden_size=hidden_size, dropout=dropout_rate)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, d_ff=feed_forward_size, dropout=dropout_rate)
        self.input_sublayer = ResidualConnection(hidden_size=hidden_size, dropout=dropout_rate)
        self.output_sublayer = ResidualConnection(hidden_size=hidden_size, dropout=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    def __init__(self, item_num, emb_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token_emb = nn.Embedding(item_num, emb_size, padding_idx=0)
        self.position_emb = nn.Embedding(max_len, emb_size)
        self.dropout = nn.Dropout(p=dropout)
        self.emb_size = emb_size

    def forward(self, batch_seqs):
        batch_size = batch_seqs.size(0)
        pos_emb = self.position_emb.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.token_emb(batch_seqs) + pos_emb
        return self.dropout(x)
    
class LSTM_contextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM_contextEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]

class LSTM_interactionEncoder(nn.Module):
    def __init__(self, num_items, embedding_dim, lstm_hidden_dim, num_layers):
        super(LSTM_interactionEncoder, self).__init__()        
        
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(lstm_hidden_dim, num_items)
    def forward(self, item_embedded):
        lstm_output, _ = self.lstm(item_embedded)
        last_hidden_state = lstm_output[:, -1, :]
        # _, (last_hidden_state, _) = self.lstm(lstm_output)
        output = self.output_layer(last_hidden_state)
        return output

class TransformerEncoder_DynamicContext(nn.Module):
    def __init__(
            self, 
            input_size_cont, # num_features_continuous
            seq_len,
            hidden_dim=512, # d_model
            num_heads=8,
    ):
        super(TransformerEncoder_DynamicContext, self).__init__()

        feed_forward_size = 1024
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim 

        # Use linear layer instead of embedding 
        self.input_embedding = nn.Linear(input_size_cont, hidden_dim)
        self.pos_enc = self.positional_encoding()

        # Multi-Head Attention
        self.multihead = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

        # position-wise Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, feed_forward_size),
            nn.ReLU(),
            nn.Linear(feed_forward_size, hidden_dim)
        )
        self.fc_out1 = nn.Linear(hidden_dim, 64)

    def positional_encoding(self):
        pe = t.zeros(self.seq_len, self.hidden_dim) # positional encoding 
        pos = t.arange(0, self.seq_len, dtype=t.float32).unsqueeze(1)
        _2i = t.arange(0, self.hidden_dim, step=2).float()
        pe[:, 0::2] = t.sin(pos / (10000 ** (_2i / self.hidden_dim)))
        pe[:, 1::2] = t.cos(pos / (10000 ** (_2i / self.hidden_dim)))
        return pe
        
    def forward(self, x_cont):
        
        # Embedding + Positional
        x = self.input_embedding(x_cont)
        self.pos_enc = self.pos_enc.to(x.device)
        x += self.pos_enc

        # Multi-Head Attention
        x_, _ = self.multihead(x,x,x)
        x_ = self.dropout_1(x_)

        # Add and Norm 1
        x = self.layer_norm_1(x_ + x)

        # Feed Forward
        x_ = self.feed_forward(x)
        x_ = self.dropout_2(x_)

        # Add and Norm 2
        x = self.layer_norm_2(x_ + x)
        # Output (customized flatten)
        x = self.fc_out1(x)
        # shape: N, num_features, 64
        x = t.flatten(x, start_dim=1)
        self.output_dim = x.size(1)
        # x = F.adaptive_avg_pool1d(x.transpose(1, 2), 64).transpose(1, 2)
        return x

## implementation of sasrec from another source - DUORec https://github.com/RuihongQiu/DuoRec/tree/master
class MultiHeadAttention_D(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttention_D, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = t.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = t.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": t.tanh,
            "sigmoid": t.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + t.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * t.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer_D(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(TransformerLayer_D, self).__init__()
        self.multi_head_attention = MultiHeadAttention_D(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer_D(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
