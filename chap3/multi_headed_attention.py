# Cf. p. 67
import logging
from transformers import AutoTokenizer, AutoConfig
import torch
import math
#from bertviz import head_view
from transformers import AutoModel

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

class AttentionHead(torch.nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = torch.nn.Linear(embed_dim, head_dim)
        self.k = torch.nn.Linear(embed_dim, head_dim)
        self.v = torch.nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):  # hidden_state.shape = (B, N_tokens, embed_dim)
        attn_outputs = scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs  # (B, N_tokens, head_dim)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads  # Ex.: 768 // 12 = 64
        self.heads = torch.nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):  # hidden_state.shape = (B, N_tokens, embed_dim)
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)  # (B, N_tokens, embed_dim = head_dim * num_heads)
        x = self.output_linear(x)  # (B, N_tokens, embed_dim)
        return x

class FeedForward(torch.nn.Module):  # position-wise feed-forward layer
    def __init__(self, config):
        super().__init__()
        self.linear_1 = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = torch.nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):  # x.shape = (B, N_tokens, embed_size)
        x = self.linear_1(x)  # (B, N_tokens, intermediate_size)
        x = self.gelu(x)  # (B, N_tokens, intermediate_size)
        x = self.linear_2(x)  # (B, N_tokens, embed_size)
        x = self.dropout(x)  # (B, N_tokens, embed_size)
        return x  # (B, N_tokens, embed_size)

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = torch.nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = torch.nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):  # x.shape = (B, N_tokens, embed_size)
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)  # (B, N_tokens, embed_size)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state)  # (B, N_tokens, embed_size)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))  # (B, N_tokens, embed_size)
        return x  # (B, N_tokens, embed_size)

class Embeddings(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout()

    def forward(self, input_ids):  # input_ids.shape = (B, N_tokens)
        # Create position IDs for input sequence
        seg_length = input_ids.size(1)  # N_tokens
        position_ids = torch.arange(seg_length, dtype=torch.long).unsqueeze(0)  # (1, N_tokens)
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)  # (B, N_tokens, embed_size)
        position_embeddings = self.position_embeddings(position_ids)  # (B, N_tokens, embed_size)
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings  # (B, N_tokens, embed_size)
        embeddings = self.layer_norm(embeddings)  # (B, N_tokens, embed_size)
        embeddings = self.dropout(embeddings)  # (B, N_tokens, embed_size)
        return embeddings  # (B, N_tokens, embed_size)

class TransformerEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = torch.nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x):  # x.shape = (B, N_tokens)
        x = self.embeddings(x)  # (B, N_tokens, embed_size)
        for layer in self.layers:
            x = layer(x)  # (B, N_tokens, embed_size)
        return x  # (B, N_tokens, embed_size)

class TransformerForSequenceClassification(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):  # x.shape = (B, N_tokens)
        x = self.encoder(x)[:, 0, :]  # Select hidden state of [CLS] token  (B, embed_size)
        x = self.dropout(x)  # (B, embed_size)
        x = self.classifier(x)  # (B, num_labels)
        return x  # (B, num_labels)

def main():
    logging.info("multi_headed_attention.main()")

    model_ckpt = "bert-base-uncased"
    config = AutoConfig.from_pretrained(model_ckpt)  # C:\Users\sebas\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    multihead_attn = MultiHeadAttention(config)
    #logging.info(f"config.transformers_version = {config.transformers_version}")  # 4.6.0.dev0

    text = "time flies like an arrow"
    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    token_emb = torch.nn.Embedding(config.vocab_size, config.hidden_size)

    inputs_embeds = token_emb(inputs.input_ids)
    attn_outputs = multihead_attn(inputs_embeds)
    logging.info(f"attn_outputs.size() = {attn_outputs.size()}")

    """model = AutoModel.from_pretrained(model_ckpt, output_attentions=True)

    sentence_a = "time flies like an arrow"
    sentence_b = "fruit flies like a banana"

    viz_inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt')
    attention = model(**viz_inputs).attentions
    sentence_b_start = (viz_inputs.token_type_ids == 0).sum(dim=1)
    tokens = tokenizer.convert_ids_to_tokens(viz_inputs.input_ids[0])

    head_view(attention, tokens, sentence_b_start, heads=[8])  # Works only in jupyter notebooks
    """

    encoder = TransformerEncoder(config)
    encoded_tsr = encoder(inputs.input_ids)
    logging.info(f"encoded_tsr.shape = {encoded_tsr.shape}")

    config.num_labels = 3
    classifier = TransformerForSequenceClassification(config)
    logits = classifier(inputs.input_ids)
    logging.info(f"logits.shape = {logits.shape}")


def scaled_dot_product_attention(query, key, value, mask=None):  # (B, N_tokens, head_dim)
    dim_k = query.size(-1)  # = head_dim
    scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(dim_k)  # (B, N_tokens, N_tokens)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.nn.functional.softmax(scores, dim=-1)  # (B, N_tokens, N_tokens)
    return torch.bmm(weights, value)  # (B, N_tokens, head_dim)

if __name__ == '__main__':
    main()