# Cf. p. 63
import logging
from transformers import AutoTokenizer, AutoConfig
import torch
from math import sqrt
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("scaled_dot_product_attention.main()")

    model_ckpt = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    text = "time flies like an arrow"

    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    logging.info(f"inputs.input_ids:\n{inputs.input_ids}")

    config = AutoConfig.from_pretrained(model_ckpt)
    token_emb = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    logging.info(f"token_emb:\n{token_emb}")

    inputs_embeds = token_emb(inputs.input_ids)
    logging.info(f"inputs_embeds.size(): {inputs_embeds.size()}")

    query = key = value = inputs_embeds
    dim_k = key.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2) / sqrt(dim_k))  # bmm = batch matrix-matrix product
    logging.info(f"scores.size() = {scores.size()}")

    weights = torch.nn.functional.softmax(scores, dim=-1)
    logging.info(f"weights.sum(dim=-1) = {weights.sum(dim=-1)}")

    attn_outputs = torch.bmm(weights, value)
    logging.info(f"attn_outputs.shape = {attn_outputs.shape}")

def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(dim_k)
    weights = torch.nn.functional.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

if __name__ == '__main__':
    main()