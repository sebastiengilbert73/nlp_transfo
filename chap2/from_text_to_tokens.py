# Cf. p. 29
import logging
import pandas as pd
from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("from_text_to_tokens.main()")

    text = "Tokenizing text is a core task of NLP."
    tokenized_text = list(text)
    logging.info(f"tokenized_text = {tokenized_text}")
    token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
    logging.info(f"token2idx = {token2idx}")
    input_ids = [token2idx[token] for token in tokenized_text]
    logging.info(f"input_ids = {input_ids}")

    categorical_df  = pd.DataFrame(
        {"Name": ["Bumblebee", "Optimus Prime", "Megatron"], "Label ID": [0, 1, 2]}
    )
    logging.info(f"categorical_df = \n{categorical_df}")
    one_hot_encoding = pd.get_dummies(categorical_df["Name"])
    logging.info(f"one_hot_encoding =\n{one_hot_encoding}")

    input_ids = torch.tensor(input_ids)
    one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
    logging.info(f"one_hot_encodings.shape = {one_hot_encodings.shape}")

    # Word tokenization
    tokenized_text = text.split()
    logging.info(f"tokenized_text = {tokenized_text}")

    # Subword tokenization
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    encoded_text = tokenizer(text)
    logging.info(f"encoded_text = {encoded_text}")
    tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
    logging.info(f"tokens = {tokens}")
    logging.info(f"Coversion to string: {tokenizer.convert_tokens_to_string(tokens)}")

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    emotions = load_dataset("emotion")
    logging.info(f"Tokenized texts:\n{tokenize(emotions['train'][:2])}")
    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
    logging.info(f"emotions_encoded['train'].column_names = {emotions_encoded['train'].column_names}")

if __name__ == '__main__':
    main()