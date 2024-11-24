# Cf. p. 38
import logging
import pandas as pd
from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import AutoModel
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("transformers_as_feature_extractors.py")

    model_ckpt = "distilbert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_ckpt).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    text = "this is a test"
    inputs = tokenizer(text, return_tensors="pt")
    logging.info(f"Input tensor shape: {inputs['input_ids'].size()}")

    # Move the encoding tensor on device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logging.info(f"outputs: \n{outputs}")

    logging.info(f"outputs.last_hidden_state.size(): {outputs.last_hidden_state.size()}")

    def extract_hidden_states(batch):
        # Place model inputs on the GPU
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        # Extract last hidden states
        with torch.no_grad():
            last_hidden_state = model(**inputs).last_hidden_state
        # Return vector for [CLS] token
        return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

    emotions = load_dataset("emotion")
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)
    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
    emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
    logging.info(f"emotions_hidden['train'].column_names = {emotions_hidden['train'].column_names}")

    # Creating a feature matrix
    X_train = np.array(emotions_hidden["train"]["hidden_state"])
    X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
    y_train = np.array(emotions_hidden["train"]["label"])
    y_valid = np.array(emotions_hidden["validation"]["label"])
    logging.info(f"X_train.shape = {X_train.shape}; X_valid.shape = {X_valid.shape}")

    # Visualizing the training set, Cf. p. 42
    from umap import UMAP
    from sklearn.preprocessing import MinMaxScaler

    # Scale features to [0, 1] range
    X_scaled = MinMaxScaler().fit_transform(X_train)
    # Initialize and fit UMAP
    mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
    # Create a DataFrame of 20 embeddings
    df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
    df_emb["label"] = y_train
    df_emb.head()

    fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    axes = axes.flatten()
    cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
    labels = emotions["train"].features["label"].names

    for i, (label, cmap) in enumerate(zip(labels, cmaps)):
        df_emb_sub = df_emb.query(f"label == {i}")
        axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,))
        axes[i].set_title(label)
        axes[i].set_xticks([]), axes[i].set_yticks([])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()