# Cf. p. 43
import logging
import torch
from transformers import AutoTokenizer
from transformers import AutoModel
from datasets import load_dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info(f"train_simple_classifier.main()")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ckpt = "distilbert-base-uncased"
    model = AutoModel.from_pretrained(model_ckpt).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def extract_hidden_states(batch):
        # Place model inputs on the GPU
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        # Extract last hidden states
        with torch.no_grad():
            last_hidden_state = model(**inputs).last_hidden_state
        # Return vector for [CLS] token
        return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

    emotions = load_dataset("emotion")
    labels = emotions['train'].features['label'].names

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
    emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
    # Creating a feature matrix
    X_train = np.array(emotions_hidden["train"]["hidden_state"])
    X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
    y_train = np.array(emotions_hidden["train"]["label"])
    y_valid = np.array(emotions_hidden["validation"]["label"])

    # We increase 'max_iter' to guarantee convergence
    lr_clf = LogisticRegression(max_iter=3000)
    lr_clf.fit(X_train, y_train)
    lr_clf_score = lr_clf.score(X_valid, y_valid)
    logging.info(f"lr_clf_score = {lr_clf_score}")

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    dummy_clf_score = dummy_clf.score(X_valid, y_valid)
    logging.info(f"dummy_clf_score = {dummy_clf_score}")

    def plot_confusion_matrix(y_preds, y_true, labels):
        cm = confusion_matrix(y_true, y_preds, normalize="true")
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
        plt.title("Normalized confusion matrix")
        plt.show()

    y_preds = lr_clf.predict(X_valid)
    plot_confusion_matrix(y_preds, y_valid, labels)


if __name__ == '__main__':
    main()