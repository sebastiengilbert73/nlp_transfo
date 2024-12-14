# Cf. p. 45
import logging
from transformers import AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from transformers import pipeline
import pandas as pd
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("fine_tuning_transformers.main()")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ckpt = "distilbert-base-uncased"
    num_labels = 6
    model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).
             to(device))
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    emotions = load_dataset("emotion")
    labels = emotions['train'].features['label'].names

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
    emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    batch_size = 64
    logging_steps = len(emotions_encoded["train"])  // batch_size
    model_name = f"{model_ckpt}-finetuned-emotion"
    training_args = TrainingArguments(
        output_dir=model_name,
        num_train_epochs=2,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        eval_strategy="epoch",  # 'evaluation_strategy' will be deprecated
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=False,
        log_level="error"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=emotions_encoded["train"],
        eval_dataset=emotions_encoded["validation"],
        tokenizer=tokenizer
    )
    trainer.train()

    preds_output = trainer.predict(emotions_encoded["validation"])
    logging.info(f"preds_output.metrics = {preds_output.metrics}")

    y_preds = np.argmax(preds_output.predictions, axis=1)
    y_valid = np.array(emotions_encoded["validation"]["label"])
    plot_confusion_matrix(y_preds, y_valid, labels)

    def forward_pass_with_label(batch):
        # Place all input tensors on the same device as the model
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        with torch.no_grad():
            output = model(**inputs)
            pred_label = torch.argmax(output.logits, axis=-1)
            loss = torch.nn.functional.cross_entropy(output.logits, batch["label"].to(device), reduction="none")
        # Place outputs on CPU for compatibility with other dataset columns
        return {"loss": loss.cpu().numpy(), "predicted_label": pred_label.cpu().numpy()}

    # Convert our dataset back to PyTorch tensors
    emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Compute loss values
    emotions_encoded["validation"] = emotions_encoded["validation"].map(
        forward_pass_with_label, batched=True, batch_size=16
    )

    def label_int2str(row):
        return emotions["train"].features["label"].int2str(row)

    # Create a DataFrame with texts, losses, and predicted/true labels
    emotions_encoded.set_format("pandas")
    cols = ["text", "label", "predicted_label", "loss"]
    df_test = emotions_encoded["validation"][:][cols]
    df_test["label"] = df_test["label"].apply(label_int2str)
    df_test["predicted_label"] = (df_test["predicted_label"].apply(label_int2str))

    highest_loss_samples = df_test.sort_values("loss", ascending=False).head(10)
    logging.info(f"highest_loss_samples:\n{highest_loss_samples.to_string(index=False)}")

    lowest_loss_samples = df_test.sort_values("loss", ascending=True).head(10)
    logging.info(f"lowest_loss_samples:\n{lowest_loss_samples.to_string(index=False)}")

    # Load the model from file
    classifier = pipeline("text-classification", os.path.join(training_args.output_dir, "checkpoint-500"))
    custom_tweet = "I saw a movie today and it was really good."
    preds = classifier(custom_tweet, top_k=None)
    preds_df = pd.DataFrame(preds)
    plt.bar(preds_df["label"], 100 * preds_df["score"], color='C0')
    plt.title(f'"{custom_tweet}"')
    plt.ylabel("Class probability (%)")
    plt.show()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()



if __name__ == '__main__':
    main()