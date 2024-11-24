# Cf. p. 23
import logging
from datasets import load_dataset
from huggingface_hub import list_datasets
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("first_look_at_HF_Datasets.main()")

    """all_datasets = list(list_datasets())  # Takes a long time
    logging.info(f"There are {len(all_datasets)} datasets currently available on the Hub")
    logging.info(f"The first 10 are: {all_datasets[:10]}")
    """

    emotions = load_dataset("emotion")
    logging.debug(emotions)

    train_ds = emotions['train']
    logging.debug(f"train_ds.column_names = {train_ds.column_names}")  # ['text', 'label']
    logging.debug(f"train_ds.features = {train_ds.features}")  # {'text': Value(dtype='string', id=None), 'label': ClassLabel(names=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'], id=None)}

    def label_int2str(row):
        return emotions["train"].features["label"].int2str(row)

    emotions.set_format(type="pandas")
    df = emotions['train'][:]
    df["label_name"] = df["label"].apply(label_int2str)

    print(df.head())

    df["label_name"].value_counts(ascending=True).plot.barh()
    plt.title("Frequency of Classes")

    # How long are our tweets?
    df["Words per tweet"] = df["text"].str.split().apply(len)
    df.boxplot("Words per tweet", by="label_name", grid=False, showfliers=False, color='black')
    plt.suptitle("")
    plt.xlabel("")

    plt.show()

    # Reset the output format
    emotions.reset_format()



if __name__ == '__main__':
    main()