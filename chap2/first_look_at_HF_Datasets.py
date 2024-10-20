# Cf. p. 23
import logging
from datasets import load_dataset
from huggingface_hub import list_datasets

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

if __name__ == '__main__':
    main()