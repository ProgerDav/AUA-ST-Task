import torch
import numpy as np
import pandas as pd


DIALOGSUM_PATH = "hf://datasets/knkarthick/dialogsum/"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, text, labels, device):
        self.text = text
        self.labels = labels

    def __getitem__(self, index):
        return self.text[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class DataCollator:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        texts, labels = zip(*batch)
        tokenized_texts = self.tokenizer(
            list(texts), padding=True, truncation=True, return_tensors="pt"
        )
        labels = torch.tensor(labels).to(self.device)

        item = {key: value.to(self.device) for key, value in tokenized_texts.items()}
        item["labels"] = labels

        return item


def prepare_data(input_path: str, tokenizer, device, *, shuffle=False, max_length=None):
    """
    Prepare data for training by creating a PyTorch dataset.

    :param input_path: the CSV file, which contains the clusters of DialogSum
    :param tokenizer: the tokenizer to use
    :param shuffle: whether to shuffle the data
    :param max_length: the maximum length of the input sequences
    :return: the test and train dataset, the label dictionary
    """
    splits = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
    clusters = pd.read_csv(input_path)
    train_df = pd.read_csv(DIALOGSUM_PATH + splits["train"])
    validation_df = pd.read_csv(DIALOGSUM_PATH + splits["validation"])
    test_df = pd.read_csv(DIALOGSUM_PATH + splits["test"])
    # Ensure only the first summary is used for each conversation
    test_df = test_df[test_df["id"].str.endswith("_1")]

    # Join the clusters with the training data
    train_df = train_df.join(clusters.set_index("id"), on="id")
    validation_df = validation_df.join(clusters.set_index("id"), on="id")
    test_df = test_df.join(clusters.set_index("id"), on="id")

    # Separte the text and labels
    train_text = train_df["summary"].values.tolist()
    train_labels = train_df["cluster"].values.tolist()
    validation_text = validation_df["summary"].values.tolist()
    validation_labels = validation_df["cluster"].values.tolist()
    test_text = test_df["summary"].values.tolist()
    test_labels = test_df["cluster"].values.tolist()

    if shuffle:
        # shuffle train_text and train_labels in the same way
        state = np.random.get_state()
        np.random.shuffle(train_text)
        np.random.set_state(state)
        np.random.shuffle(train_labels)

    # tokenize text
    # train_encodings = tokenizer(train_text, truncation=True, padding=True, max_length=max_length)
    # validation_encodings = tokenizer(validation_text, truncation=True, padding=True, max_length=max_length)
    # test_encodings = tokenizer(test_text, truncation=True, padding=True, max_length=max_length)
    train_encodings = train_text
    validation_encodings = validation_text
    test_encodings = test_text

    # encode labels to ensure they are treated as categorical
    train_labels_encoded = [int(label) for label in train_labels]
    validation_labels_encoded = [int(label) for label in validation_labels]
    test_labels_encoded = [int(label) for label in test_labels]

    # create dataset
    train_data = Dataset(train_encodings, train_labels_encoded, device)
    validation_data = Dataset(validation_encodings, validation_labels_encoded, device)
    test_data = Dataset(test_encodings, test_labels_encoded, device)

    return (
        test_data,
        train_data,
        validation_data,
        [int(i) for i in sorted(clusters["cluster"].unique())],
    )
