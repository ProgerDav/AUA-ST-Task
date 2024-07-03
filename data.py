import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.text.items()}
        item['labels'] = self.labels[index]
        return item

    def __len__(self):
        return len(self.labels)


def prepare_data(dataset, tokenizer, *, shuffle=False, max_length=None):
    """
    Prepare data for training by creating a PyTorch dataset.

    :param dataset: the dataset dictionary
    :param tokenizer: the tokenizer to use
    :param shuffle: whether to shuffle the data
    :param max_length: the maximum length of the input sequences
    :return: the test and train dataset, the label dictionary
    """
    train_text, train_labels = dataset['train']

    if shuffle:
        # shuffle train_text and train_labels in the same way
        state = np.random.get_state()
        np.random.shuffle(train_text)
        np.random.set_state(state)
        np.random.shuffle(train_labels)

    # tokenize text
    train_encodings = tokenizer(train_text, truncation=True, padding=True, max_length=max_length)
    test_text, test_labels = dataset['test']
    test_encodings = tokenizer(test_text, truncation=True, padding=True, max_length=max_length)
    # encode labels
    label_dict = dataset['label_dict']
    train_labels_encoded = [label_dict[label] for label in train_labels]
    test_labels_encoded = [label_dict[label] for label in test_labels]
    # create dataset
    train_data = Dataset(train_encodings, train_labels_encoded)
    test_data = Dataset(test_encodings, test_labels_encoded)

    return test_data, train_data, dataset['label_dict']