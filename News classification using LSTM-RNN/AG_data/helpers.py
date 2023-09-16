import random
import numpy as np
import pandas as pd
import spacy
from torchtext import data
import torch
import os
import multiprocessing

SEED = 42
BATCH_SIZE = 128


def process_raw_data(csv_path: str, describe_dataset=False):
    df = pd.read_csv(csv_path, header=None, index_col=None)
    df.columns = ['label', 'title', 'content']
    df = df.iloc[1:]
    df['label'] = df['label'].astype(int)
    df['label'] = df['label'] - 1
    df = df.sample(frac=1).reset_index(drop=True)

    file_name = csv_path.split("/")[-1]
    parent_folder = "/".join(csv_path.split("/")[:-1])
    save_dir = f"{parent_folder}/processed_{file_name}"
    df[['label', 'content']].to_csv(save_dir, index=None)

    if describe_dataset:
        unique_labels = np.unique(df['label'].values)
        label_count = np.bincount(df['label'])
        print(f"Data in '{file_name}':\nLabels: {unique_labels}\nLabel counts: {label_count}")

    del df
    
    
def save_dataset(dataset, path):
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save the TabularDataset using torchtext's serialization
    dataset.save(path)

def load_dataset(path, fields):
    if not isinstance(path, Path):
        path = Path(path)
    
    return TabularDataset(
        path=str(path),
        format='csv',  # Adjust the format if necessary
        skip_header=True,
        fields=fields
    )


def get_datasets(raw_data_path, use_processed=False, describe_dataset=False, valid_ratio=0.05, random_state=SEED):
    
    for stage in ['train', 'test']:
        data_path = f"{raw_data_path}/{stage}.csv"
        if os.path.exists(data_path) and use_processed:
            print(f"Using existing processed {stage}ing data...")
        else:
            print(f"Generating processed {stage}ing data...")
            process_raw_data(data_path, describe_dataset)

    text_pt_path = os.path.join(raw_data_path, 'text_vocab.pt')
    label_pt_path = os.path.join(raw_data_path, 'label_vocab.pt')

    nlp = spacy.load("en_core_web_sm")
    text = data.Field(sequential=True,
                        use_vocab=True,
                        tokenize=lambda text: [token.text for token in nlp(text)],
                        include_lengths=True)
    label = data.LabelField(dtype=torch.float)

    print("Generating datasets...")
    fields = [('label', label), ('content', text)]

    train_dataset, test_data = tuple(data.TabularDataset(
        path=f"{raw_data_path}/processed_{stage}.csv",
        format='csv', skip_header=True, fields=fields
    ) for stage in ['train', 'test'])
    
    train_data, validation_data = train_dataset.split(
        split_ratio=[1 - valid_ratio, valid_ratio],
        random_state=random.seed(random_state)
    )

    if os.path.exists(text_pt_path) and os.path.exists(label_pt_path) and use_processed:
        print("Using existing vocabularies...")
        text_vocab = torch.load(text_pt_path)
        label_vocab = torch.load(label_pt_path)
        text.vocab = text_vocab
        label.vocab = label_vocab
    else:
        print("Building new vocabularies...")
        text.build_vocab(train_data, max_size=5000, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
        label.build_vocab(train_data)
        torch.save(text.vocab, f"{raw_data_path}/text_vocab.pt")
        torch.save(label.vocab, f"{raw_data_path}/label_vocab.pt")
        print(f"Text and label vocabularies are saved to folder {raw_data_path}")

    print(f"Vocabulary size: {len(text.vocab)}")
    print(f"Number of classes: {len(label.vocab)}")
    print(f"Training samples: {len(train_data)}, Validation samples: {len(validation_data)}, Test samples: {len(test_data)}")
    return train_data, validation_data, test_data, text, label    


def get_dataloaders(train_data, validation_data, test_data, batch_size=BATCH_SIZE, device='cpu'):
    return data.BucketIterator.splits(
        (train_data, validation_data, test_data),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x : len(x.content),
        device=device
    )


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data, validation_data, test_data, text, label = get_datasets(raw_data_path='./AG_data',
                                                                          use_processed=True,
                                                                          describe_dataset=False)
    train_dl, validation_ld, test_ld = get_dataloaders(train_data, validation_data, test_data, device=device)
