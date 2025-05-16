import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer
import torch
from dataclasses import dataclass
from typing import List, Dict


class PreTokenisedParquetDataset(Dataset):
    """
    Each Parquet row contains a *list<int32>* field 'input_ids'
    of length *sequence_length+1* from `pretokenize.py`.
    """
    def __init__(self, parquet_file: str, sequence_length: int):
        self.table = pq.read_table(parquet_file, memory_map=True)
        if "input_ids" not in self.table.column_names:
            raise ValueError(
                f"{parquet_file} lacks column 'input_ids'. "
                "Run pretokenize.py first."
            )
        self.sequence_length = sequence_length
        self._len = len(self.table)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        token_ids = self.table["input_ids"][idx].as_py()
        assert (
            len(token_ids) == self.sequence_length + 1
        ), f"Row {idx} has length {len(token_ids)}"
        return {"input_ids": token_ids}

class ParquetDataset(Dataset):
    def __init__(self, parquet_file: str, tokenizer, sequence_length: int, training_samples: int):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.training_samples = training_samples
    def __len__(self):
        return self.training_samples
    def __getitem__(self, idx: int):
        sample_str = str(self.parquet_ds["text"][idx % self.real_length])
        return self.tokenizer.encode_plus(
            sample_str,
            max_length=self.sequence_length + 1,
            padding='max_length',
            truncation=True,
            padding_side="right"
        )

class IterableParquetDataset(IterableDataset):
    def __init__(self, parquet_file: str, tokenizer, sequence_length: int, bos_token_id: int = 1):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.bos_token_id = bos_token_id
        self.current_index = 0
        self.token_buffer = []
    def __iter__(self):
        self.token_buffer = []
        self.current_index = 0
        return self
    def __next__(self):
        while len(self.token_buffer) < self.sequence_length:
            if self.current_index >= self.real_length:
                raise StopIteration
            text_str = str(self.parquet_ds["text"][self.current_index])
            self.current_index += 1
            token_ids = self.tokenizer.encode(text_str, add_special_tokens=False)
            token_ids = [self.bos_token_id] + token_ids
            self.token_buffer.extend(token_ids)
        chunk = self.token_buffer[:self.sequence_length]
        self.token_buffer = self.token_buffer[self.sequence_length:]
        inputs = torch.tensor(chunk, dtype=torch.long)
        labels = torch.full((self.sequence_length,), -100, dtype=torch.long)
        if len(chunk) > 1:
            labels[0] = chunk[1]
        return inputs, labels

dataset_path = "/capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet"
sequence_length = 200
tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")
dataset = ParquetDataset(parquet_file=dataset_path, tokenizer=tokenizer, sequence_length=sequence_length, training_samples=32)
sample = dataset[0]
first_200_tokens = sample["input_ids"][:200]
decoded_text = tokenizer.decode(first_200_tokens, skip_special_tokens=True)

@dataclass
class CollatorForCLM:
    sequence_length: int
    pad_token_id: int
    def __call__(self, examples: List[Dict[str, List[int]]]):
        input_ids = torch.LongTensor([examples[i]["input_ids"] for i in range(len(examples))])
        inputs = input_ids[:, :-1].clone()
        labels = input_ids[:, 1:]
        labels[labels == self.pad_token_id] = -100
        assert inputs.shape[1] == labels.shape[1] == self.sequence_length
        assert inputs.shape == labels.shape
        return inputs, labels

collator = CollatorForCLM(sequence_length=sequence_length, pad_token_id=tokenizer.pad_token_id)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
for batch_inputs, batch_labels in dataloader:
    print(f"Input shape: {batch_inputs.shape}")
    print(f"Labels shape: {batch_labels.shape}")
    ignored_count = (batch_labels == -100).sum().item()
    total_label_tokens = batch_labels.numel()
    print(f"Ignored tokens in loss: {ignored_count} out of {total_label_tokens} ({ignored_count/total_label_tokens*100:.2f}%)")
    break
