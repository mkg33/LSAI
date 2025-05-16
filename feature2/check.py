from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import PreTokenisedParquetDataset, CollatorForCLM

tok = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")
ds = PreTokenisedParquetDataset("/iopsstor/scratch/cscs/mgwozdz/datasets/train_data_tok.parquet", sequence_length=200)
dl = DataLoader(ds, batch_size=16, shuffle=True,
                collate_fn=CollatorForCLM(sequence_length=200,
                                          pad_token_id=tok.pad_token_id))
x, y = next(iter(dl))
assert x.shape == y.shape == (16, 200)
assert (y != -100).sum()  # at least one non-masked token
print("all shapes & masking OK")
