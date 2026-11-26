"""GLUE data module: tokenization, train/dev split and dataloaders.

The training set is split 85/15 with a fixed generator (seed 0); the 15% part is the
validation set and the GLUE validation set is used as the test set. MNLI has two
validation sets, so its dataloaders are lists (matched, mismatched, 15% split).
"""
import torch
import datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, DataCollatorWithPadding

import common


class SimpleDataModule():
    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 512,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        dev_frac=0.15,
        seed=0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.dev_frac = dev_frac
        self.seed = seed
        self.model_config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_fields = common.TASK_TEXT_FIELDS[task_name]
        self.num_labels = common.TASK_NUM_LABELS[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)
        hf_collator = DataCollatorWithPadding(self.tokenizer)
        self.collator = lambda x: {kx: vx for kx, vx in hf_collator(x).items()}
        self.dataset = None

    def setup(self):
        if self.dataset is None:
            self.dataset = datasets.load_dataset("glue", self.task_name)

        train_size = len(self.dataset["train"])
        train_set_size = int(train_size * (1 - self.dev_frac))
        test_set_size = train_size - train_set_size
        split_generator = torch.Generator().manual_seed(0)
        train_set, test_set = torch.utils.data.random_split(
            self.dataset["train"], [train_set_size, test_set_size], generator=split_generator)
        self.dataset["train"] = train_set.dataset.select(train_set.indices)
        if self.task_name == "mnli":
            self.dataset["test_matched"] = self.dataset["validation_matched"]
            self.dataset["test_mismatched"] = self.dataset["validation_mismatched"]
        else:
            self.dataset["test"] = self.dataset["validation"]
        self.dataset["validation"] = test_set.dataset.select(test_set.indices)
        del train_set
        del test_set

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)
        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]
        self.test_splits = [x for x in self.dataset.keys() if "test" in x]

    def train_dataloader(self):
        train_gen = torch.Generator(device="cpu")
        train_gen.manual_seed(self.seed)
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, num_workers=0,
                          collate_fn=self.collator, generator=train_gen, pin_memory=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, num_workers=0,
                              collate_fn=self.collator, pin_memory=True)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=0,
                               collate_fn=self.collator, pin_memory=True) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.test_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, num_workers=0,
                              collate_fn=self.collator)
        elif len(self.test_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=0,
                               collate_fn=self.collator) for x in self.test_splits]

    def convert_to_features(self, example_batch, indices=None):
        if len(self.text_fields) > 1:
            features = self.tokenizer(
                example_batch[self.text_fields[0]], example_batch[self.text_fields[1]],
                max_length=self.max_seq_length, truncation=True, padding=True
            )
        else:
            features = self.tokenizer(
                example_batch[self.text_fields[0]], max_length=self.max_seq_length, truncation=True, padding=True
            )
        if "label" in example_batch:
            features["labels"] = example_batch["label"]
        return features
