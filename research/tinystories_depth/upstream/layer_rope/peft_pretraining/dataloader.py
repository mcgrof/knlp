import itertools

import torch
from torch.utils.data import IterableDataset, get_worker_info


class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length, legacy_dataloader=False, multi_epoch=False,
                 skip_batches=0):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.legacy_dataloader = legacy_dataloader
        self.multi_epoch = multi_epoch
        self.skip_batches = skip_batches  # batches this rank consumed before the resumed checkpoint

    def _epochs(self):
        """Repeated passes over the data, each reshuffled through set_epoch."""
        epoch = 0
        while True:
            if epoch > 0:
                self.data.set_epoch(epoch)
            n = 0
            for example in self.data:
                n += 1
                yield example
            if n == 0:
                raise RuntimeError(f"pass {epoch} over the training data yielded no examples")
            epoch += 1

    def __iter__(self):
        examples = self._epochs() if self.multi_epoch else self.data
        worker_info = get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        if self.skip_batches % num_workers != 0:
            raise ValueError(
                f"Cannot resume exactly: skip_batches={self.skip_batches} is not divisible by "
                f"num_workers={num_workers} (the DataLoader hands out batches round-robin over workers)."
            )
        skip = (self.skip_batches // num_workers) * self.batch_size  # examples this worker already consumed
        if self.legacy_dataloader and worker_info is not None:
            # Upstream LayerNorm-Scaling sharding, kept to reproduce our depth-scaling runs:
            # https://github.com/lmsdss/LayerNorm-Scaling/blob/3ac54dde961b04780b152bc8b38b71b2f8e1c16a/peft_pretraining/dataloader.py#L20-L24
            # HF datasets already gives each DataLoader worker its own shards, so this second stride
            # reads only 1/num_workers of the data.
            iter_data = itertools.islice(examples, worker_info.id + num_workers * skip, None, num_workers)
        else:
            iter_data = itertools.islice(examples, skip, None)

        batch = []
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}
