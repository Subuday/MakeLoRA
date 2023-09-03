from typing import Generic, TypeVar, Union, Iterable, Callable, List, Any

import torch
from torch.utils.data import Dataset, IterDataPipe, MapDataPipe, IterableDataset, BatchSampler, Sampler, \
    SequentialSampler
from torch.utils.data import _utils

from torch.utils.data.datapipes.datapipe import _IterDataPipeSerializationWrapper, _MapDataPipeSerializationWrapper, T

from data_utils import FT_Dataset

T_co = TypeVar('T_co', covariant=True)

# Ideally we would parameterize `DataLoader` by the return type of `collate_fn`, but there is currently no way to have that
# type parameter set to a default value if the user doesn't pass in a custom 'collate_fn'.
# See https://github.com/python/mypy/issues/3737.
_collate_fn_t = Callable[[List[T]], Any]

default_collate: _collate_fn_t = _utils.collate.default_collate


class _DatasetKind:
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)


class DataLoader(Generic[T_co]):

    def __init__(
            self,
            dataset: Dataset[T_co],
            batch_size: int,
            shuffle: bool = False,
            generator=None
    ):
        self.dataset = dataset
        self._dataset_kind = _DatasetKind.Map
        self.generator = generator
        if shuffle:
            self.sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
        else:
            self.sampler = SequentialSampler(dataset)  # type: ignore[arg-type]
        self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last=False)
        self.batch_size = batch_size
        self.collate_fn = _utils.collate.default_collate

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        return _SingleProcessDataLoaderIter(self)

    # We quote '_BaseDataLoaderIter' since it isn't defined yet and the definition can't be moved up
    # since '_BaseDataLoaderIter' references 'DataLoader'.
    def __iter__(self) -> '_BaseDataLoaderIter':
        return self._get_iterator()

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        return self.batch_sampler

    def __len__(self) -> int:
        return len(self._index_sampler)


class _BaseDataLoaderIter:

    def __init__(self, loader: DataLoader) -> None:
        self._dataset = loader.dataset
        self._dataset_kind = loader._dataset_kind
        self._auto_collation = loader._auto_collation
        self._index_sampler = loader._index_sampler
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)

    def __iter__(self) -> '_BaseDataLoaderIter':
        return self

    def _reset(self):
        self._sampler_iter = iter(self._index_sampler)

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        if self._sampler_iter is None:
            # TODO(https://github.com/pytorch/pytorch/issues/76750)
            self._reset()  # type: ignore[call-arg]
        data = self._next_data()
        return data

    def __len__(self) -> int:
        return len(self._index_sampler)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self._dataset,
            self._auto_collation,
            self._collate_fn,
            drop_last=False
        )

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        return data


if __name__ == '__main__':
    train_data = FT_Dataset("./data/e2e/train.jsonl", batch_size=8, max_seq_length=512)
    dataloader = DataLoader(dataset=train_data, batch_size=8)
    for data in dataloader:
        print(data["id"].shape)
        print(data["query"].shape)
        print(data["input"].shape)
        print(data["mask"].shape)
        break
