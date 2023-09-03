from data_utils import padding_tokens, FT_Dataset


def test_padding_tokens():
    assert padding_tokens([1, 2, 3], 5, 0, direct=1) == ([1, 2, 3, 0, 0], 3)


def test_ft_dataset_length():
    assert FT_Dataset(ft_file="data/e2e/train.jsonl", batch_size=8, max_seq_length=5).__len__() == 42064
