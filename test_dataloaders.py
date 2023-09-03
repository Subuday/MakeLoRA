from dataloader import DataLoader
from data_utils import FT_Dataset


def test_data_loader_length():
    batch_size = 8
    train_data = FT_Dataset("./data/e2e/train.jsonl", batch_size=batch_size, max_seq_length=512)
    assert DataLoader(dataset=train_data, batch_size=batch_size).__len__() == 5258


def test_data_loader_data_shape():
    batch_size = 8
    dataset = FT_Dataset("./data/e2e/train.jsonl", batch_size=batch_size, max_seq_length=512)
    dataloader = DataLoader(dataset=dataset, batch_size=8)

    for batch in dataloader:
        assert batch["id"].shape == (batch_size, )
        assert batch["query"].shape == (batch_size, 512)
        assert batch["query_len"].shape == (batch_size, )
        assert batch["input"].shape == (batch_size, 512)
        assert batch["target"].shape == (batch_size, 512)
        assert batch["mask"].shape == (batch_size, 512)
        break
