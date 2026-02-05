from torch.utils.data import Dataset

from cookie.data import MyDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)
