from torch.utils.data import Subset
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

fine_to_coarse = [4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                  3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                  6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                  0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                  5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                  16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                  10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                  2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13]


class CIFAR100Subset(CIFAR100):
    def __init__(self, subset: list[int], **kwargs):
        super().__init__(**kwargs)
        self.subset = subset
        assert max(subset) <= max(self.targets)
        assert min(subset) >= min(self.targets)

        self.aligned_indices = []
        for idx, label in enumerate(self.targets):
            if fine_to_coarse[label] in subset:
                self.aligned_indices.append(idx)

    def get_class_names(self):
        return [self.classes[i] for i in self.subset]

    def __len__(self):
        return len(self.aligned_indices)

    def __getitem__(self, item):
        return super().__getitem__(self.aligned_indices[item])


def partition_data_on_coarse_labels(root, partition, train=True):

    minimal_transform = transforms.Compose([transforms.ToTensor()])
    cifar100_subset = CIFAR100Subset(
        subset=partition,
        root=root,
        train=train,
        download=True,
        transform=minimal_transform
    )
    return cifar100_subset
