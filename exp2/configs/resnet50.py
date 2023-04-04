from torch.utils.data import DataLoader, sampler, Dataset
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.optim as optim

import sys
sys.path.append('./modules')
from resnet import resnet50


#! Dataset settings
NUM_TRAIN = 49000

transform_test = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# simple data augmentation for training
transform_train = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# load training dataset
dset_train = dset.CIFAR10('./datasets', train=True, download=True, transform=transform_train)
loader_train = DataLoader(dset_train, batch_size=128, num_workers=4,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

# load validation dataset
dset_val = dset.CIFAR10('./datasets', train=True, download=True, transform=transform_test)
loader_val = DataLoader(dset_val, batch_size=128, num_workers=4,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

# load test dataset
dset_test = dset.CIFAR10('./datasets', train=False, download=True, transform=transform_test)
loader_test = DataLoader(dset_test, batch_size=128, num_workers=4)


#! model settings
model = resnet50(pretrained=True, num_classes=10)

#! optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)

#! learning policy
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                    mode='max', 
                                                    factor=0.5, 
                                                    patience=10, 
                                                    verbose=True)

#! training settings
total_epochs = 30
grad_clip = None
# checkpoint_config = dict(interval=12)
work_dir = './work_dir/resnet/resnet50'
resume_from = None

#! log settings
log_config = dict(interval=70)

