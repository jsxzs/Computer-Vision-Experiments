import numpy as np
import random
import torch
        
def data_poison(dataset, ratio:float, target_label:int, combine=True):
    """poison dataset for backdoor attacks

    Args:
        dataset (torch.utils.data.Dataset): Dataset to be poisoned.
        ratio (float): Poisoned sample number over total sample number, in every class
        target_label (int): All labels of poisoned samples will be set to target_label
        combine (bool, optional): If True, the returned dataset will contain poisoned and original samples; if false, only poisoned samples will be returned. Defaults to True.

    Returns:
        list: new dataset
    """
    # classify samples in dataset
    classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    data_cls = [[] for _ in classes]       
    for d in dataset:
        data_cls[d[1]].append(d)
        
    # poison data
    poisoned_data = list()
    for i, _ in enumerate(classes):
        if i != target_label:
            poison_num = int(ratio * len(data_cls[i]))
            random.seed(1)
            index = np.random.randint(len(data_cls[i]), size=poison_num)
            for j in index:
                img = data_cls[i][j][0].clone()
                img = img.permute(1, 2, 0)
                # implant a mark in images
                # set a 4x4 white square on the bottom right corner
                img[-4:, -4:, :] = torch.from_numpy((1.0 - np.array([0.4914, 0.4822, 0.4465])) 
                                                        / np.array([0.2023, 0.1994, 0.2010]))
                img = img.permute(2, 0, 1)
                                
                poisoned_data.append((img, target_label))
    

    poisoned_dataset = list()
    # combine poisoned data with original data
    if combine:
        poisoned_dataset += list(dataset)
    poisoned_dataset += poisoned_data
    
    random.seed(10)
    random.shuffle(poisoned_dataset)
    
    return poisoned_dataset
    
