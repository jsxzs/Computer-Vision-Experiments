import torch
from .test import test

def get_fm_size(model, device):
    model.eval()
    model(torch.zeros(1, 3, 32, 32).to(device))
    return model.get_features().shape


def mean_feature_maps(loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    feature_maps = torch.zeros(get_fm_size(model, device)).to(device)
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _ = model(x)
            
            feature_maps += model.get_features().mean(dim=0, keepdim=True)
    
    return feature_maps.squeeze().cpu() / len(loader)


def prune(model, features, loader):
    args_sort = torch.argsort(features.mean(dim=(1, 2)).squeeze())

    num = 32
    pruned_num = list()
    acc_history = list()
    for i in range(512//num+1):
        pruned_num.append(i*num)
        
        # prune
        model.get_pruned_layer().weight.data[args_sort<pruned_num[-1]] = 0
        model.get_pruned_layer().bias.data[args_sort<pruned_num[-1]] = 0
        
        acc = test(loader, model)
        print("%d neurons: %.2f%%" % (pruned_num[-1], acc))
        acc_history.append(acc)
        
    return pruned_num, acc_history