import numpy as np
import torch


def test(loader, model, verbose=False, cls_acc=False):
    """test accuracy of the model

    Args:
        loader (torch.utils.data.DataLoader): _description_
        model (torch.nn.Modules): _description_
        verbose (bool, optional): If true, print the accuracy. Defaults to False.
        cls_acc (bool, optional): If true, print accuracy of each class. Defaults to False.

    Returns:
        float: total accuracy
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_correct = np.array([0 for _ in classes])
    class_total = np.array([0 for _ in classes])
    num_correct = 0
    num_samples = 0
    
    model = model.to(device=device)
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

            batch_total = torch.bincount(y).cpu().numpy()
            batch_correct = torch.bincount(y[preds==y]).cpu().numpy()
            class_total += np.pad(batch_total, (0,10-len(batch_total)), 'constant', constant_values=(0, 0))
            class_correct += np.pad(batch_correct, (0,10-len(batch_correct)), 'constant', constant_values=(0, 0))
        
        acc = float(num_correct) / num_samples
        
        if verbose:
            print("Total Accuracy: %.2f%%" % (100 * acc))
            if cls_acc:
                for i, value in enumerate(classes):
                    print("%s: %.2f%%" % (classes[i], 100 * float(class_correct[i]) / class_total[i]))
                
        return acc
