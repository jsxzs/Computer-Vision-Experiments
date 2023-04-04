import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def visualize_single(train_loss_history, val_loss_history, train_acc_history, val_acc_history, lr_history):
    """visualize the training process of a single model

    Args:
        loss_history (list): loss history
        train_acc_history (list): accuracy history on training dataset
        val_acc_history (list): accuracy history on validation dataset
    """
    plt.figure(figsize=(15, 2.5))

    # plot the training loss curve
    plt.subplot(1, 3, 1)
    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.plot(train_loss_history, '-', label='train')
    plt.plot(val_loss_history, '-', label='val')
    plt.legend()

    # plot the training and validation accuracy curves
    plt.subplot(1, 3, 2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(train_acc_history, '-', label='train')
    plt.plot(val_acc_history, '-', label='val')
    plt.legend()
    
    # plot the learning rate curve
    plt.subplot(1, 3, 3)
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.plot(lr_history, '-', label='lr')
    plt.legend()
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    
def visualize_multi(names, train_loss, train_acc, val_loss, val_acc):
    """visualize the validation accuracy history of multiple models to compare

    Args:
        names (list): model names
        acc_history (list): the val_acc_history of each model
    """
    plt.figure(figsize=(10,6))
    plt.subplots_adjust(wspace =0.3, hspace =0.5)#调整子图间距
    
    # train loss
    plt.subplot(2, 2, 1)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    for idx, nm in enumerate(names):
        plt.plot(train_loss[idx], '-', label=nm)
    plt.legend()
    
    # train acc
    plt.subplot(2, 2, 2)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    for idx, nm in enumerate(names):
        plt.plot(train_acc[idx], '-', label=nm)
    plt.legend()
    
    # val loss
    plt.subplot(2, 2, 3)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    for idx, nm in enumerate(names):
        plt.plot(val_loss[idx], '-', label=nm)
    plt.legend()
    
    # val acc
    plt.subplot(2, 2, 4)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    for idx, nm in enumerate(names):
        plt.plot(val_acc[idx], '-', label=nm)
    plt.legend()
    
    plt.show()