import argparse
import numpy as np
import torch.nn as nn
import torch
import logging
import os
import time
import random
import json
import warnings
warnings.filterwarnings("ignore")
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', type=str, help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--pretrained', action='store_true', help='whether to load pretrained weights on ImageNet')
    parser.add_argument('--resume_from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def getLogger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # FileHandler
    fHandler = logging.FileHandler(log_file, mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)  # print format in FileHandler
    logger.addHandler(fHandler)  

    return logger


def write_json(data, json_file):
    with open(json_file, 'a+', encoding='utf-8') as f:
        line = json.dumps(data, ensure_ascii=False)
        f.write(line+'\n')

    
def check_accuracy(loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    num_correct = 0
    num_samples = 0
    model = model.to(device=device)
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        loss = 0
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            loss += criterion(scores, y).item()
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        return loss/len(loader), acc

#TODO: save checkpoint every few epochs and training can resume from certain checkpoint
def train(model,
          optimizer,
          lr_scheduler,
          loader_train,
          loader_val,
          logger,
          json_file,
          work_dir=None,
          grad_clip=None,
          epochs=1,
          print_every=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)  # move the model parameters to the device  
    criterion = nn.CrossEntropyLoss()
    
    best_epoch, best_acc, best_state_dict = 0, 0, None
    
    # start training
    logger.info("Start running, max: %d epochs" % epochs)
    for e in range(epochs):
        train_loss = 0
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = criterion(scores, y)

            # Zero out all of the gradients for the variables which the optimizer will update.
            optimizer.zero_grad()

            loss.backward()  # backwards pass
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step() # update the parameters of the model
            train_loss += loss.item()

            # print in log and write in json
            if (t+1) % print_every == 0:
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                logger.info("Epoch [%d][%d/%d] loss: %.4f, lr: %.5f" % 
                            (e+1, t+1, len(loader_train), loss.item(), lr))
                write_json({'mode': "train", 'epoch': e+1, 'iter': t+1, 'loss': loss.item(), 'lr': lr}, 
                           json_file)
                
            # # update the learning rate
            # lr_scheduler.step()
        
        train_loss /= len(loader_train)

        # Check train and val accuracy after each epoch
        _, train_acc = check_accuracy(loader_train, model)
        val_loss, val_acc = check_accuracy(loader_val, model)
        
        # print in log and write in json
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info("Epoch [%d][%d/%d] train_loss: %.4f, val_loss: %.4f, lr: %.5f, train_acc: %.4f%%, val_acc: %.4f%%" % 
                    (e+1, t+1, len(loader_train), train_loss, val_loss, lr, 100*train_acc, 100*val_acc))
        write_json({'mode': "check", 'epoch': e+1, 'iter': t+1, 'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc, 'lr': lr}, 
                   json_file)
        
        # update the best accuracy and the best model
        if val_acc > best_acc:
            best_epoch = e+1
            best_acc = val_acc
            best_state_dict = model.state_dict()
        
        # update the learning rate
        lr_scheduler.step(train_loss)
        
        # # stop training when current learning rate is smaller than target_lr
        # if optimizer.state_dict()['param_groups'][0]['lr'] < target_lr:
        #     break
    
    model.load_state_dict(best_state_dict)
    # save weights of the best model
    if work_dir is not None:
        torch.save(best_state_dict, work_dir+f'/epoch{best_epoch}.pth')

    return


def main():
    args = parse_args()
    
    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed)

    # read config
    cfg = Config.fromfile(args.config)
    
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = cfg.work_dir + '/' + filename + '.log'
    json_file = cfg.work_dir + '/' + filename + '.json'
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)
    logger = getLogger(log_file)
    
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))

    train(
        cfg.model,
        cfg.optimizer,
        cfg.lr_scheduler,
        cfg.loader_train,
        cfg.loader_val,
        logger,
        json_file,
        work_dir=cfg.work_dir,
        grad_clip=cfg.grad_clip,
        epochs=cfg.total_epochs,
        print_every=cfg.log_config['interval'])    

if __name__ == '__main__':
    main()
