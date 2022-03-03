import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from tqdm import tqdm
import time

## off the shelf networks:
import cifarAlexNet
import cifarResNet
import cifarVGG

def epoch(loader, model, opt=None, device=None, use_tqdm=False):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    if opt is None:
        model.eval()
    else:
        model.train()

    if use_tqdm:
        pbar = tqdm(total=len(loader))

    model.to(device)
    for X,y in loader:
        X,y = X.to(device), y.to(device)
  

        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]

        if use_tqdm:
            pbar.update(1)

    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def train_model( model, model_path, train_loader, test_loader, device ):
    model.to(device)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for ep in range(80):
        if ep == 50:
            for param_group in opt.param_groups:
                    param_group['lr'] = 0.01
        train_err, train_loss = epoch(train_loader, model, opt, device=device, use_tqdm=False)  #train for 1 epoch
        test_err, test_loss = epoch(test_loader, model, device=device, use_tqdm=False)  #evaluate accuracy on test dataset

        print('epoch', ep, 'train err', train_err, 'test err', test_err)
        torch.save(model.state_dict(), model_path)
    return model

def pgd_linf_untargeted(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def epoch_transfer_attack(loader, model_source, model_target, attack, device, use_tqdm=True, n_test=None, **kwargs):
    source_err = 0.
    target_err = 0.
    target_err2 = 0.
    
    model_source.eval()
    model_target.eval()

    total_n = 0

    if use_tqdm:
        pbar = tqdm(total=n_test)

    model_source.to(device)
    model_target.to(device)
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model_source, X, y, **kwargs)


        yp_target = model_target(X+delta).detach()
        yp_source = model_source(X+delta).detach()
        source_err += (yp_source.max(dim=1)[1] != y).sum().item()
        target_err += (yp_target.max(dim=1)[1] != y).sum().item()
        if use_tqdm:
            pbar.update(X.shape[0])

        total_n += X.shape[0]
        if n_test is not None:
            if total_n >= n_test:
                break

    return source_err / total_n, target_err / total_n


def trainOrLoad(model, model_name, train_loader, test_loader, device):
    print('training/loading:', model_name, '.....')
    models_dir = './models'
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    model_path = models_dir + '/' + model_name + '.pth'
    if os.path.exists( model_path ):
        model.load_state_dict( torch.load( model_path, map_location=device ) )
        print('loaded', model_path)
    else:
        print('training', model_path)
        model = train_model(model, model_path, train_loader, test_loader, device)   
    return model 

def compareModels( model_comparison, comparison_name, model_target, target_name, test_loader, device ):
    eps = 0.031    
    err1, err2 = epoch_transfer_attack(test_loader, model_comparison, model_target, attack=pgd_linf_untargeted, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps, randomize=True, alpha=0.003, n_test=5000) # n_test is the number of test examples you use to evaluate the success rate
    print('\n', target_name, comparison_name, '\ndiff nets scores, comparison:', err1, err2, '\n')
    return

def trainScore_3_32_32( savedModelFile, savedOtherFile, batchSize ):
    print('comparing models')
    ### load datasets
    norm_mean = 0
    norm_var = 1
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
    ])
    cifar_train = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
    cifar_test = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
    train_loader = DataLoader(cifar_train, batch_size = batchSize, shuffle=True)
    test_loader = DataLoader(cifar_test, batch_size = batchSize, shuffle=True)

    ### train on CIFAR-10
    device = torch.device('cuda:0')


    # model extracted from processing the signal
    guess_name = os.path.basename(savedModelFile).split("_")[-2] + '_' + os.path.basename(savedModelFile).split("_")[-1][:-4] 
    model_guess = trainOrLoad( torch.load( savedModelFile ), guess_name, train_loader, test_loader, device )

    # model for comparison:
    target_name = os.path.basename(savedOtherFile).split("_")[-2] + '_' + os.path.basename(savedOtherFile).split("_")[-1][:-4] 
    model_target = trainOrLoad( torch.load( savedOtherFile ), target_name, train_loader, test_loader, device )

    test_err, test_loss = epoch(test_loader, model_target, device=device, use_tqdm=False)  #evaluate accuracy on test dataset
    print('target_name', target_name, 'test_err', test_err, 'test_loss', test_loss)

    test_err, test_loss = epoch(test_loader, model_guess, device=device, use_tqdm=False)  #evaluate accuracy on test dataset
    print('proxy_name', guess_name, 'test_err', test_err, 'test_loss', test_loss)        

    # compute scores:
    compareModels( model_guess, guess_name, model_target, target_name, test_loader, device )
    return

def main():
    if len(sys.argv) == 4:
        trainScore_3_32_32( sys.argv[1], sys.argv[2], int(sys.argv[3]) )
    else:
        print('missing windowLength argument, or too many arguments')
        print('insufficient arguments %d' % len(sys.argv))
        for i in range(len(sys.argv)):
            print('%d %s' % (i, sys.argv[i]))

if __name__ == '__main__':
    main()
