import argparse
import os
import time
from collections import defaultdict
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim.lr_scheduler as lr_scheduler
from dataset import ModelNet40
from model import NaivePCTCls, SPCTCls, PCTCls
from util import cal_loss, Logger
import matplotlib
matplotlib.use('agg')  # Use the 'agg' backend
import matplotlib.pyplot as plt
import pandas as pd
mse_loss = nn.MSELoss()


models = {'navie_pct': NaivePCTCls,
          'spct': SPCTCls,
          'pct': PCTCls}


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')


def train(args, io):
    history = defaultdict(list)
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=1,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=1,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = models[args.model]().to(device)
    model = nn.DataParallel(model)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    #scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    scheduler=lr_scheduler.StepLR(opt, step_size=7, gamma=0.1)
    criterion = cal_loss
    best_test_acc = 0

    for epoch in range(args.epochs):
        train_loss = []
        count = 0.0  # numbers of data
        model.train()
        train_pred = []
        real_val=[]
        train_true = []
        idx = 0  # iterations
        total_time = 0.0
        correct_predictions=0
        for data, label in (train_loader):
            data, label = data.to(device), label.to(device).float().squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()

            start_time = time.time()
            logits = model(data)
            #loss = criterion(logits, label)
#######ena bedeltou
            #print("\n********len(logits)",len(logits))
            #print("\n********len(label)",len(label))
            #print("\n********type(logits)",type(logits))
            #print("\n********type(label)",type(label))
            preds=torch.squeeze(logits)
            #print("\n********size(logits)",preds.size())
            #print(torch.tensor(logits))
            #label = label.to(device).float().squeeze()
            #print("\n********size(label)",label.size())
            #print(torch.tensor(label))
            loss =mse_loss(preds,label)
            #print("\n********loss",loss)
            real_val.extend(label.tolist())
            #print("\n**********************************\n",len(real_val))
            train_pred.extend(preds.tolist())
            #print("\n**********************************\n",len(train_pred))
            
            #print("****************************correct_predictions\n",correct_predictions)
            train_loss.append(loss.item())
            loss.backward()
            opt.step()
            opt.zero_grad()
            end_time = time.time()
            total_time += (end_time - start_time)
            count=count+batch_size
        threshold = 0.25
        #print("len train pred",len(train_pred))
        #print("len real val ", len (real_val))
        matches = (np.array(train_pred) >= (np.array(real_val) - threshold)) & (np.array(train_pred) <= (np.array(real_val) + threshold))
        #print("****************************\n",matches)
        #print("len matches ", len (matches))
        matches_tensor = torch.from_numpy(matches.astype(int))
        correct_predictions += torch.sum(matches_tensor)
        """
        plt.scatter(real_val, train_pred)
        plt.xlabel('Real Values')
        plt.ylabel('Predicted Values')
        plt.title('Real Values vs Predicted Values(train)')
        plt.show()
        """
        # Mise à jour du scheduler
        scheduler.step()    
        print ('train total time is',total_time)
        print('Train',epoch )
        print("Train_acc",float(correct_predictions)/(count))
        print("Train_loss",np.mean(train_loss))
                           

        ####################
        # Test
        ####################
        test_loss = []
        count = 0.0
        model.eval()
        test_pred = []
        real_val_test = []
        correct_predictions_test=0
        total_time = 0.0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).float().squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            start_time = time.time()
            logits = model(data)
            preds=torch.squeeze(logits)
            end_time = time.time()
            total_time += (end_time - start_time)
            loss =mse_loss(preds,label)
            real_val_test.extend(label.tolist())
            test_pred.extend(preds.tolist())
            count=count+batch_size
        threshold = 0.4
        matches = (np.array(test_pred) >= (np.array(real_val_test) - threshold)) & (np.array(test_pred) <= (np.array(real_val_test) + threshold))
        #print("****************************\n",matches)
        matches_tensor = torch.from_numpy(matches.astype(int))
        correct_predictions_test += torch.sum(matches_tensor)
        test_loss.append(loss.item())
            

            
            
        print("len(real_val_test)",len(real_val_test))
        print("len(test_pred)",len(test_pred))
        print ('test total time is', total_time)
        """
        plt.scatter(real_val_test, test_pred)
        plt.xlabel('Real Values')
        plt.ylabel('Predicted Values')
        plt.title('Real Values vs Predicted Values(val)')
        plt.show()
        """
        print('test',epoch )
        print("test_acc",float(correct_predictions_test)/count)
        print("Test_loss",np.mean(test_loss))
        test_acc= float(correct_predictions_test)/count
        history['train_acc'].append(float(correct_predictions)/(count))
        history['train_loss'].append(np.mean(train_loss))
        history['val_acc'].append(test_acc)
        history['val_loss'].append(np.mean(test_loss))

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
            best_test_loss=np.mean(test_loss)
            best_train_loss=np.mean(test_loss)
        scheduler.step()


    # Your code for creating the plots here
    epochs = range(1, len(history['train_loss']) + 1)

    plt.plot(epochs, history['train_loss'], 'b', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training Loss vs. Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_validation_loss.png')  # Save the plot to a file
    plt.close()  # Close the figure to release resources

    plt.plot(epochs, history['train_acc'], 'b', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r', label='Validation Accuracy')
    plt.title('Training Accuracy vs. Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_validation_accuracy.png')  # Save the plot to a file
    plt.close()  # Close the figure to release resources
    print("best acc",best_test_acc)
    print("best test loss",best_test_loss)
    print("best train loss ",best_train_loss)

def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = models[args.model]().to(device)
    model = nn.DataParallel(model) 
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    real_val_test = []
    test_pred = []
    correct_predictions_test=0
    count=0
    test_loss=[]
    for data, label in test_loader:
        
        data, label = data.to(device), label.to(device).float().squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        preds = torch.squeeze(logits)
        real_val_test.extend(label.tolist())
        test_pred.extend(preds.tolist())
        count=count+batch_size
    threshold = 0.25
    matches = (np.array(test_pred) >= (np.array(real_val_test) - threshold)) & (np.array(test_pred) <= (np.array(real_val_test) + threshold))
    matches_tensor = torch.from_numpy(matches.astype(int))
    correct_predictions_test += torch.sum(matches_tensor)
    loss =mse_loss(preds,label)
    test_loss.append(loss.item())
        
    print("\n len(real_val_test)",len(real_val_test))
    plt.scatter(real_val_test, test_pred,c="r")
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    plt.title('Real Values vs Predicted Values(test)')
    plt.show()
    print("test_acc",float(correct_predictions_test)/count)
    print("Test_loss",np.mean(test_loss))
    df = pd.DataFrame({'real_val_test': real_val_test, 'test_pred': test_pred})
    print(df)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='pct', choices=['navie_pct', 'spct', 'pct'],
                        help='which model you want to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_(args)

    io = Logger('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
