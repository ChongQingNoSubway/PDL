from __future__ import print_function

import numpy as np
import argparse
import matplotlib.pyplot as plt
plt.rcParams.update({'text.color': "white"})

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from utils import *
from mnist_dataloader import MnistBags
from model import Attention

from pdl import LinearScheduler


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--num_epoch', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--weight_decay', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=100, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--PDL', default=False, type=bool, help='whether to use sprase coding')
parser.add_argument('--model', default='attention', type=str, help='which attention mode')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                               mean_bag_length=args.mean_bag_length,
                                               var_bag_length=args.var_bag_length,
                                               num_bag=args.num_bags_train,
                                               seed=args.seed,
                                               train=True),
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)

test_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                              mean_bag_length=args.mean_bag_length,
                                              var_bag_length=args.var_bag_length,
                                              num_bag=args.num_bags_test,
                                              seed=args.seed,
                                              train=False),
                                    batch_size=1,
                                    shuffle=False,
                                    **loader_kwargs)

if args.model=='attention':
    model = Attention()
elif args.model=='gated_attention':
    model = GatedAttention()
if args.cuda:
    model.cuda()

criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0)

dropout_schedule = LinearScheduler(model,start_value=0,stop_value=0.2,nr_steps=args.num_epoch)

def train(epoch):
    model.train()
    train_loss = 0.
    train_accuracy = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        
        data = data.squeeze(0)
        bag_label = bag_label.float()

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        bag_prediction, _ = model(data)
        
        loss_bag = criterion(bag_prediction.squeeze(0), bag_label)
        loss_total = loss_bag 
        loss_total = loss_total.mean()
        # backward pass
        loss_total.backward()
        # step
        optimizer.step() 

        train_loss += loss_total.item()
        train_accuracy += (torch.sigmoid(bag_prediction.squeeze(0)) > 0.5).eq(bag_label).item()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}, Train acc: {:.4f}'.format(epoch, train_loss, train_accuracy))


def vis_attention(bags:np.ndarray, attention_weights:np.ndarray, instance_labels:np.ndarray, save_path="test.jpg", topk=5):
    topk_ind = attention_weights.argsort()[-topk:][::-1]

    bag_length = bags.shape[0]

    nrows = 2

    if bag_length % nrows == 0:
        ncols = bag_length // nrows
    else:
        ncols = (bag_length // nrows) + 1 

    # ncols = int(np.sqrt(bag_length))

    # if bag_length % ncols == 0:
    #     nrows = (bag_length // ncols)
    # else:
    #     nrows = (bag_length // ncols) + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*2 - 1.5, nrows*2))

    for i in range(bag_length):
        xi, yi = np.unravel_index(i, (nrows, ncols))
        ax = axes[xi, yi]

        if int(instance_labels[i]) == 1:
            ax.imshow(bags[i], cmap='hot')
        else:
            ax.imshow(bags[i], cmap='gray')

        if i in topk_ind:
            # ax.set_title(f"$a_{{{i}}}$ = {attention_weights[i]:.3f}")
            ax.text(0.47, -0.08, f"$a_{{{i}}}$ = {attention_weights[i]:.5f}", {'color': 'red', 'fontsize': 14}, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        else:
            # ax.set_title(f"$a_{{{i}}}$ = {attention_weights[i]:.3f}")
            ax.text(0.47, -0.08, f'$a_{{{i}}}$ = {attention_weights[i]:.5f}', {'color': 'black', 'fontsize': 14}, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.savefig(save_path, dpi=400)
    plt.close('all')

def test(verbose=0, vis=False):
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        instance_labels = label[1]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        data = data.squeeze(0)
        bag_label = bag_label.float()

        bag_prediction, attention_weights = model(data)
        loss_bag = criterion(bag_prediction.squeeze(0), bag_label)
        loss_total = loss_bag 
        loss_total = loss_total.mean()
        
        test_loss += loss_total.item()
        
        predicted_label = (torch.sigmoid(bag_prediction.squeeze(0)) > 0.5)
        test_accuracy += predicted_label.eq(bag_label).item()

        if verbose:
            if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
                bag_level = (bag_label.long().cpu().data.numpy()[0], int(predicted_label.long().cpu().data.numpy()[0]))
                instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                    np.round(attention_weights.cpu().data.numpy().astype(np.float64)[0], decimals=3).tolist()))

                print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                    'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

        if vis:
            if int(bag_label.squeeze()):
                attention_weights_np = attention_weights.cpu().data.numpy().squeeze(0)
                
                data_plot = data.squeeze().cpu().numpy()
                save_path = join(save_dir, "vis", "bag_" + str(batch_idx) + "-lab_" + str(int(predicted_label)) + str(int(bag_label)) + ".jpg")
               
                vis_attention(data_plot, attention_weights_np, instance_labels.numpy()[0].tolist(), save_path)

    test_accuracy /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test acc: {:.4f}'.format(test_loss, test_accuracy))

    return test_loss, test_accuracy


if __name__ == "__main__":
    save_dir = join("trained_models/mnist", "wod" if args.PDL else "wod", f"num_train_bag_{args.num_bags_train}")
    maybe_mkdir_p(save_dir)
    maybe_mkdir_p(join(save_dir, "vis"))
    print('Start Training')
    best_accuracy = 0.0

    for epoch in range(1, args.num_epoch + 1):
        #dropout_schedule.step()
        train(epoch)
        test_loss, test_accuracy = test(verbose=0, vis=False)
        scheduler.step()

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

            torch.save(model.state_dict(), join(save_dir, "model_best.pt"))

    torch.save(model.state_dict(), join(save_dir, "model_last.pt"))

    # <------------------ Visualization ------------------>
    print("resuming the best model ...")
    model.load_state_dict(torch.load(join(save_dir, "model_best.pt")))
    test(verbose=1, vis=False)

