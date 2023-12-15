import torch
import torch.nn as nn
import numpy as np
import os
import math
import time
from data_generation import make_lego_datasets, generate_data, CharTokenizer
from models import MLP, CNN_NLP, MYMLP, TransformerEncoder
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    


def train(args, model, trainloader, optimizer, criterion, print_acc=False, writer=None):
    total_loss = 0
    correct = [0]*args.n_var
    train_var_pred = [i for i in range(args.n_train_var)] 
    total = 0
    model.train()
    for batch, labels, order in trainloader:
        batch, labels, order = batch.to(args.device), labels.to(args.device), order.to(args.device)
        x = batch
        y = labels
        inv_order = order.permute(0, 2, 1)

        optimizer.zero_grad()
        pred = model(x.float()).unsqueeze(-1)

        ordered_pred = torch.bmm(inv_order, pred[:, 0:-1:5, :]).squeeze()

        loss = 0
        for idx in train_var_pred:

            loss += criterion(ordered_pred[:, idx], y[:, idx].float()) / len(train_var_pred)
            total_loss += loss.item() / len(train_var_pred)
    
            correct[idx] += ((ordered_pred[:, idx]>0).long() == y[:, idx]).float().mean().item()
            
        total += 1
    
        loss.backward()
        optimizer.step()
    
    train_acc = [corr/total for corr in correct]
    print("   Train Loss: %f" % (total_loss/total))
    if print_acc:
        if args.epoch % 20 == 0:
            for idx in train_var_pred:
                print("     %s: %f" % (idx, train_acc[idx]))
    if writer is not None:
        writer.add_scalar("Loss/train", total_loss / (total + 1), args.epoch)
        for idx in train_var_pred:
            writer.add_scalar("Train/nb%s"%(idx), train_acc[idx], args.epoch)

    

    return train_acc

def test(args, model, testloader, criterion, writer=None):
    test_acc = []
    total_loss = 0
    correct = [0]*args.n_var
    test_var_pred = [i for i in range(args.n_var)]
    total = 0
    model.eval()
    with torch.no_grad():
        for batch, labels, order in testloader:
            batch, labels, order = batch.to(args.device), labels.to(args.device), order.to(args.device)
            x = batch
            y = labels
            inv_order = order.permute(0, 2, 1)
            pred = model(x.float()).unsqueeze(-1)

            ordered_pred = torch.bmm(inv_order, pred[:, 0:-1:5, :]).squeeze()
            for idx in test_var_pred:
                loss = criterion(ordered_pred[:, idx], y[:, idx].float())
                total_loss += loss.item() / len(test_var_pred)
                correct[idx] += ((ordered_pred[:, idx]>0).long() == y[:, idx]).float().mean().item()
                          
            total += 1
        
        test_acc = [corr/total for corr in correct]
        print("   Test  Loss: %f" % (total_loss/total))
        for idx in test_var_pred:
            print("     %s: %f" % (idx, test_acc[idx]))
    if writer is not None:
        writer.add_scalar("Loss/test", total_loss / (total + 1), args.epoch)
        for idx in test_var_pred:
            writer.add_scalar("Test/nb%s"%(idx), test_acc[idx], args.epoch)
   

    return test_acc


def main(args):
    seed_everything(args.seed)

    n_var, n_train_var = args.n_var, args.n_train_var
    n_train, n_test = args.n_train, args.n_test
    batch_size = args.batch_size

    tokenizer = CharTokenizer(args.voca_size)
    trainloader, testloader = make_lego_datasets(n_var, n_train, n_test, batch_size, args.voca_size)

    if args.model == "fcn":
        model = MLP(d_input = args.n_var * 5 * (args.voca_size + 6), d_hide=args.d_hide , d_output=args.n_var * 5, n_layers=args.n_layers, dropout=args.dropout)
    elif args.model == "cnn":
        model = CNN_NLP(
                        embed_dim=args.voca_size + 6,
                        filter_sizes=[5, 5, 5],
                        num_filters=[100, 100, 100],
                        num_classes=args.n_var * 5,
                        num_layers=args.n_layers,
                        dropout=args.dropout)
    elif args.model == "myfcn":
        model = MYMLP(d_input = args.n_var * 5 * (args.voca_size + 6), d_hide=args.d_hide , d_output=args.n_var * 5, n_layers=args.n_layers, dropout=args.dropout)
    elif args.model == "transformer":
        model = TransformerEncoder(
                        d_model=args.voca_size + 6,
                        dim_feedforward=256,
                        scaleup_dim=(args.voca_size + 6) * 4,
                        nhead=4,
                        num_layers=args.n_layers,
                        embedding_type="scaleup",
                        pos_encoder_type="learned",
                        dropout=args.dropout)
    else:
        raise NotImplementedError
    
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    # Move your model to the device
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise NotImplementedError

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    writer = SummaryWriter(log_dir=f'{args.dir}/voca_size_{args.voca_size}_n_var_{args.n_var}_n_train_var_{args.n_train_var}_n_train_{args.n_train}_n_test_{args.n_test}_batch_size_{args.batch_size}_d_hide_{args.d_hide}_n_layers_{args.n_layers}_dropout_{args.dropout}_lr_{args.lr}_T_max_{args.T_max}_epochs_{args.epochs}_optimizer_{args.optimizer}_model_{args.model}')

    for epoch in tqdm(range(args.epochs)):
        args.epoch = epoch
        start = time.time()
        print('Epoch %d, lr %f' % (epoch, optimizer.param_groups[0]['lr']))

        train(args, model, trainloader, optimizer, criterion, print_acc=True, writer=writer)
        if epoch % 20 == 0:
            test(args, model, testloader, criterion, writer=writer)
        scheduler.step()

        print('Time elapsed: %f s' %(time.time() - start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--voca_size", type=int, default=8)
    parser.add_argument("--n_var", type=int, default=8)
    parser.add_argument("--n_train_var", type=int, default=4)
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--d_hide", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--T_max", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--model", type=str, default="fcn")
    parser.add_argument("--dir", type=str, default="fcnruns")

    args = parser.parse_args()
    assert args.n_var <= args.voca_size , "var nb of a seq should be smaller than voca_size"
    main(args)