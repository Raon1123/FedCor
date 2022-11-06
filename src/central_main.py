import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm

from utils import get_dataset, get_model, get_device, setup_seed
from options import args_parser

def main(args, seed):
    # load dataset and user groups
    train_dataset, test_dataset, _, _, _ = get_dataset(args,seed)
    device = get_device(args)
    if seed is not None:
        setup_seed(seed)
    
    # BUILD MODEL
    data_size = train_dataset[0][0].shape
    global_model = get_model(args, data_size)

    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    epochs = args.epochs * args.local_ep
    test_accuracy_list = []

    train_loader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=0)

    np = global_model.parameters()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(np, lr=args.lr,
                                    momentum=args.momentum,weight_decay=args.reg)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(np, lr=args.lr,
                                        weight_decay=args.reg)

    bar = tqdm(range(epochs))
    for epoch in bar:
        run_trainepoch(global_model, train_loader, optimizer, device)
        acc, _ = run_testepoch(global_model, test_loader, device)
        test_accuracy_list.append(acc)
        bar.set_postfix_str(str(acc))

    print(test_accuracy_list)


def ce_criterion(pred, target, *args):
    ce_loss = F.cross_entropy(pred, target)
    return ce_loss, float(ce_loss)


def run_trainepoch(model, train_dataloader, optimizer, device="cpu"):
    criterion = ce_criterion
    for (datas, labels) in train_dataloader:
        datas, labels = datas.to(device), labels.to(device)

        model.zero_grad()
        outputs = model(datas)
        total_loss, celoss = criterion(outputs, labels)
        total_loss.backward()
        optimizer.step()


def run_testepoch(model, test_dataloader, device="cpu"):
    criterion = ce_criterion
    loss, total, correct = 0.0, 0.0, 0.0

    model.eval()
    
    for batch_idx, (datas, labels) in enumerate(test_dataloader):
        datas, labels = datas.to(device), labels.to(device)
        outputs = model(datas)
        batch_loss, _ = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss/(batch_idx+1)


if __name__=="__main__":
    args = args_parser()
    for seed in args.seed:
        main(args, seed)