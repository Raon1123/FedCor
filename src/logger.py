import datetime
import os

from torch.utils.tensorboard import SummaryWriter

def get_expstr(args, seed):
    now = datetime.datetime.now()
    exp_list = [
        now.strftime("%y%m%d_%H%M%S"),
        args.prefix,
        args.dataset,
        args.optimizer,
        str(seed)
    ]
    return "_".join(exp_list)


def get_writter(args, seed):
    logdir = './logdir'
    expstr = get_expstr(args, seed)
    log_PATH = os.path.join(logdir, expstr)

    writer = SummaryWriter(log_dir=log_PATH)
    return writer


def log_experiment(writer, train_loss, train_acc, test_acc, epoch):
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Acc/Train', train_acc, epoch)
    writer.add_scalar('Acc/Test', test_acc, epoch)