import argparse
import itertools
import math
import os
import time

import torch

from data_utils import FT_Dataset
from dataloader import DataLoader
from model import GPT2Config, GPT2LMModel
from optimizier import create_adam_optimizer_from_args, add_optimizer_params, create_optimizer_scheduler

parser = argparse.ArgumentParser(description='PyTorch GPT2 ft script')
parser.add_argument('--train_data', required=True, help='location of training data corpus')
parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--valid_data', required=True, help='location of validation data corpus')
parser.add_argument('--valid_batch_size', type=int, default=4, help='validation batch size')
parser.add_argument('--model_card', default='gpt2.md', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'], help='model names')
parser.add_argument('--max_epoch', type=int, default=None, help='max epoch of training')
parser.add_argument('--label_smooth', default=0.0, type=float, help='label smoothing')
parser.add_argument("--device", help='device')
parser.add_argument('--log_interval', type=int, default=100, help='log interval')
parser.add_argument('--save_interval', type=int, default=500, help='save interval')
parser.add_argument('--work_dir', type=str, default='./model', help='working folder.')
parser.add_argument('--eval_interval', type=int, default=2000, help='eval interval')
add_optimizer_params(parser)


class AverageMeter(object):
    """Computes and stores the average and current value
         Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def optimizer_step(_loss, _optimizer, _model, _schedule):
    _loss.backward()

    _optimizer.step()
    _optimizer.zero_grad()

    _schedule.step()


def evaluate(model, valid_loader, args):
    model.eval()
    total_loss = 0.
    start_time = time.time()

    avg_lm_loss = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value for key, value in data.items()}

            _input = data['input'].to(args.device)
            _target = data['target'].to(args.device)
            _msk = data['mask'].to(args.device)

            _lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk)
            loss = _loss.mean()

            avg_lm_loss.update(loss.item())

            if idx % 100 == 0:
                print('eval samples:', idx, 'loss:', loss.float())

        total_time = time.time() - start_time
        print('average loss', avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)

def train_validate(
        model,
        optimizer,
        scheduler,
        train_loader,
        args,
        train_step,
        epoch
):
    model.train()
    avg_lm_loss = AverageMeter()
    print('start to train the model................', epoch)
    log_start_time = time.time()
    best_val_ppl = None

    for idx, data in enumerate(train_loader):
        data = {key: value for key, value in data.items()}

        _input = data['input'].to(args.device)
        _target = data['target'].to(args.device)
        _msk = data['mask'].to(args.device)

        _, _lm_loss = model(
            _input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth
        )

        _lm_loss = _lm_loss.mean()

        train_step += 1
        avg_lm_loss.update(_lm_loss.item())
        optimizer_step(_lm_loss, optimizer, model, scheduler)

        if train_step % args.log_interval == 0:
            elapsed = time.time() - log_start_time
            lr = optimizer.param_groups[0]['lr']
            log_str = f'| epoch {epoch:3d} step {train_step:>8d} | {idx + 1:>6d} batches | ' \
                      f'lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | ' \
                      f'loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | ' \
                      f'ppl {math.exp(avg_lm_loss.avg):5.2f}'

            print(log_str)
            log_start_time = time.time()
            avg_lm_loss.reset()

        if train_step % args.save_interval == 0:
            model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
            if not os.path.exists(args.work_dir):
                os.makedirs(args.work_dir)
            my_state_dict = model.state_dict()
            model_state_dict = {k: my_state_dict[k] for k in my_state_dict}
            torch.save({'model_state_dict': model_state_dict}, model_path)

        if train_step % args.eval_interval == 0:
            eval_start_time = time.time()

            valid_loss, valid_ppl = evaluate(model, valid_loader, args)

            if best_val_ppl is None or valid_ppl < best_val_ppl:
                best_val_ppl = valid_ppl

            log_str = f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | ' \
                      f'time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | ' \
                      f'valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} '
            print('-' * 100)
            print(log_str)
            print('-' * 100)

            model.train()

        if train_step == args.max_step:
            break

    return train_step


if __name__ == '__main__':
    args = parser.parse_args()

    train_data = FT_Dataset(
        ft_file=args.train_data,
        batch_size=args.train_batch_size,
        max_seq_length=512
    )

    train_loader = DataLoader(
        train_data, batch_size=args.train_batch_size, shuffle=False
    )

    valid_data = FT_Dataset(
        ft_file = args.valid_data,
        batch_size=args.valid_batch_size,
        max_seq_length=512
    )

    valid_loader = DataLoader(
        valid_data, batch_size=args.valid_batch_size, shuffle=False
    )

    if args.model_card == 'gpt2.sm':
        config = GPT2Config(n_embd=768, n_layer=12, n_head=12)
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(n_embd=1024, n_layer=24, n_head=16)
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(n_embd=1280, n_layer=36, n_head=20)

    lm_net = GPT2LMModel(config)

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        lm_net = lm_net.cuda()

    # if args.lora_dim > 0:
    #       lora.mark_only_lora_as_trainable(lm_net)

    optimizer = create_adam_optimizer_from_args(lm_net, args)

    args.max_step = (args.max_epoch * train_data.num_batches)

    scheduler = create_optimizer_scheduler(optimizer, args)

    try:
        train_step = 0
        for epoch in itertools.count(start=1):
            train_step = train_validate(
                lm_net,
                optimizer,
                scheduler,
                train_loader,
                args,
                train_step=train_step,
                epoch=epoch
            )

            if train_step >= args.max_step:
                print('-' * 100)
                print('End of training')
                break
    except KeyboardInterrupt:
        print('-' * 100)
        print('Exiting from training early')
