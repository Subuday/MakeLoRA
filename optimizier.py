import argparse

import torch

from data_utils import FT_Dataset
from dataloader import DataLoader
from model import GPT2Config, GPT2LMModel
from torch.optim.lr_scheduler import LambdaLR


def add_optimizer_params(parser: argparse.ArgumentParser):
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay rate')
    parser.add_argument('--adam_epislon', default=1e-6, type=float, help='adam epsilon')
    parser.add_argument('--adam_beta1', default=0.9, type=float, help='adam beta1 term')
    parser.add_argument('--adam_beta2', default=0.98, type=float, help='adam beta2 term')
    parser.add_argument('--warmup_step', type=int, default=0, help='upper epoch limit')


def create_adam_optimizer_from_args(model, args):
    optimizer = torch.optim.AdamW(
        create_grouped_parameters(model),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epislon,
        weight_decay=args.weight_decay
    )
    return optimizer


def create_grouped_parameters(model):
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()],  # if not any(nd in n for nd in no_decay)],
        }
    ]
    return optimizer_grouped_parameters


def create_optimizer_scheduler(optimizer, args):
    return get_linear_schedule_with_warmup(optimizer, args.warmup_step, args.max_step, last_epoch=-1)


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    last_epoch
):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch GPT2 ft script')
    add_optimizer_params(parser)
    args = parser.parse_args()

    train_data = FT_Dataset("./data/e2e/train.jsonl", batch_size=8, max_seq_length=512)
    dataloader = DataLoader(dataset=train_data, batch_size=8)
    batch = next(iter(dataloader))
    config = GPT2Config(n_embd=768, n_layer=12, n_head=12)
    model = GPT2LMModel(config=config)

    optimizer = create_adam_optimizer_from_args(model, args)
