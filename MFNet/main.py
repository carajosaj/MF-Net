import os
import json
import argparse
import datetime
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from KFNet.loss.loss import LabelSmoothingCrossEntropy
from dataloader import load_data
from engine import train_one_epoch, evaluate
# from model import convnext_tiny as create_model
from KFNet.model.KFNet import KFNet as create_model
from optim_factory import create_optimizer
import utils

def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='Filter configs to train')
    parser.add_argument('--num_classes', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=4e-3)


    parser.add_argument('--data_path', type=str,
                        default="/home/junnki/dataset/500fish/500fish")
    parser.add_argument('--data_json', type=str,
                        default="./Fish_data.json")

    parser.add_argument('--project', default='KFNet', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--log_dir', default='./',
                        help='path where to tensorboard log')
    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--enable_wandb', type=str2bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--save_ckpt_num', default=3, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=True)

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()
    return args

def main(args):


    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    global_rank = utils.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0 and args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    train_loader, val_loader, num_training_steps_per_epoch, val_num = load_data(args.data_path, args.data_json,
                                                                                img_size=224,
                                                                                batch_size=args.batch_size)

    model = create_model(num_classes=args.num_classes).to(device)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    # params = [p for p in model.parameters() if p.requires_grad]
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = create_optimizer(args, model)
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, 1e-6, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=20)
    wd_schedule_values = utils.cosine_scheduler(
        0.05, 0.05, args.epochs, num_training_steps_per_epoch)
    print(wd_schedule_values)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model_without_ddp=model,
        optimizer=optimizer)

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(val_loader, model, device)
        print(f"Accuracy of the network on {str(val_num)} test images: {test_stats['acc1']:.5f}%")
        return

    max_accuracy = 0.
    criterion = LabelSmoothingCrossEntropy(0.1)
    start_time = time.time()
    for epoch in range(args.epochs):
        # train
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch,
            log_writer=log_writer, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch)

        if (epoch + 1) % args.save_ckpt_num == 0 or epoch + 1 == args.epochs:
            utils.save_model(
                args=args, model_without_ddp=model, optimizer=optimizer, epoch=epoch)
        if val_loader is not None:
            test_stats = evaluate(val_loader, model, device)
            print(f"Accuracy of the model on the {str(val_num)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir:
                    utils.save_model(
                            args=args, model_without_ddp=model, optimizer=optimizer, epoch="best")
                    print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': params}

        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': params}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        else:
            print("log.txt is not saved")

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

        remaining_time = time.time() - start_time
        remaining_time_str = str(datetime.timedelta(seconds=int(remaining_time / (epoch + 1) * (args.epochs - epoch - 1))))
        if epoch != 179:
            print('Remaining time {}'.format(remaining_time_str))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    args = parse_args()
    main(args)
