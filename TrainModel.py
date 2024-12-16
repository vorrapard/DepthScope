import argparse
import time
import datetime
from os import path, listdir

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
import torchvision.utils as vision_utils
from tensorboardX import SummaryWriter

from data import loadZipToMem, getTrainingTestingData
from loss import ssim
from utils import AverageMeter, DepthNorm, init_or_load_model

def main() -> None:
    # Command line arguments
    parser = argparse.ArgumentParser(description='Monocular Depth Estimation')
    parser.add_argument(
        '--epochs',
        default=5,
        type=int,
        help='Number of epochs to run for training'
        )
    parser.add_argument(
        '--lr',
        default=0.0001,
        type=float,
        help='Initial learning rate'
        )
    parser.add_argument(
        '--batch',
        default=8,
        type=int,
        help='Batch size'
        )
    parser.add_argument(
        '--checkpoint',
        default='',
        type=str,
        help='Directory or path to last saved checkpoint from which to resume training'
        )
    parser.add_argument(
        '--device',
        default='cuda',
        type=str,
        help='Device to run training on'
    )
    parser.add_argument(
        '--enc_pretrain',
        default=True,
        type=bool,
        help='Use pretrained encoder'
    )
    parser.add_argument(
        '--data',
        default='',
        type=str,
        help='Path to image dataset'
    )
    parser.add_argument(
        '--save',
        default='',
        type=str,
        help='Directory to save checkpoints in'
    )
    parser.add_argument(
        '--model',
        default='DenseDepth',
        type=str,
        help='Name of model to train'
    )
    args = parser.parse_args()

    # Set up various constants
    prefix = args.model.strip() + '_'
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')

    # Load data
    print('Loading data...')
    data = loadZipToMem(args.data)
    print('Data loaders ready!')

    # Load from checkpoint if given
    try:
        if path.isfile(args.checkpoint):
            ckpt = args.checkpoint
        elif path.isdir(args.checkpoint):
            ckpt = sorted([path.join(args.checkpoint, f) for f in listdir(args.checkpoint) if prefix in f])[-1]
        else:
            ckpt = None
    except:
        ckpt = None
    if ckpt:
        print('Loading from checkpoint...')
        model, optimizer, scheduler, start_epoch = init_or_load_model(
            depthmodel=args.model,
            enc_pretrain=args.enc_pretrain,
            lr=args.lr,
            ckpt=ckpt,
            device=device
        )
        print(f'Resuming from epoch #{start_epoch}')
    # Initialize new model if no checkpoint present
    else:
        print('Initializing new model...')
        model, optimizer, scheduler, start_epoch = init_or_load_model(
            depthmodel=args.model,
            enc_pretrain=args.enc_pretrain,
            lr=args.lr,
            ckpt=None,
            device=device
        )

    # Set up logging
    writer = SummaryWriter(
        comment=f'{prefix}-learning_rate={args.lr}-epoch={args.epochs}-batch_size={args.batch}'
    )

    # Loss functions
    l1_criterion = nn.L1Loss()

    # Start training
    print(f'Device: {device}')
    print('Starting training...')

    losses_total = AverageMeter()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # Set up averaging
        batch_time = AverageMeter()
        losses = AverageMeter()

        # Switch to train mode
        model.train()
        model = model.to(device)
        epoch_start = time.time()
        end = time.time()

        trainloader, testloader = getTrainingTestingData(data, batch_size=args.batch)
        num_trainloader = len(trainloader)
        num_testloader = len(testloader)

        for idx, batch in enumerate(trainloader):
            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.Tensor(batch['image']).to(device)
            depth = torch.Tensor(batch['depth']).to(device)

            # Normalize depth
            normalized_depth = DepthNorm(depth)

            # Predict
            output = model(image)

            # Compute loss
            l1_loss = l1_criterion(output, normalized_depth)
            loss_temp, _ = ssim(output, normalized_depth, 1000.0 / 10.0)
            ssim_loss = torch.clamp(
                (1 - loss_temp) * 0.5,
                min=0,
                max=1
            )
            loss = (1.0 * ssim_loss) + (0.1 * l1_loss)

            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(num_trainloader-idx))))

            # Log progress
            num_iters = epoch * num_trainloader + idx
            if idx % 5 == 0:
                # Print status to console
                print(
                    f'Epoch: #{epoch} Batch: {idx}/{num_trainloader}\t'
                    f'Time (current/total) {batch_time.val:.3f}/{batch_time.sum:.3f}\t'
                    f'eta {eta}\t'
                    f'Loss (current/average) {losses.val:.4f}/{losses.avg:.4f}\t'
                )
                writer.add_scalar('Train/Loss', losses.val, num_iters)

            # Delete resources
            del image
            del depth
            del output

        # Switch to validation mode
        model.eval()

        for idx, batch in enumerate(testloader):
            # Prepare sample and target
            image = torch.Tensor(batch['image']).to(device)
            depth = torch.Tensor(batch['depth']).to(device)

            # Normalize depth
            normalized_depth = DepthNorm(depth)

            # Predict
            output = model(image)

            # Compute loss
            l1_loss = l1_criterion(output, normalized_depth)
            loss_temp, _ = ssim(output, normalized_depth, 1000.0 / 10.0)
            ssim_loss = torch.clamp(
                (1 - loss_temp) * 0.5,
                min=0,
                max=1
            )
            loss = (1.0 * ssim_loss) + (0.1 * l1_loss)

            # Update step
            losses_total.update(loss.data.item(), image.size(0))

        # Step the scheduler
        scheduler.step()

        print(
            '----------------------------------\n'
            f'Epoch: #{epoch}, Avg. Net Loss: {losses_total.avg:.4f}\n'
            '----------------------------------'
        )

        # Save checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'loss': losses.avg
                },
                path.join(args.save, f'{prefix}ckpt_{epoch}_{int(losses.avg * 100)}.pth')
            )
            writer.add_scalar('Train/Loss.avg', losses.avg, epoch)

if __name__ == '__main__':
    main()
