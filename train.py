import os
import torch
import utils
import models
import random
import argparse
import warnings
import itertools
import numpy as np
import multiprocessing
from collections import deque
from dataset import MatchedImageDataset


def train_bidirectional(model, optims, imgs, noise_size, gains, lr_k, gammas, criterion, rec_factor, player_margins):
    batch_size = imgs[0].size(0)
    reconstructions = {}

    noise = [None, None]
    if noise_size > 0:
        for noise_idx in range(2):
            noise[noise_idx] = torch.FloatTensor(batch_size, noise_size).normal_(0, 0.33).to(imgs[0].device)

    # Critic Training
    optims[-1].zero_grad()

    p1_fake = model[0](imgs[0], n=noise[0])  # player 1's fake
    p2_fake = model[1](imgs[1], n=noise[1])  # player 2's fake

    p1_fake_critic_stream = model[0](imgs[0], n=noise[0], dual=True)  # player 1's fake (Critic Stream)
    p2_fake_critic_stream = model[1](imgs[1], n=noise[1], dual=True)  # player 2's fake (Critic Stream)
    p1_fake_recon = model[0](p2_fake.detach(), n=noise[0], dual=True)  # player 1's reconstruction of player 2's fake (Critic Stream)
    p2_fake_recon = model[1](p1_fake.detach(), n=noise[1], dual=True)  # player 2's reconstruction of player 1's fake (Critic Stream)

    p1_real_recon_err = criterion(p1_fake_critic_stream, imgs[1])  # Distance from p1's reconstruction of real to real image
    p1_fake_recon_err = criterion(p1_fake_recon, imgs[1])  # Distance from p1's reconstruction of p2's fake to real image
    p2_real_recon_err = criterion(p2_fake_critic_stream, imgs[0])  # Distance from p2's reconstruction of real to real image
    p2_fake_recon_err = criterion(p2_fake_recon, imgs[0])  # Distance from p2's reconstruction of p2's fake to real image

    reconstructions.update(
        {"P1 real reconstruction": p1_real_recon_err.cpu().detach().numpy(), "P2 real reconstruction": p2_real_recon_err.cpu().detach().numpy(),
         "P1 fake reconstruction": p1_fake_recon_err.cpu().detach().numpy(), "P2 fake reconstruction": p2_fake_recon_err.cpu().detach().numpy(),
         "P1 K": gains[0], "P2 K": gains[1]})

    critic_loss = p1_real_recon_err + p2_real_recon_err

    if player_margins[0] == 0 or player_margins[0] > p1_fake_recon_err:
        critic_loss += gains[0] * (player_margins[0] - p1_fake_recon_err)

    if player_margins[1] == 0 or player_margins[1] > p2_fake_recon_err:
        critic_loss += gains[1] * (player_margins[1] - p2_fake_recon_err)

    critic_loss.backward()
    optims[-1].step()

    # Balancing
    p1_balance = (p1_real_recon_err - gammas[0] * p1_fake_recon_err).cpu().detach().numpy()
    gains[0] = min(max(gains[0] + lr_k * p1_balance, 0), 1)

    p2_balance = (p2_real_recon_err - gammas[1] * p2_fake_recon_err).cpu().detach().numpy()
    gains[1] = min(max(gains[1] + lr_k * p2_balance, 0), 1)

    # Generative Training P1
    optims[0].zero_grad()

    p1_fake = model[0](imgs[0], n=noise[0])  # player 1's fake
    p2_fake_recon = model[1](p1_fake, n=noise[1], dual=True)  # player 2's reconstruction from fake (Critic stream)

    p1_gen_error = criterion(p2_fake_recon, imgs[0])  # Distance from p2's reconstruction of p1's fake to real image
    if rec_factor > 0:
        p1_gen_error += rec_factor * criterion(p1_fake, imgs[1])

    p1_gen_error.backward()
    optims[0].step()

    # Generative Training P2
    optims[1].zero_grad()

    p2_fake = model[1](images[1], n=noise[1])  # player 2's fake
    p1_fake_recon = model[0](p2_fake, n=noise[0], dual=True)  # player 1's reconstruction from fake

    p2_gen_error = criterion(p1_fake_recon, images[1])  # Distance from p1's reconstruction of p2's fake to real image
    if rec_factor > 0:
        p2_gen_error += rec_factor * criterion(p2_fake, images[0])

    p2_gen_error.backward()
    optims[1].step()

    preview_images = {"Real Domain A": imgs[0][0].cpu().detach().numpy(),
                      "Real Domain B": imgs[1][0].cpu().detach().numpy(),
                      "Translated Domain A (From Real)": p1_fake[0].cpu().detach().numpy(),
                      "Translated Domain A (From Fake)": p1_fake_recon[0].cpu().detach().numpy(),
                      "Translated Domain B (From Real)": p2_fake[0].cpu().detach().numpy(),
                      "Translated Domain B (From Fake)": p2_fake_recon[0].cpu().detach().numpy()}

    return gains, reconstructions, preview_images


def train(model, optims, imgs, noise_size, gain, lr_k, gamma, criterion, rec_factor, rec_criterion, margin):
    batch_size = imgs[0].size(0)
    reconstructions = {}

    noise = None
    if noise_size > 0:
        noise = torch.FloatTensor(batch_size, noise_size).normal_(0, 0.33).to(imgs[0].device)

    # Discriminator training
    model[1].zero_grad()
    p1_fake = model[0](imgs[0], n=noise)  # player 1's fake
    p2_fake = model[1](imgs[1])  # player 2's fake
    p2_fake_recon = model[1](p1_fake.detach())  # player 2's reconstruction from fake

    p2_real_recon_err = criterion(p2_fake, imgs[0])  # Distance from p2's reconstruction of real to real image
    p2_fake_recon_err = criterion(p2_fake_recon, imgs[0])  # Distance from p2's reconstruction of p1's fake to real image

    # Balancing
    balance = (p2_real_recon_err - gamma * p2_fake_recon_err).cpu().detach().numpy()
    reconstructions.update({"P2 real reconstruction": p2_real_recon_err.cpu().detach().numpy(),
                            "P2 fake reconstruction": p2_fake_recon_err.cpu().detach().numpy(),
                            "P2 K": gain})

    p2_loss = p2_real_recon_err
    if margin == 0 or margin > p2_fake_recon_err:
        p2_loss += gain * (margin - p2_fake_recon_err)

    p2_loss.backward()
    optims[1].step()

    # Generator training
    model[0].zero_grad()
    p1_fake = model[0](imgs[0], n=noise)  # player 1's fake
    p2_fake_recon = model[1](p1_fake)  # player 2's reconstruction of player 1's fake

    p1_fooling_error = criterion(p2_fake_recon, imgs[0])  # Distance from p2's reconstruction of p1's fake to real image

    p1_loss = p1_fooling_error + rec_factor * rec_criterion(p1_fake, images[1])

    p1_loss.backward()
    optims[0].step()

    # The controller step
    gain = min(max(gain + lr_k * balance, 0), 1)
    preview_images = {"Real Domain A": imgs[0][0].cpu().detach().numpy(),
                      "Real Domain B": imgs[1][0].cpu().detach().numpy(),
                      "Translated Domain A": p1_fake[0].cpu().detach().numpy(),
                      "Translated Domain B (From Real)": p2_fake[0].cpu().detach().numpy(),
                      "Translated Domain B (From Fake)": p2_fake_recon[0].cpu().detach().numpy()}

    return gain, reconstructions, preview_images


parser = argparse.ArgumentParser()

# Dataset arguments
parser.add_argument("--folders", "-f", nargs='+', help="Folders containing matched images for training")
parser.add_argument("--resize", "-r", type=int, default=256, help='The size of the image')
parser.add_argument("--no_aug", action='store_true', help="Does not perform data augmentation (mirroring and random cropping)")
parser.add_argument("--validation", "-v", nargs='+', help="Folders containing matched validation images")
parser.add_argument("--ext", "-e", nargs='+', help="Extensions for images")

# Training parameters
parser.add_argument("--name", "-n", default="DINO", help="The name for the experiment")
parser.add_argument("--batch", "-b", type=int, default=8, help="The batch size used in training")
parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate to use')
parser.add_argument('--rec_weight', type=float, default=1.0, help='The factor for the reconstruction loss')
parser.add_argument('--gammas', nargs='+', type=float, default=[0.7, 0.7], help='The balance for each player')
parser.add_argument('--margins', "-m", nargs='+', type=float, default=[0.2, 0.2], help='The maximum reconstruction error')
parser.add_argument('--lr_k', type=float, default=0.001, help='The learning rate of k')
parser.add_argument('--epochs', type=int, default=200, help='The number of training epochs')
parser.add_argument("--gpu", "-g", type=int, default=-1, help="The GPU to use for the processing")
parser.add_argument('--latent', "-l", type=int, default=256, help='The size for the latent space')
parser.add_argument('--noise', type=int, default=32, help='The dimension of the noise to add to the latent space')
parser.add_argument('--norm', default="instance", help='The type of normalisation to use in the networks')
parser.add_argument('--lr_decay_start', type=int, default=0, help='The epoch where the lr decay starts')
parser.add_argument("--workers", type=int, help="The number of workers to use for data loading")
parser.add_argument('--dual', type=int, help='The layer for branching the dual heads')
parser.add_argument("--checkpoint", "-c", help="The checkpoint to resume training on")

# Visualisation
parser.add_argument("--wandb", action='store_true', help="Uses weights and biases to visualize the results")
parser.add_argument("--log_every", type=int, default=200, help="The frequency with which to log data")
parser.add_argument("--no_val_imgs", type=int, default=4, help="Number of validation images to display")

# Output Parameters
parser.add_argument("--output", "-o", default="output", help="The location for the checkpoints")
parser.add_argument("--save_every", type=int, default=5, help="The frequency with which to save checkpoints")
parser.add_argument("--keep_latest", type=int, default=5, help="How many checkpoints to keep")

args = parser.parse_args()

parameters = vars(args)

checkpoint = None
if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
    parameters = checkpoint["parameters"]

if args.gpu >= 0:
    device = torch.device("cuda:" + str(args.gpu))
else:
    device = torch.device("cpu")

if args.workers is None:
    workers = max(multiprocessing.cpu_count() - 2, 1)  # Use two fewer worker than the number of cores
else:
    workers = args.workers

parameters["workers"] = workers

augmentation = not parameters["no_aug"]

dataset = MatchedImageDataset(parameters["folders"], parameters["resize"], mirror=augmentation, random_crop=augmentation, ext=parameters["ext"])
loader = torch.utils.data.DataLoader(dataset, batch_size=parameters["batch"], shuffle=True, num_workers=workers, drop_last=True, pin_memory=True)

if args.validation is not None:
    val_dataset = MatchedImageDataset(args.validation, parameters["resize"])

output_dir = args.output + "/" + utils.filify(parameters["name"])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the visualization
if args.wandb:
    import wandb

    resume_training = checkpoint is not None
    wandb.init(project=parameters["name"], config=parameters, resume=resume_training)

# The network architectures
m = []
m.append(models.EncoderDecoder(parameters["resize"], parameters["latent"], norm=parameters["norm"], noise_size=parameters["noise"],
                               dual_head_layer=parameters["dual"]).to(device))
m.append(models.EncoderDecoder(parameters["resize"], parameters["latent"], norm=parameters["norm"],
                               noise_size=0 if parameters["dual"] is None else parameters["noise"],
                               dual_head_layer=parameters["dual"]).to(device))

mse_criterion = torch.nn.MSELoss().to(device)
l1_criterion = torch.nn.L1Loss().to(device)
optimisers = []
lr_schedulers = []

if parameters["dual"] is not None:
    optimisers.append(torch.optim.Adam(m[0].dec.primary_stream_params(), lr=args.lr, betas=(0.5, 0.999)))
    optimisers.append(torch.optim.Adam(m[1].dec.primary_stream_params(), lr=args.lr, betas=(0.5, 0.999)))
    optimisers.append(torch.optim.Adam(itertools.chain(m[0].enc.parameters(), m[0].dec.secondary_stream_params(), m[1].enc.parameters(),
                                                       m[1].dec.secondary_stream_params()), lr=args.lr, betas=(0.5, 0.999)))
    for idx in range(len(optimisers)):
        lr_schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimisers[idx], lr_lambda=lambda i: (
                1 - max(i - parameters["lr_decay_start"], 0) / (args.epochs - parameters["lr_decay_start"]))))

    k = [0.0, 0.0]  # The gain for each player
else:
    for idx in range(2):
        optimisers.append(torch.optim.Adam(m[idx].parameters(), lr=args.lr, betas=(0.5, 0.999)))
        lr_schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimisers[idx], lr_lambda=lambda i: (
                1 - max(i - parameters["lr_decay_start"], 0) / (args.epochs - parameters["lr_decay_start"]))))

    k = 0.0

if parameters["margins"] is None:
    margins = [0, 0]
else:
    margins = parameters["margins"]

checkpoint_buffer = deque()

start_epoch = 0
if checkpoint is not None:
    start_epoch = checkpoint["epoch"]
    k = checkpoint["gain"]
    checkpoint_buffer = checkpoint["checkpoint_buffer"]
    for idx in range(len(m)):
        m[idx].load_state_dict(checkpoint["model_p" + str(idx + 1)])

    for idx in range(len(optimisers)):
        optimisers[idx].load_state_dict(checkpoint["optim" + str(idx)])
        lr_schedulers[idx].load_state_dict(checkpoint["scheduler" + str(idx)])

for epoch in range(start_epoch, args.epochs):
    m[0].train()
    m[1].train()
    for i, data in enumerate(loader):

        images = [d.to(device) for d in data]

        if parameters["dual"] is not None:
            k, training_info, preview_imgs = train_bidirectional(m, optimisers, images, parameters["noise"], k, parameters["lr_k"],
                                                                 parameters["gammas"], mse_criterion, parameters["rec_weight"],
                                                                 parameters["margins"])
        else:
            k, training_info, preview_imgs = train(m, optimisers, images, parameters["noise"], k, parameters["lr_k"], parameters["gammas"][0],
                                                   mse_criterion, parameters["rec_weight"], l1_criterion, parameters["margins"][0])

        if args.wandb and i % args.log_every == 0:
            training_info.update({'epoch': epoch})
            wandb.log(training_info)
            wandb.log({"Training examples": [wandb.Image(np.rollaxis(im, 0, 3), caption=label) for label, im in preview_imgs.items()]})

    if args.validation is not None:  # Validation Loop
        m[0].eval()
        m[1].eval()

        preview_idx = random.randint(0, len(val_dataset) - 1)
        total_psnr = [0, 0]
        val_preview = None
        for example_no, val_idx in enumerate(random.sample(range(len(val_dataset)), args.no_val_imgs)):
            val_imgs = val_dataset[val_idx]
            row_imgs = [val_imgs[0].detach(), val_imgs[1].detach()]

            n1 = None
            if parameters["noise"]:
                n1 = torch.FloatTensor(1, parameters["noise"]).normal_(0, 0.33).to(device)

            row_imgs.append(m[0](val_imgs[0].to(device).unsqueeze(0), n=n1).squeeze().cpu().detach())

            if parameters["dual"] is not None:
                n2 = None

                if parameters["noise"]:
                    n2 = torch.FloatTensor(1, parameters["noise"]).normal_(0, 0.33).to(device)
                row_imgs.append(m[1](val_imgs[1].to(device).unsqueeze(0), n=n2).squeeze().cpu().detach())

            row = torch.cat(row_imgs, dim=2)
            if val_preview is None:
                val_preview = row
            else:
                val_preview = torch.cat((val_preview, row), dim=1)

        if args.wandb:
            wandb.log({"Val examples": wandb.Image(val_preview, caption="Validation Set Examples")})
        elif epoch % args.save_every:
            utils.save_img(val_preview, output_dir + "/validation_results/epoch" + str(epoch) + ".png")

    if epoch % args.save_every == 0:
        save_state = {"model_p1": m[0].state_dict(), "model_p2": m[1].state_dict(), "epoch": epoch, "gain": k, "parameters": parameters,
                      "checkpoint_buffer": checkpoint_buffer}

        for idx in range(len(optimisers)):
            save_state.update({"optim" + str(idx): optimisers[idx].state_dict(), "scheduler" + str(idx): lr_schedulers[idx].state_dict()})

        torch.save(save_state, output_dir + "/checkpoints_epoch" + str(epoch) + ".pth")
        if args.keep_latest is not None:
            checkpoint_buffer.append(output_dir + "/checkpoints_epoch" + str(epoch) + ".pth")
            if len(checkpoint_buffer) > args.keep_latest:
                chkpt_to_remove = checkpoint_buffer.popleft()
                if os.path.exists(chkpt_to_remove):
                    os.remove(chkpt_to_remove)

    for idx in range(len(optimisers)):
        lr_schedulers[idx].step()
