import os
import torch
import utils
import models
import random
import argparse
import warnings
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", help="folder containing images")
parser.add_argument("--resize", "-r", type=int, default=256, help='The size of the image')
parser.add_argument("--ext", "-e", nargs='+', default=[".jpg", ".png"], help="extensions for images")
parser.add_argument("--gpu", "-g", type=int, default=-1, help="The GPU to use for the processing")
parser.add_argument("--output", "-o", default="results", help="The location for the checkpoints")
parser.add_argument("--checkpoint", "-c", help="The checkpoint to resume training on")
parser.add_argument("--direction", default="AtoB", help="The direction to use")

args = parser.parse_args()

if args.checkpoint is None:
    print("You need to specify a checkpoint to load the model from")
    exit()

checkpoint = torch.load(args.checkpoint)
parameters = checkpoint["parameters"]

if "dual" not in parameters and args.direction == "BtoA":
    print("BtoA direction requested but checkpoint has trained unidirectional DINO")
    exit()

if args.gpu >= 0:
    device = torch.device("cuda:" + str(args.gpu))
else:
    device = torch.device("cpu")

output_dir = args.output + "/" + utils.filify(parameters["name"])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model = models.EncoderDecoder(parameters["resize"], parameters["latent"], norm=parameters["norm"], noise_size=parameters["noise"],
                              dual_head_layer=None if "dual" not in parameters else parameters["dual"])

if args.direction == "AtoB":
    model.load_state_dict(checkpoint["model_p1"])
else:
    model.load_state_dict(checkpoint["model_p2"])

files = utils.list_files(args.input, exts=args.ext)
model.to(device)
model.eval()
trans = transforms.Compose([transforms.Resize((args.resize, args.resize)), transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
for f in files:
    img = trans(Image.open(f).convert("RGB")).to(device).unsqueeze(0)
    noise = None
    if model.dec.noise_size > 0:
        noise = torch.FloatTensor(1, model.dec.noise_size).normal_(0, 0.33).to(device)

    gen_img = model(img, n=noise)
    output_path = f.replace(args.input, args.output + "/")
    utils.save_img(gen_img, output_path)
