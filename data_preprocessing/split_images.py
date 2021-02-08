from PIL import Image
import dtk.filesystem as fs
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--input", "-i", nargs='?', help="folders containing matched images")
parser.add_argument("--output", "-o", nargs='+', help="folders containing matched output files")

args = parser.parse_args()

files = fs.list_files(args.input)

for folder in args.output:
    if not os.path.exists(folder):
        os.makedirs(folder)

for f in files:
    filename = os.path.basename(f)
    im = Image.open(f)
    im_a = im.crop((0, 0, im.width//2, im.height))
    im_b = im.crop((im.width//2,0 ,im.width , im.height))
    im_a.save(args.output[0]+"/"+filename)
    im_b.save(args.output[1]+"/"+filename)
