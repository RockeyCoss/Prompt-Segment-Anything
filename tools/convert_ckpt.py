import torch
import argparse

parer = argparse.ArgumentParser()
parer.add_argument('source_file')
parer.add_argument('des_file')
args = parer.parse_args()

ckpt = torch.load(args.source_file, map_location='cpu')
ckpt = ckpt['model']
torch.save(ckpt, args.des_file)