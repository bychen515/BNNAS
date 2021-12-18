import os
import torch
import pickle



def main():
    info = torch.load('log/ea_results.pth.tar')['vis_dict']
    cands = sorted([cand for cand in info if 'err' in info[cand]],
                   key=lambda cand: info[cand]['err'])[:10]

    for cand in cands:
        print(cand, info[cand]['err'])

if __name__ == '__main__':
    main()