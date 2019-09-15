import torch


def fusion(wc, wp, wq, xc, xp, xq):
    return wc.mul(xc) + wp.mul(xp) + wq.mul(xq)


def main():
    pass
