#!/usr/bin/env python
#-*- coding: utf-8 -*-



from train_MLBiNet import train


if __name__ == "__main__":
    niter = 10
    for i in range(niter):
        train(seed_id=i)
