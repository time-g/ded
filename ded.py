#!/usr/bin/env python3

import sys

import numpy as np


def mahalanobis_depth(x, mean, icov):
    x_mean = x - mean
    dot = x_mean @ icov @ x_mean.T
    if dot.shape == ():
        depth = 1 / (1 + dot)
    else:
        depth = 1 / (1 + dot.diagonal())
    return depth


def ded(data):
    """
    D   = data depth(mahalanobis_depth)
    Dk  = data depth of cluster(depth of each point in a cluster)
    DM  = depth median; max(D)
    DMk = depth median of cluster; max(Dk)
    delta_k = mean(|DK - DMk|), standard deviation of cluster
    DW  = depth within cluster -> mean(delta_k)
    delta = mean(|D - DM|), standard deviation of the data
    DB  = depth between cluster, delta - DW
    DeD = depth difference, DW - DB
    """
    icov = np.linalg.pinv(np.cov(data.T))
    data_depth = mahalanobis_depth(data, data.mean(axis=0), icov)
    depth_median = np.max(data_depth)
    # standard deviation
    delta = np.abs(data_depth - depth_median).mean()
    ded_vector = []
    k_vector = []
    for k in range(2, 20):
        bord = data.shape[0] / k
        start = 0
        end = 0
        delta_k = np.zeros(k)
        for i in range(0, k):
            start = end + 1
            end = start + bord - 1
            chunk = data[int(start) : int(end)]
            icov = np.linalg.pinv(np.cov(chunk.T))
            # depth of each point in a cluster
            data_depth_k = mahalanobis_depth(chunk, np.mean(chunk, axis=0), icov)
            depth_median_k = np.max(data_depth_k)
            # std(standard deviation) of cluster
            delta_k[i] = np.abs(data_depth_k - depth_median_k).mean()

        depth_within_cluster = delta_k.mean()
        depth_between_cluster = delta - depth_within_cluster

        depth_diff = depth_within_cluster - depth_between_cluster
        ded_vector.append(depth_diff)
        k_vector.append(k)

    optimum_k = np.argmax(ded_vector) + 2

    return optimum_k


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} file.csv")
        exit(1)

    data = np.genfromtxt(sys.argv[1], delimiter="\t")
    data = data[:, [0, 1]]
    print(ded(data))
    # return data


if __name__ == "__main__":
    main()
