import torch

def Ratio_Pool(x):
    b, c, h, w = x.size()
    x_1 = x.clone()
    for batch in range(b):
        for chnl in range(c):
            mean_value = x_1[batch][chnl].median()
            sum_value = x_1[batch][chnl].sum()
            for i in range(h):
                for j in range(w):
                    x_1[batch][chnl][i][j] = x_1[batch][chnl][i][j].mul(((x_1[batch][chnl][i][j] - mean_value) / sum_value))

    return x_1