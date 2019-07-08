import os
import numpy as np
from Gmm_nn import nngmm
import matplotlib.pyplot as plt
from scipy.io import loadmat


def Slice_oneframe(Data):
    """
    frome the source data slice one frame
    :param Data: N*4 source data
    :return: ONE FRAME DATA
    """
    Rssi = Data[0, 3]
    Data_frame = []
    j = 0
    for i in range(len(Data[:, 0])):
        if Data[i, 3] == Rssi:
            if j == 200:
                Data_frame.append(Data[i, 1:3])
        else:
            Rssi = Data[i, 3]
            j = j + 1
            if j > 201:
                break

    return np.array(Data_frame)

def main():
    """
    Classify a two dimensional data with reconstructed GMM
    :return: 0
    """
    diroot = r"D:\Users\dell\AppData\Roaming\SPB_Data\RBFNN\Matlab_File"

    Data_r = loadmat(os.path.join(diroot, 'direct02_16-09-07_0718_001.mat'))
    data = Data_r['x']
    Dataframe = Slice_oneframe(data)
    Data = Dataframe
    Row, Colum = Dataframe.shape
    k = 8
    epoch = 0

    # initial parameters mu sigma and phi
    sigma = np.zeros([k,Colum,Colum])
    phi = np.zeros(k)

    np.random.shuffle(Dataframe)
    mu = Dataframe[0:k,:]

    for i in range(k):
        phi[i] = 1/k
        sigma[i,:,:] = np.cov(np.transpose(Data))

    mu_p = mu
    sigma_p = sigma
    phi_p = phi

    Net = nngmm(k)

    while True:

        Hid_lay = Net.Gauss_kerl(Data, mu, sigma, phi)
        mu, sigma, phi = Net.Out_L(Data, Hid_lay)
        mu_s = abs(mu-mu_p).sum(axis=1).sum(axis=0)
        sigma_s = abs(sigma-sigma_p).sum(axis=2).sum(axis=1).sum(axis=0)
        phi_s = abs(phi-phi_p).sum(axis=0)

        para = mu_s+sigma_s+phi_s
        if para<0.0001:
            print('mu:\n', mu)
            print('sigma\n',sigma)
            print('weit\n',phi)

            fig = plt.figure(1)
            plt.scatter(Data[:,0],Data[:,1])
            plt.scatter(mu[0, 0], mu[0, 1], c='r')
            plt.scatter(mu[1, 0], mu[1, 1], c='g')
            plt.scatter(mu[2, 0], mu[2, 1], c='b')
            plt.scatter(mu[3, 0], mu[3, 1], c='y')
            plt.scatter(mu[4, 0], mu[4, 1], c='w')
            fig.show()
            break

        mu_p = mu
        sigma_p = sigma
        phi_p = phi
        epoch = epoch + 1
        if epoch % 10 == 0:
            print('epoch:{:d}, the change quantity of parameter{:f}'.format(epoch, para))


if __name__ == "__main__":
    main()
