import math
import numpy as np
from numpy.linalg import inv, det


class nngmm():
    def __init__(self, k):
        self.k = k

    def Gauss_kerl(self, Data, mu, sigma, weit):
        """
        calculate distance between each point and kernel
        :param Data: one frame data
        :return: the probability
        """
        R, C = Data.shape
        proba = np.zeros([R, self.k])
        for i in range(self.k):
            temp = Data - np.tile(mu[i, :], (R, 1))

            mid_v = np.diag(np.exp(-1/2*temp.dot(inv(sigma[i,:,:])).dot(np.transpose(temp))))

            proba[:,i] = pow(2*math.pi, -C/2)*pow(det(sigma[i,:,:]),-1/2)*mid_v

        pro_w = proba*weit
        Gama_pot = pro_w/pro_w.sum(axis=1)[:,np.newaxis]

        return Gama_pot

    def Out_L(self, Data, Proba):
        """
        calculate MLE and judge whether or not network is convergence
        :param Data: one frame data
        :param Proba: calculate the probability of each point belong to every cluster
        :return: parameters need to be introduced into the next iterate progress
        """
        r,c = Data.shape
        mu = np.transpose(Proba).dot(Data)/Proba.sum(axis=0)[:,np.newaxis]

        sigma = np.zeros([self.k,c,c])
        for i in range(self.k):
            sigma_sum = np.zeros([c, c])
            #temp = Data-np.tile(Mu,(r,1))
            for j in range(r):
                sigma_sum = sigma_sum+Proba[j,i]*((Data[j,:]-mu[i,:])[:,np.newaxis]).dot(np.transpose((Data[j,:]-mu[i,:])[:,np.newaxis]))
            sigma[i,:,:] = sigma_sum/sum(Proba[:,i])

        weit = Proba.sum(axis=0)/r

        return mu, sigma, weit