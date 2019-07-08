____________________________________
the detail of E/M algorithm


1、 Initialize the mean \mu_k, 
    the covariance matrix \Sigma_k and 
    the mixing coefficients \pi_k 
    by some random values. (or other values)

2、 Compute the \gamma_k values for all k.

3、 Again Estimate all the parameters 
    using the current \gamma_k values.

4、 Compute log-likelihood function.

5、 Put some convergence criterion

6、 If the log-likelihood value converges to some value 
    ( or if all the parameters converge to some values ) 
    then stop, 
    else return to Step2


file Gmm_nn.py and Gmm_rebuild.py is independent file that using neural network reconstruuct GMM
