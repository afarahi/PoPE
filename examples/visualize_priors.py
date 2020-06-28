import pymc3 as pm
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import os

def visualize_priors():

    plot_LKJ_prior(n_samples=2000)

    marten_examples()


def sample_LKJ_prior(nu=2, shape=2, n_samples=200000):
    """
    Sample LKJ prior

    Parameters
    ----------
    nu : float,
        LKJ prior \nu parameter

    shape : int
        dimensionality of the covariance matrix.

    mcmc_samples : int
        Number of samples drawn from the prior.

    Returns
    -------
    r: numpy-array, shape (n_sample, )
        MCMC samples.
    """

    with pm.Model() as model_correlation:
        # generate a sample of
        sd_dist = pm.Gamma.dist(alpha=2, beta=1, shape=2)
        chol_packed = pm.LKJCholeskyCov('chol_packed', n=shape, eta=nu, sd_dist=sd_dist)
        chol = pm.expand_packed_triangular(shape, chol_packed)
        vals = pm.MvNormal('true_quantities', mu=0.0, chol=chol, shape=(1, shape))

    with model_correlation:
        # Use elliptical slice sampling
        trace = pm.sample(n_samples, chains=2)

    r = []
    for chol_p in zip(trace['chol_packed'][:]):
        cov = make_cov_mtx_from_chol_vec(chol_p, ndim=shape)
        r += [cov[1, 0] / np.sqrt(cov[0, 0] * cov[1, 1])]

    return r


def plot_LKJ_prior(n_samples=2000):

    import matplotlib.pylab as plt

    r1 = sample_LKJ_prior(nu=1, shape=2, n_samples=n_samples)
    r2 = sample_LKJ_prior(nu=2, shape=2, n_samples=n_samples)
    r3 = sample_LKJ_prior(nu=4, shape=2, n_samples=n_samples)

    df = pd.DataFrame({'r1': r1, 'r2': r2, 'r3': r3})

    plt.hist(df.r3, range=(-1, 1), bins=50, color='steelblue', density=True, label=r'$\nu = 4$')
    plt.hist(df.r2, range=(-1, 1), bins=50, color='darkorchid', alpha=0.8, density=True, label=r'$\nu = 2$')
    plt.hist(df.r1, range=(-1, 1), bins=50, histtype='step', color='black', lw=6.0, density=True, label=r'$\nu = 1$')
    plt.legend(loc=1, prop={'size': 16})
    plt.xlabel('correlation', size=20)
    plt.ylabel('PDF', size=20)
    plt.grid()
    plt.xlim([-1.0, 1.0])

    check_directory('./plots/')
    plt.savefig('./plots/LKJ_prior_corr.pdf', bbox_inches='tight')
    plt.close()


def marten_examples():

    def marten_prior(X_star, l = 1.0):

        with pm.Model() as latent_gp_model:
            cov_func = pm.gp.cov.Matern52(1, l)

            # Specify the GP.  The default mean function is `Zero`.
            gp = pm.gp.Latent(cov_func=cov_func)

            # Place a GP prior over the function f.
            f = gp.prior("f", X=X_star)

        with latent_gp_model:
            f_star = gp.conditional("f_star", X_star)

        # A mean function that is zero everywhere
        mean_func = pm.gp.mean.Zero()

        # The latent function values are one sample from a multivariate normal
        # Note that we have to call `eval()` because PyMC3 built on top of Theano
        f_true = np.random.multivariate_normal(mean_func(X_star).eval(), cov_func(X_star).eval(), 1).flatten()

        return f_true

    n = 500  # The number of data points
    X = np.linspace(0, 2, n)[:, None]

    fig = plt.figure(figsize=(28, 10))

    ax = plt.subplot(2, 2, 1)
    for i in range(6):
        y = marten_prior(X, l=0.05)
        ax.plot(X, y, lw=4)
        plt.title(r'$l = 0.05$', size=30)
    ax.set_ylabel("y", size=30)
    plt.xticks([0, 0.5, 1.0, 1.5, 2.0], 5 * [' '])
    plt.yticks([-2, 0, 2])
    plt.xlim([0, 2])
    plt.ylim([-3, 3])
    plt.grid()

    ax = plt.subplot(2, 2, 2)
    for i in range(6):
        y = marten_prior(X, l=0.1)
        ax.plot(X, y, lw=4)
        plt.title(r'$l = 0.1$', size=30)
    plt.xticks([0, 0.5, 1.0, 1.5, 2.0], 5 * [' '])
    plt.yticks([-2, 0, 2],  3 * [' '])
    plt.xlim([0, 2])
    plt.ylim([-3, 3])
    plt.grid()

    ax = plt.subplot(2, 2, 3)
    for i in range(6):
        y = marten_prior(X, l=0.5)
        ax.plot(X, y, lw=4)
        plt.title(r'$l = 0.5$', size=30)
    ax.set_ylabel("y", size=30)
    ax.set_xlabel("X", size=30)
    plt.xticks([0, 0.5, 1.0, 1.5, 2.0], ['0', '0.5', '1', '1.5', '2'])
    plt.yticks([-2, 0, 2])
    plt.xlim([0, 2])
    plt.ylim([-3, 3])
    plt.grid()

    ax = plt.subplot(2, 2, 4)
    for i in range(6):
        y = marten_prior(X, l=1.0)
        ax.plot(X, y, lw=4)
        plt.title(r'$l = 1$', size=30)
    ax.set_xlabel("X", size=30)
    plt.xticks([0, 0.5, 1.0, 1.5, 2.0], ['0', '0.5', '1', '1.5', '2'])
    plt.yticks([-2, 0, 2], 3*[''])
    plt.xlim([0, 2])
    plt.ylim([-3, 3])
    plt.grid()

    plt.subplots_adjust(wspace=0.03)

    check_directory('./plots/')
    plt.savefig('./plots/example_gp_prior.pdf', bbox_inches='tight')

def check_directory(dir):
    """
    Check if it already exists using os.path.exists(directory). If not, create one.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)