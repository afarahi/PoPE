from PoPE.profile_estimator import estimate_mean_property_profile, estimate_property_covariance
from PoPE.plotting_tools import plot_mean_profile_fit
from PoPE.profile_estimator import differentiate_mean_profile

import pymc3 as pm
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import os

import matplotlib as mpl

mpl.rcParams['xtick.direction'], mpl.rcParams['ytick.direction'] = 'in', 'in'
mpl.rcParams['xtick.major.size'], mpl.rcParams['xtick.minor.size'] = 14, 8
mpl.rcParams['xtick.major.width'], mpl.rcParams['xtick.minor.width'] = 1.0, 0.8
mpl.rcParams['xtick.major.pad'], mpl.rcParams['xtick.minor.pad'] = 10, 10
mpl.rcParams['ytick.major.size'], mpl.rcParams['ytick.minor.size'] = 14, 8
mpl.rcParams['ytick.major.width'], mpl.rcParams['ytick.minor.width'] = 1.0, 0.8
mpl.rcParams['xtick.labelsize'], mpl.rcParams['ytick.labelsize'] = 16, 16
default_cmap = plt.cm.coolwarm


def estimate_TNG_profiles():

    # load data and add measurement noise
    Xs, Ys1, Ys2, Yobs1, Yobs2, Ys1err, Ys2err = load_data(n=3, min_SNR=1.0, max_SNR=5.0)

    # compute the average profile
    mp, gp, model = estimate_mean_property_profile(Xs.T, Yobs1, Yobs2, Ys1err, Ys2err, verbose=False,
                                                   Xu_shapes=[20, 7], kernel_scales=[2.0, 2.0])

    # visualize the average profile
    plot_mean_relation_gp(mp, gp, Xs, Ys1, Ys2)

    # visualize the logarithmic derivative the average profile
    plot_differential_gp(mp, gp, model)

    run_correlation_model(gp, mp, Xs.T, Yobs1, Yobs2, Ys1err, Ys2err, Ys1, Ys2, mcmc_samples=500)


def load_data(n=5, i=1, min_SNR=1.0, max_SNR=5.0):

    def add_noise(Ys, min_SNR=4.0, max_SNR=5.0):

        n_sample = len(Ys)

        inv_min_SNR = 1.0 / min_SNR
        inv_max_SNR = 1.0 / max_SNR

        Yerr = (inv_min_SNR + (inv_max_SNR - inv_min_SNR) * np.random.random(n_sample))

        Yobs = np.random.normal(Ys, Yerr)

        return Yobs, Yerr

    df = pd.read_csv('./data/TNG100_densities.csv')

    cols = df.columns
    for icol in cols[1:]:
        if icol in ['fstars', 'fgas', 'fb', 'qdm', 'qgas', 'sdm', 'sgas']: continue
        df[icol] = df[icol].apply(np.log10)

    df = df[df['x'] < 0.7]
    df = df[df['x'] < 0.7]
    df = df[df['x'] > -1.1]
    df = df[df['M200'] < 14.0]

    unique_ids = np.unique(df.HaloID)
    new_index = range(len(unique_ids))
    df_index = pd.DataFrame({'index': new_index, 'HaloID': unique_ids})
    df = df.join(df_index.set_index('HaloID'), on='HaloID')

    r = np.array(df.x)
    y1 = np.array(df.mstars_Msun)
    y2 = np.array(df.mgas_Msun)
    m200 = np.array(df.M200)

    mask = y1 > 0.0
    mask *= y2 > 0.0
    mask *= r > -1.1
    mask *= r < 0.6

    Xs = np.array([r[mask][i::n], m200[mask][i::n]])
    Ys1 = y1[mask][i::n] - 10.0
    Ys2 = y2[mask][i::n] - 12.0

    print("Halos sample size : ", len(Ys1))

    Yobs1, Ys1err = add_noise(Ys1, min_SNR=min_SNR, max_SNR=max_SNR)
    Yobs2, Ys2err = add_noise(Ys2, min_SNR=min_SNR, max_SNR=max_SNR)

    plt.hist(1.0 / np.sqrt((np.exp(Ys1err ** 2) - 1.0)), range=(min_SNR, max_SNR), color='lightsteelblue',
             bins=40, alpha=1.0, label=r'$\log_{10}(\rho_{\star, \rm obs})$')
    plt.hist(1.0 / np.sqrt((np.exp(Ys2err ** 2) - 1.0)), range=(min_SNR, max_SNR), histtype='step', color='lightcoral',
             lw=5.0, bins=40, alpha=1.0, label=r'$\log_{10}(\rho_{\rm gas, obs})$')

    plt.xlim([min_SNR, max_SNR])
    plt.legend(loc=1, prop={'size':18})
    plt.grid()
    plt.xlabel('SNR', size=22)
    plt.ylabel('PDF', size=22)

    check_directory('./plots/')
    plt.savefig('./plots/SNR_TNG.png', bbox_inches='tight')
    plt.close()

    return Xs, Ys1, Ys2, Yobs1, Yobs2, Ys1err, Ys2err


def plot_mean_relation_gp(mp, gp, Xs, Ys1, Ys2):

    from examples.kllr import Plot_Fit_Split

    df = pd.DataFrame({'x':Xs[0], 'm':Xs[1], 'stars':Ys1, 'gas':Ys2})

    x1 = np.linspace(-1.0, 0.5, 201)
    x2 = np.array([12.25, 12.75, 13.5])
    X_new = pm.math.cartesian(x1[:, None], x2[:, None])

    plt.figure(figsize=(12, 10))

    ax = plt.subplot(2, 2, 1)
    plt.scatter(Xs[0], Ys1, s=2, color='lightsteelblue', alpha=0.5)
    plot_mean_profile_fit(X_new, mp, gp[0], x2_dim_len=3, ax=ax, color=['darkmagenta', 'black', 'darkgreen'],
                          label=[r'$\log_{10}(M_{200}) = 12.25$', r'$\log_{10}(M_{200}) = 12.75$',
                                 r'$\log_{10}(M_{200}) = 13.5$'])
    plt.xlim([-1.0, 0.5])
    plt.ylim([-3.0, 3.0])
    plt.xticks([-1, -0.5, 0.0, 0.5], 4*[' '])
    plt.ylabel(r'$\log(\rho_{\rm \star} / 10^{10})$', size=22)
    plt.title('GP Fits', size=20)
    plt.grid()
    plt.legend(loc=3, prop={'size': 12})

    ax = plt.subplot(2, 2, 2)
    plt.scatter(Xs[0], Ys1, s=2, color='lightsteelblue', alpha=0.5)
    ax = Plot_Fit_Split(df, 'x', 'stars', 'm', split_bins=[12, 12.5, 13, 14], show_data=False,
                        nbins=25, xrange=[-1.2, 0.6], split_mode='Data',
                        kernel_type='gaussian', kernel_width=0.15, xlog=False,
                        labels=[r' ', r' ', r'$\log_{10}(M_{200})$'],
                        color=['darkmagenta', 'black', 'darkgreen'], ax=ax)
    plt.legend(loc=3, prop={'size': 12})
    plt.xlim([-1.0, 0.5])
    plt.ylim([-3.0, 3.0])
    plt.xticks([-1, -0.5, 0.0, 0.5], 4*[' '])
    plt.yticks([-3, -2, -1, 0, 1, 2, 3], 7*[' '])
    plt.title('KLLR Fits', size=20)

    ax = plt.subplot(2, 2, 3)
    plt.scatter(Xs[0], Ys2, s=2, color='lightcoral', alpha=0.5)
    plot_mean_profile_fit(X_new, mp, gp[1], x2_dim_len=3, ax=ax, color=['darkmagenta', 'black', 'darkgreen'],
                          label=[r'$\log_{10}(M_{200}) = 12.25$', r'$\log_{10}(M_{200}) = 12.75$',
                                 r'$\log_{10}(M_{200}) = 13.5$'])
    plt.xlim([-1.0, 0.5])
    plt.ylim([-2.0, 2.0])
    plt.xticks([-1, -0.5, 0.0, 0.5], ['-1', '-0.5', '0', '0.5'])
    plt.yticks([-2, -1, 0, 1, 2], ['-2', '-1', '0', '1', '2'])

    plt.xlabel(r'$\log_{10}\left(\frac{r}{R_{200}}\right)$', size=22)
    plt.ylabel(r'$\log_{10}(\rho_{\rm gas} / 10^{12})$', size=22)
    plt.grid()
    plt.legend(loc=3, prop={'size': 12})

    ax = plt.subplot(2, 2, 4)
    plt.scatter(Xs[0], Ys2, s=2, color='lightcoral', alpha=0.5)
    ax = Plot_Fit_Split(df, 'x', 'gas', 'm', split_bins=[12, 12.5, 13, 14], show_data=False,
                        nbins=25, xrange=[-1.2, 0.6], split_mode='Data',
                        kernel_type='gaussian', kernel_width=0.15, xlog=False,
                        labels=[r' ', r' ', r'$\log_{10}(M_{200})$'],
                        color=['darkmagenta', 'black', 'darkgreen'], ax=ax)
    plt.legend(loc=3, prop={'size': 12})
    plt.xlim([-1.0, 0.5])
    plt.ylim([-2.0, 2.0])
    plt.xticks([-1, -0.5, 0.0, 0.5], [' ', '-0.5', '0', '0.5'])
    plt.yticks([-2, -1, 0, 1, 2], 5*[' '])
    plt.xlabel(r'$\log_{10}\left(\frac{r}{R_{200}}\right)$', size=22)

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    check_directory('./plots/')
    plt.savefig('./plots/inferred_mean_relation_TNG.png', bbox_inches='tight')
    plt.close()


def plot_differential_gp(mp, gp, model):


    x1 = np.linspace(-1.0, 0.5, 201)
    x2 = np.array([12.5])
    X_new_1 = pm.math.cartesian(x1[:, None], x2[:, None])
    x2 = np.array([13.5])
    X_new_2 = pm.math.cartesian(x1[:, None], x2[:, None])

    plt.figure(figsize=(5, 7))

    ax = plt.subplot(2, 1, 1)

    df_dx_mu, df_dx_sd, X_df = differentiate_mean_profile(X_new_1, mp, gp[0], model, derivative_index=0, sample=400)
    ax.plot(X_df, df_dx_mu, '-', color='black', lw=2, label=r'$\log_{10}(M_{200}) = 12.5$')
    ax.fill_between(X_df, df_dx_mu - df_dx_sd, df_dx_mu + df_dx_sd, color='black', alpha=0.5)

    df_dx_mu, df_dx_sd, X_df = differentiate_mean_profile(X_new_2, mp, gp[0], model, derivative_index=0, sample=400)
    ax.plot(X_df, df_dx_mu, '--', color='orange', lw=2, label=r'$\log_{10}(M_{200}) = 13.5$')
    ax.fill_between(X_df, df_dx_mu - df_dx_sd, df_dx_mu + df_dx_sd, color='orange', alpha=0.5)

    plt.ylabel(r'$\frac{\partial \langle \log(\rho_{\star})\rangle}{\partial \log (r)}$', size=22)
    plt.legend(loc=3, prop={'size': 12})
    plt.xlim([-1.0, 0.5])
    plt.ylim([-6.0, 0.0])
    plt.xticks([-1, -0.5, 0.0, 0.5], 4*[' '])
    plt.grid()

    ax = plt.subplot(2, 1, 2)

    df_dx_mu, df_dx_sd, X_df = differentiate_mean_profile(X_new_1, mp, gp[1], model, derivative_index=0, sample=400)
    ax.plot(X_df, df_dx_mu, '-', color='black', lw=2, label=r'$\log_{10}(M_{200}) = 12.5$')
    ax.fill_between(X_df, df_dx_mu - df_dx_sd, df_dx_mu + df_dx_sd, color='black', alpha=0.5)

    df_dx_mu, df_dx_sd, X_df = differentiate_mean_profile(X_new_2, mp, gp[1], model, derivative_index=0, sample=400)
    ax.plot(X_df, df_dx_mu, '--', color='orange', lw=2, label=r'$\log_{10}(M_{200}) = 13.5$')
    ax.fill_between(X_df, df_dx_mu - df_dx_sd, df_dx_mu + df_dx_sd, color='orange', alpha=0.5)

    plt.ylabel(r'$\frac{\partial \langle \log(\rho_{\rm gas})\rangle}{\partial \log (r)}$', size=22)
    plt.xlabel(r'$\log_{10}\left(\frac{r}{R_{200}}\right)$', size=22)

    plt.legend(loc=3, prop={'size': 12})
    plt.xlim([-1.0, 0.5])
    plt.ylim([-6.0, 0.0])
    plt.xticks([-1, -0.5, 0.0, 0.5], [-1, -0.5, 0.0, 0.5])
    plt.grid()

    check_directory('./plots/')
    plt.savefig('./plots/TNG_differential_slope.pdf', bbox_inches='tight')
    plt.close()


def run_correlation_model(gp, mp, Xs, Yobs1, Yobs2, Ys1err, Ys2err, Ys1, Ys2, mcmc_samples=500):

    from examples.kllr import Plot_Cov_Corr_Matrix_Split

    df = pd.DataFrame({'x':Xs.T[0], 'm':Xs.T[1], 'stars':Ys1, 'gas':Ys2})

    plt.figure(figsize=(6, 4.5))
    ax = plt.subplot(111)
    ax = Plot_Cov_Corr_Matrix_Split(df, 'x', ['stars', 'gas'], 'm', split_bins=[12, 12.5, 13, 14],
                                    nbins=25, xrange=[-1.0, 0.5], nBootstrap=100,
                        Output_mode='corr', kernel_type='gaussian', kernel_width=0.15, split_mode='Data',
                        percentile=[16., 84.], xlog=False, labels=[r' ', r' ', r' ', r'$\log_{10}(M_{200})$'],
                        color=['darkmagenta', 'black', 'darkgreen'], verbose=True, ax=ax)

    xbins = np.linspace(-1.0, 0.5, 7)
    xbins = [[xbins[i], xbins[i + 1]] for i in range(len(xbins) - 1)]

    mbins = [[12, 12.5], [12.5, 13.0], [13.0, 14.0]]

    i = -0.02
    k = 0
    colors = ['darkmagenta', 'black', 'darkgreen']
    for mbin in mbins:

        mask_m = Xs.T[1] < mbin[1]
        mask_m *= Xs.T[1] > mbin[0]

        rmed = []
        rmin = []
        rmax = []
        x = []
        for xbin in (xbins):

            mask_x = Xs[mask_m].T[0] < xbin[1]
            mask_x *= Xs[mask_m].T[0] > xbin[0]

            r, sig_1, sig_2 = estimate_property_covariance(gp, mp, Xs[mask_m][mask_x], Yobs1[mask_m][mask_x],
                                                           Yobs2[mask_m][mask_x], Ys1err[mask_m][mask_x], Ys2err[mask_m][mask_x],
                                                           mcmc_samples=mcmc_samples, verbose=False)

            rmed += [np.percentile(r, 50)]
            rmin += [np.percentile(r, 50) - np.percentile(r, 16)]
            rmax += [np.percentile(r, 84) - np.percentile(r, 50)]
            x += [(xbin[1] + xbin[0]) / 2.0]

        ax.errorbar(np.array(x) + i, rmed, yerr=[rmin, rmax], fmt='o', color=colors[k])
        i += 0.02
        k += 1

    plt.legend(loc=4, prop={'size':12})
    plt.xlabel(r'$\log_{10}\left(\frac{r}{R_{200}}\right)$', size=22)
    plt.ylabel(r'${\rm corr}(\log(\rho_{\rm \star}) , \log(\rho_{\rm gas}))$', size=22)
    plt.ylim([-1, 1])
    plt.xlim([-1.0, 0.5])
    plt.grid(True)

    check_directory('./plots/')
    plt.savefig('./plots/inferred_correlations_TNG.png', bbox_inches='tight')


def check_directory(dir):
    """
    Check if it already exists using os.path.exists(directory). If not, create one.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

