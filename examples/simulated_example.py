from PoPE.profile_estimator import estimate_mean_property_profile, estimate_property_covariance
from PoPE.plotting_tools import plot_mean_profile_fit

import matplotlib.pylab as plt
import numpy as np
import os

def check_directory(dir):
    """
    Check if it already exists using os.path.exists(directory). If not, create one.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def controled_experiment():

    # generate a random fake profile
    x, Yexp1, Yexp2, Ys1, Ys2, Yobs1, Yobs2, Ys1err, Ys2err = generate_fake_profiles(N=3000)

    # visualize the fake profiles
    visualize_data(x, Ys1, Ys2, Yobs1, Yobs2, Ys1err, Ys2err)

    Xs = x[:, None]

    # estimate mean profiles
    mp, gp, model = estimate_mean_property_profile(Xs, Yobs1, Yobs2, Ys1err, Ys2err, Xu_shapes=15, kernel_scales=[2.0])

    # visualize the estimated mean profiles
    plot_mean_relation_gp(mp, gp, Xs, Ys1, Ys2)

    # estimate and visualize the correlation profile
    run_correlation_model(gp, mp, Xs, Yobs1, Yobs2, Ys1err, Ys2err, mcmc_samples=1500)


def run_correlation_model(gp, mp, Xs, Ys1, Ys2, Ys1err, Ys2err, mcmc_samples=500):

    xbins = np.linspace(0.0, 2.0, 11)
    xbins = [[xbins[i], xbins[i + 1]] for i in range(len(xbins) - 1)]

    rmed = []
    rmin = []
    rmax = []
    x = []
    for xbin in (xbins):

        mask = Xs.T[0] < xbin[1]
        mask *= Xs.T[0] > xbin[0]

        r, sig_1, sig_2 = estimate_property_covariance(gp, mp, Xs[mask], Ys1[mask], Ys2[mask], Ys1err[mask], Ys2err[mask],
                                                       nu=4.0, mcmc_samples=mcmc_samples)

        rmed += [np.percentile(r, 50)]
        rmin += [np.percentile(r, 50) - np.percentile(r, 16)]
        rmax += [np.percentile(r, 84) - np.percentile(r, 50)]
        x += [(xbin[1] + xbin[0]) / 2.0]

    plt.errorbar(np.array(x), np.array(x)*0.0 + 0.5, color='orange', lw=5.0, label='Input Correlation')
    plt.errorbar(np.array(x), rmed, yerr=[rmin, rmax], fmt='o', label='Inferred')

    plt.legend(loc=4, prop={'size': 18})
    plt.xlabel('r', size=23)
    plt.ylabel('correlation', size=23)
    plt.ylim([-1, 1])
    plt.grid()
    check_directory('./plots/')
    plt.savefig('./plots/inferred_correlations_fake_sim.png', bbox_inches='tight')


def generate_fake_profiles(N=10000, min_SNR=1.0, max_SNR=3.0, corr=0.5):

    x = np.random.uniform(0.01, 2.0, N)
    y_exp = -x**2/5.0 - 2.0*x/5.0 + 1
    z_exp = -x**3/5.0 + 2.0*x/5.0

    Yexp = np.array([y_exp, z_exp]).T

    cov = [[0.25, corr*0.25], [corr*0.25, 0.25]]  # diagonal covariance

    Ys = np.random.multivariate_normal([0.0, 0.0], cov, size=N) + Yexp

    inv_min_SNR = 1.0 / min_SNR
    inv_max_SNR = 1.0 / max_SNR

    Yobs = Ys.copy()
    Yerr1 = (inv_min_SNR + (inv_max_SNR - inv_min_SNR) * np.random.random(N))
    Yobs.T[0] = np.random.normal(Yobs.T[0], Yerr1)
    Yerr2 = (inv_min_SNR + (inv_max_SNR - inv_min_SNR) * np.random.random(N))
    Yobs.T[1] = np.random.normal(Yobs.T[1], Yerr2)

    return x, Yexp.T[0], Yexp.T[1], Ys.T[0], Ys.T[1], Yobs.T[0], Yobs.T[1], Yerr1, Yerr2


def compute_mean_profile(x):

    y_exp = -x**2/5.0 - 2.0*x/5.0 + 1
    z_exp = -x**3/5.0 + 2.0*x/5.0

    return y_exp, z_exp


def visualize_data(x, Ys1, Ys2, Yobs1, Yobs2, Ys1err, Ys2err):
    """
        Visualize a fake simulated sample.
    """

    xnew = np.linspace(0.0, 2.0, 1001)
    y1m, y2m = compute_mean_profile(xnew)

    plt.figure(figsize=(13, 10))

    ax = plt.subplot(2, 2, 1)
    plt.scatter(x, Ys1, s=2, color='lightgrey')
    ax.plot(xnew, y1m, 'k--', lw=3.0)
    plt.xlim([0.0, 2.0])
    plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0], 5*[''])
    plt.ylabel(r'$\ln(\rho_1)$', size=24)
    plt.ylim([-3, 3])
    plt.grid()
    plt.title('True profile measures', size=18)

    ax = plt.subplot(2, 2, 2)
    ax.errorbar(x[::10], Yobs1[::10], yerr=Ys1err[::10], fmt='.', markersize=2.0, color='steelblue')
    ax.plot(xnew, y1m, 'k--', lw=3.0)
    plt.xlim([0.0, 2.0])
    plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0], 5*[''])
    plt.yticks([-3, -2, -1, 0, 1, 2, 3], 7*[''])
    plt.ylim([-3, 3])
    plt.title('Observed profile measures', size=18)
    plt.grid()

    ax = plt.subplot(2, 2, 3)
    plt.scatter(x, Ys2, s=2, color='lightgrey')
    ax.plot(xnew, y2m, 'k--', lw=3.0)
    plt.xlim([0.0, 2.0])
    plt.ylim([-3, 3])
    plt.xlabel('r', size=24)
    plt.ylabel(r'$\ln(\rho_2)$', size=24)
    plt.grid()

    ax = plt.subplot(2, 2, 4)
    ax.errorbar(x[::10], Yobs2[::10], yerr=Ys2err[::10], fmt='.',  markersize=2.0, color='indianred')
    ax.plot(xnew, y2m, 'k--', lw=3.0)
    plt.xlim([0.0, 2.0])
    plt.ylim([-3, 3])
    plt.yticks([-3, -2, -1, 0, 1, 2, 3], 7*[''])
    plt.xlabel('r', size=24)
    plt.grid()

    plt.subplots_adjust(hspace=0.05, wspace=0.1)
    check_directory('./plots/')
    plt.savefig('./plots/fake_simulated_data.png', bbox_inches='tight')
    plt.close()

    plt.hist(1.0 / np.sqrt((np.exp(Ys1err ** 2) - 1.0)), range=(1.0, 3.0), bins=60, alpha=1.0, label=r'$\ln(\rho_{1, \rm obs})$')
    plt.hist(1.0 / np.sqrt((np.exp(Ys2err ** 2) - 1.0)), range=(1.0, 3.0), histtype='step', color='indianred',
             lw=3.0, bins=60, alpha=1.0, label=r'$\ln(\rho_{2, \rm obs})$')

    plt.xlim([1.0, 3.0])
    plt.legend(loc=1, prop={'size':18})
    plt.grid()
    plt.xlabel('SNR', size=22)
    plt.ylabel('PDF', size=22)
    check_directory('./plots/')
    plt.savefig('./plots/SNR_fake_simulation.pdf', bbox_inches='tight')
    plt.close()


def plot_mean_relation_gp(mp, gp, Xs, Ys1, Ys2):

    x1 = np.linspace(0.0, 2.0, 201)
    X_new = x1[:, None]
    Ysm1, Ysm2 = compute_mean_profile(x1)

    plt.figure(figsize=(6, 10))

    ax = plt.subplot(2, 1, 1)
    plt.scatter(Xs, Ys1, s=2, color='lightgrey')
    ax = plot_mean_profile_fit(X_new, mp, gp[0], ax=ax, color=['steelblue'], show_confidence_intervals=True)
    ax.plot(X_new, Ysm1, 'k--')
    plt.xlim([0.0, 2.0])
    plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0], 5*[''])
    plt.ylabel(r'$\ln(\rho_1)$', size=24)
    plt.grid()

    ax = plt.subplot(2, 1, 2)
    plt.scatter(Xs, Ys2, s=2, color='lightgrey')
    ax = plot_mean_profile_fit(X_new, mp, gp[1], ax=ax, color=['indianred'], show_confidence_intervals=True)
    ax.plot(X_new, Ysm2, 'k--')
    plt.xlim([0.0, 2.0])
    plt.xlabel('r', size=24)
    plt.ylabel(r'$\ln(\rho_2)$', size=24)
    plt.grid()

    plt.subplots_adjust(hspace=0.05)
    check_directory('./plots/')
    plt.savefig('./plots/inferred_mean_profiles_fake_simulation.png', bbox_inches='tight')
    plt.close()


def controled_experiment_2():

    def run_local_correlation_model(gp, mp, Xs, Ys1, Ys2, Ys1err, Ys2err, mcmc_samples=500):

        r, sig_1, sig_2 = modeling_correlation(gp, mp, Xs, Ys1, Ys2, Ys1err, Ys2err,
                                               Xbin=[0.0, 2.0], mcmc_samples=mcmc_samples)

    # generate a random profile
    for iN in [500, 1000, 5000]:

        x, Yexp1, Yexp2, Ys1, Ys2, Yobs1, Yobs2, Ys1err, Ys2err = generate_profile(N=iN, min_SNR=1.0,
                                                                                   max_SNR=10.0, corr=0.25)
        Xs = x[:, None]

        mp, gp = correlation_analysis_model(Xs, Yobs1, Yobs2, Ys1err, Ys2err, Xu_x_shape=20)
        plot_mean_relation_gp(mp, gp, Xs, Ys1, Ys2)

        run_local_correlation_model(gp, mp, Xs, Yobs1, Yobs2, Ys1err, Ys2err, mcmc_samples=1500)
