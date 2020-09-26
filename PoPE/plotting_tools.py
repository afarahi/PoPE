import numpy as np
import matplotlib.pylab as plt

import matplotlib as mpl

mpl.rcParams['xtick.direction'], mpl.rcParams['ytick.direction'] = 'in', 'in'
mpl.rcParams['xtick.major.size'], mpl.rcParams['xtick.minor.size'] = 14, 8
mpl.rcParams['xtick.major.width'], mpl.rcParams['xtick.minor.width'] = 1.0, 0.8
mpl.rcParams['xtick.major.pad'], mpl.rcParams['xtick.minor.pad'] = 10, 10
mpl.rcParams['ytick.major.size'], mpl.rcParams['ytick.minor.size'] = 14, 8
mpl.rcParams['ytick.major.width'], mpl.rcParams['ytick.minor.width'] = 1.0, 0.8
mpl.rcParams['xtick.labelsize'], mpl.rcParams['ytick.labelsize'] = 16, 16
default_cmap = plt.cm.Purples


def plot_mean_profile_fit(X_new, mp, gp, x2_dim_len = 1, ax = None, color = ['indianred'], X_new_index = 0,
                          show_confidence_intervals = True, label = None, model = None):
    """
    Run GP model and construct the average profile of two properties

    Parameters
    ----------
    Xs : array-like, shape (n_data, n_feature)
        Independent variables

    mp : instance of pymc3.MAP fit

    gp : list, len=2
        instances of GP average profiles for property s1 and s2 respectively

    x2_dim_len : int
        Number of binns of data.

    ax : instance of Matplotlib.axes
        Axes on which results were plotted

    color : list, len=x2_dim_len
        List of colors.

    X_new_index : int
        Index of the independent variable used for plotting (xaxis)

    show_confidence_intervals : bool
        True,

    label :
        if None do nothing.

    model : pymc3 Model instance
        if provided sample from the model if None use an analytical form. Analytical form is preferred.
        It draw 200 samples. Number of samples is hard-coded.

    Returns
    -------
    ax : instance of Matplotlib.axes
    """

    if ax == None:
        ax = plt.figure(figsize=(5, 5))

    if model is None:
        mu, var = gp.predict(X_new, point=mp, diag=True)
        sd = np.sqrt(var)
    else:
        ind = np.random.random()
        var_name = 'f_pred_%0.8f' % ind

        with model:
          f_pred = gp.conditional(var_name, X_new)
          pred_samples = pm.sample_posterior_predictive([mp], vars=[f_pred], samples=200)

        mu = np.mean(pred_samples[var_name].T, axis=1)
        sd = np.std(pred_samples[var_name].T, axis=1)

    line_type = ['-', ':', '--']
    for i in range(x2_dim_len):

        # plot mean
        if label is None:
            ax.plot(X_new.T[X_new_index][i::x2_dim_len], mu[i::x2_dim_len], line_type[i%3], color=color[i], lw=2)
        else:
            ax.plot(X_new.T[X_new_index][i::x2_dim_len], mu[i::x2_dim_len], line_type[i%3], color=color[i], lw=2, label=label[i])

        if show_confidence_intervals:
            # plot 2σ intervals
            ax.fill_between(X_new.T[X_new_index][i::x2_dim_len],
                            mu[i::x2_dim_len] - 2 * sd[i::x2_dim_len],
                            mu[i::x2_dim_len] + 2 * sd[i::x2_dim_len], color=color[i], alpha=0.5)
    return ax

