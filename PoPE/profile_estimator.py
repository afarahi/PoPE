import pymc3 as pm
import numpy as np
import theano.tensor as tt


def make_cov_mtx_from_chol_vec(chol_vec, ndim):
    """
    Construct a covariance matrix from Cholesky decomposition

    Parameters
    ----------
    chol_vec : array-like, shape (ndim,)
        Training data.

    ndim : int
        Covariance matrix dimension.

    Returns
    -------
    Covariance matrix : array-like, shape (ndim, ndim)
        Returns reconstructed covariance matrix.

    """
    chol_mtx = np.zeros((ndim, ndim))
    chol_mtx[np.tril_indices(ndim)] = chol_vec
    cov_mtx = np.dot(chol_mtx, chol_mtx.T)
    return (cov_mtx)


def construct_xu(Xs, Xu_shapes):
    """
    Construct a Xu grid for the GP model.

    Parameters
    ----------
    Xs : array-like, shape (n_data, n_feature)
        Independent varibales

    Xu_shapes : array-like, shape (n_data,)
        Size of Xu vector per feature dimension.

    Returns
    -------
    Xu matrix : array-like, shape (Xu_shapes[0] x ... x Xu_shapes[n_feature], 2)
        Xu grid.

    """

    input_dim = Xs.shape[1]

    if input_dim == 1:
        xu1 = np.linspace(np.min(Xs.T[0]) + 0.01,  np.max(Xs.T[0]) - 0.01, Xu_shapes)
        return xu1[:, None]
    if input_dim > 1:
        xu = [ np.linspace(np.min(Xs.T[i]) + 0.01, np.max(Xs.T[i]) + 0.01 - 0.01, Xu_shapes[i])[:, None]
              for i in range(input_dim) ]
        return pm.math.cartesian(*xu)


def estimate_mean_property_profile(Xs, Ys1, Ys2, Ys1err, Ys2err, Xu_shapes, kernel_scales, verbose = True):
    """
    Run GP model and construct the average profile of two properties

    Parameters
    ----------
    Xs : array-like, shape (n_data, n_feature)
        Independent varibales

    Ys1, Ys2 : array-like, shape (n_data, n_feature)
        Property Profile for property s1 and s2 respectively

    Ys1err, Ys2err : array-like, shape (n_data, n_feature)
        Property profile measurement error for property s1 and s2 respectively

    Xu_shapes : array-like, int, shape (n_feature,)
        Size of Xu vector per feature dimension.

    kernel_scales : array-like, shape (n_feature,)
        Matern52 kernel scale per feature dimension.

    verbose : boolean
        Controls the verbosity of the model's output.

    Returns
    -------
    mp, [gp1, gp2], model:
        instance of pymc3.MAP fit
        instances of GP average profiles
        instance of pymc3 model

    """

    # check for attribute compatibility.
    check_attributes(Xs, Ys1, Ys2, Ys1err, Ys2err)
    check_attributes_compatible_with_feature(Xs.shape[1], Xu_shapes, 'Xu_shapes')
    check_attributes_compatible_with_feature(Xs.shape[1], kernel_scales, 'kernel_scales')

    input_dim = Xs.shape[1]

    Xu_init = construct_xu(Xs, Xu_shapes)

    # construct the model
    with pm.Model() as model:

        eta1 = pm.HalfNormal("eta1", sigma=2)
        eta2 = pm.HalfNormal("eta2", sigma=2)

        cov_func1 = eta1 ** 2
        for i in range(input_dim):
            cov_func1 *= pm.gp.cov.Matern52(input_dim=input_dim, ls=kernel_scales[i], active_dims=[i])

        cov_func2 = eta2 ** 2
        for i in range(input_dim):
            cov_func2 *= pm.gp.cov.Matern52(input_dim=input_dim, ls=kernel_scales[i], active_dims=[i])

        gp1 = pm.gp.MarginalSparse(cov_func=cov_func1, approx="VFE")
        gp2 = pm.gp.MarginalSparse(cov_func=cov_func2, approx="VFE")

        sigma_1 = pm.HalfCauchy("sigma_1", beta=5)
        sigma_2 = pm.HalfCauchy("sigma_2", beta=5)

        y1_err_ = tt.sqrt(Ys1err * Ys1err + sigma_1 * sigma_1)
        y2_err_ = tt.sqrt(Ys2err * Ys2err + sigma_2 * sigma_2)

        y1_ = gp1.marginal_likelihood("y1", X=Xs, Xu=Xu_init, y=Ys1, noise=y1_err_)
        y2_ = gp2.marginal_likelihood("y2", X=Xs, Xu=Xu_init, y=Ys2, noise=y2_err_)

    # find MAP
    with model:
        mp = pm.find_MAP(method="BFGS", progressbar=verbose)

    return mp, [gp1, gp2], model


def estimate_property_covariance(gp, mp, Xs, Ys1, Ys2, Ys1err, Ys2err,
                                 nu=2.0, mcmc_samples=5000, verbose=True):
    """
    Run Covariance estimation model and infer the 2x2 property covariance gives a set of binned data.

    Parameters
    ----------
    Xs : array-like, shape (n_data, n_feature)
        Independent variables.

    Ys1, Ys2 : array-like, shape (n_data,)
        Property Profile for property s1 and s2 respectively

    Ys1err, Ys2err : array-like, shape (n_data,)
        Property profile measurement error for property s1 and s2 respectively

    nu : float (0 < nu)
        Hyper-parameter of the LKJ prior. Recomend to set it to 1 < nu < 5.

    mcmc_samples : int
        The number of samples to draw. Defaults to 5000.

    verbose : boolean
        Controls the verbosity of the model's output.

    Returns
    -------
    mp, [gp1, gp2], model:
        instance of pymc3.MAP fit
        instances of GP average profiles
        instance of pymc3 model

    """

    check_attributes(Xs, Ys1, Ys2, Ys1err, Ys2err)

    if nu < 0:
        raise ValueError("nu must be larger than zero (a value between 1 and 5 is recommended) , nu = %0.2f."%nu)

    # data size
    n_sample = len(Ys1)

    # size of the covariance matrix. The current code only works for 2D covariance matrix.
    shape = 2

    # If there are less than 100 data points return -10, -10, -10
    if n_sample < 100:
        return -10, -10, -10

    # Construct the prior on the mean profiles (employing the fitted GP model)
    mu1, var1 = gp[0].predict(Xs, point=mp, pred_noise=False, diag=True)
    mu2, var2 = gp[1].predict(Xs, point=mp, pred_noise=False, diag=True)
    std1 = np.sqrt(var1)
    std2 = np.sqrt(var2)

    with pm.Model() as model_correlation:

        y1_expected = pm.Normal('y1_expected', mu=mu1, sigma=std1, shape=n_sample)
        y2_expected = pm.Normal('y2_expected', mu=mu2, sigma=std2, shape=n_sample)

        # generate a sample of
        sd_dist = pm.HalfNormal.dist(sigma=5.0, shape=shape)
        chol_packed = pm.LKJCholeskyCov('chol_packed', n=shape, eta=nu, sd_dist=sd_dist)
        chol = pm.expand_packed_triangular(shape, chol_packed)
        vals = pm.MvNormal('true_quantities', mu=0.0, chol=chol, shape=(n_sample, shape))

        y1_e = vals.T[0] + y1_expected
        y2_e = vals.T[1] + y2_expected

        y1_ = pm.Normal('y1_likelihood', mu=y1_e, sigma=Ys1err + 0.1, observed=Ys1)
        y2_ = pm.Normal('y2_likelihood', mu=y2_e, sigma=Ys2err + 0.1, observed=Ys2)

    with model_correlation:
        # Use elliptical slice sampling
        trace = pm.sample(mcmc_samples, chains=2, init='jitter+adapt_diag', progressbar=verbose)

    r = []
    sig_1 = []
    sig_2 = []
    for chol_p in zip(trace['chol_packed'][:]):
        cov = make_cov_mtx_from_chol_vec(chol_p, ndim=shape)
        r += [cov[1, 0] / np.sqrt(cov[0, 0] * cov[1, 1])]
        sig_1 += [np.sqrt(cov[0, 0])]
        sig_2 += [np.sqrt(cov[1, 1])]

    print(' r = %0.2f +- %0.2f' % (np.mean(r), np.std(r)))
    return r, sig_1, sig_2


def differentiate_mean_profile(X_new, mp, gp, model, derivative_index=0, sample=200, verbose=True):
    """
    This function draws N random curves from the fitted GP model (average profile) and numerically compute
     their derivative respect to variable defined with `derivative_index`, i.e. df/dx_i.

    Parameters
    ----------
    Xs : array-like, shape (n_grid, n_feature)
        Grid of independent variables

    mp : instance of pymc3.MAP fit

    gp : list, len=2
        instances of GP average profiles for property s1 and s2 respectively

    model : pymc3 Model instance
        if provided sample from the model if None use an analytical form. Analytical form is preferred.
        It draw 200 samples. Number of samples is hard-coded.

    derivative_index : int
        Number of binns of data.

    sample : int

    verbose : boolean
        Controls the verbosity of the model's output.

    Returns
    -------
    tuple

    df_dx_mu : array-like, shape (n_grid, n_feature)
        Average of df/dx_i

    df_dx_sd : array-like, shape (n_grid, n_feature)
        Standard deviation of df/dx_i

    X_df : array-like, shape (n_grid, n_feature)
        Grid.
    """

    ind = np.random.random()
    var_name = 'f_pred_%0.8f'%ind

    with model:
        f_pred = gp.conditional(var_name, X_new)
        pred_samples = pm.sample_posterior_predictive([mp], vars=[f_pred], samples=sample, progressbar=verbose)

    dx = X_new.T[derivative_index][1:] - X_new.T[derivative_index][:-1]
    df = pred_samples[var_name].T[1:] - pred_samples[var_name].T[:-1]

    df_dx = df.T / dx
    X_df = (X_new.T[derivative_index][1:] + X_new.T[derivative_index][:-1]) / 2.0

    df_dx_mu = np.mean(df_dx.T, axis=1)
    df_dx_sd = np.std(df_dx.T, axis=1)

    return df_dx_mu, df_dx_sd, X_df


def estimate_mean_property_profile_1D(Xs, Ys1, Ys2, Ys1err, Ys2err, Xu_x_shape=8):
    """
    same as estimate_mean_property_profile but only works for 1D Xs.
    """

    check_attributes(Xs, Ys1, Ys2, Ys1err, Ys2err)

    xmin = np.min(Xs.T[0])
    xmax = np.max(Xs.T[0])  # distance

    xu1 = np.linspace(xmin + 0.1, xmax - 0.1, Xu_x_shape)
    Xu_init = xu1[:, None]
    # Xu_init = np.random.rand(cov_shape)

    input_dim = Xs.shape[1]

    with pm.Model() as model:

        x_scale_1 = pm.Gamma("x_scale_1", alpha=2, beta=1)
        eta1 = pm.HalfCauchy("eta1", beta=5)

        cov_func1 = eta1 ** 2 * pm.gp.cov.Matern52(input_dim=input_dim, ls=x_scale_1, active_dims=[0])

        x_scale_2 = pm.Gamma("x_scale_2", alpha=2, beta=1)
        eta2 = pm.HalfCauchy("eta2", beta=5)
        cov_func2 = eta2 ** 2 * pm.gp.cov.Matern52(input_dim=input_dim, ls=x_scale_2, active_dims=[0])

        gp1 = pm.gp.MarginalSparse(cov_func=cov_func1, approx="FITC")
        gp2 = pm.gp.MarginalSparse(cov_func=cov_func2, approx="FITC")

        sigma_1 = pm.HalfCauchy("sigma_1", beta=5)
        sigma_2 = pm.HalfCauchy("sigma_2", beta=5)

        y1_err_ = tt.sqrt(Ys1err * Ys1err + sigma_1 * sigma_1)
        y2_err_ = tt.sqrt(Ys2err * Ys2err + sigma_2 * sigma_2)

        # Xu = pm.Flat("Xu", shape=20, testval=Xu_init)

        y1_ = gp1.marginal_likelihood("y1", X=Xs, Xu=Xu_init, y=Ys1, noise=y1_err_)
        y2_ = gp2.marginal_likelihood("y2", X=Xs, Xu=Xu_init, y=Ys2, noise=y2_err_)

    # compute MAP
    with model:
        mp = pm.find_MAP(method="BFGS")

    return mp, [gp1, gp2]


def check_attributes(Xs, Ys1, Ys2, Ys1err, Ys2err):
    """
    Check if common attributes are in correct format.
    """

    if len(Xs.shape) != 2:
        raise ValueError(
            "Incompatible dimension for X. X should be two dimensional numpy array,"
            " len(X.shape) = %i."%len(Xs.shape))

    if Xs.shape[0] != Ys1.shape[0]:
        raise ValueError(
            "Incompatible dimension for Xs and Y. Xs and Y should be one dimensional numpy array,"
            " len(Xs.shape) = %i while len(Y.shape) = %i." % (Xs.shape[0], Ys1.shape[0]))

    if Ys1.shape[0] != Ys2.shape[0]:
        raise ValueError(
            "Incompatible dimension for Ys1 and Ys2. Ys1 and Ys2 should have the same length,"
            " Ys1.shape[0] = %i while Ys2.shape[0] = %i." % (Ys1.shape[0], Ys2.shape[0]))

    if Ys1.shape[0] != Ys1err.shape[0]:
        raise ValueError(
            "Incompatible dimension for Ys1 and Ys1err. Ys1 and Ys1err should have the same length,"
            " Ys1.shape[0] = %i while Ys1err.shape[0] = %i." % (Ys1.shape[0], Ys1err.shape[0]))

    if Ys2.shape[0] != Ys2err.shape[0]:
        raise ValueError(
            "Incompatible dimension for Ys2 and Ys2err. Ys2 and Ys2err should have the same length,"
            " Ys2.shape[0] = %i while Ys2err.shape[0] = %i." % (Ys2.shape[0], Ys2err.shape[0]))

def check_attributes_compatible_with_feature(n_feature, attr, label):
    """
    Check if an attribute has the same length as number of features.
    """
    if n_feature != len(attr):
        raise ValueError("Incompatible %s length. It's length should be the same as feature size" +
                         ", n_feature=%i. "%(label, n_feature))