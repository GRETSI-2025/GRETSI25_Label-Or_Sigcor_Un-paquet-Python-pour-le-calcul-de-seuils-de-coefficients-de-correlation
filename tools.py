"""
A number of tool functions to generate figures fror the article submitted to Gretsi 2025

Author: GJC Becq 
Date: 2025-03-28

"""

import numpy as np
import sigcor
from matplotlib import pyplot as plt

def plot_fr(k, ax=None, N=1000, seed=1, bins_exp=100, n_trial=100, bins_th=100, 
                xlabel=False, ylabel=False, **settings):
    if ax: 
        pass
    else: 
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    np.random.seed(seed)
    fr_exp_all = np.zeros((n_trial, bins_exp))
    for i_trial in range(n_trial): 
        r = np.zeros((N, ))
        for i in range(N): 
            x1 = np.random.randn(k)
            x2 = np.random.randn(k)
            r[i] = np.corrcoef(x1, x2)[0, 1]
        (fr_exp, x_exp) = np.histogram(r, bins=bins_exp, range=[-1, 1], density=True)
        fr_exp_all[i_trial, :] = fr_exp
    hr_th = histogram_r_theoretical(k, bins_th)
    dx_th = (2 / bins_th)
    fr_th = hr_th / dx_th 
    xc_th = np.arange(-1, 1, 2 / bins_th) + 1 / bins_th
    xc = (x_exp[:bins_exp] + x_exp[1:])/2
    m = np.mean(fr_exp_all, 0)
    s = np.std(fr_exp_all, 0)
    ax.plot(xc, m, color="tab:blue") 
    ax.fill_between(xc, m - s, m + s, alpha=0.5, color="tab:blue"); 
    ax.plot(xc_th, fr_th, color="tab:red"); 
    if xlabel: 
        ax.set_xlabel("r", **settings)
    if ylabel: 
        ax.set_ylabel("f(r)", **settings)

def get_curve_rho(alpha): 
    N1 = np.linspace(2, 9, 8, dtype=int) # 2 to 9 with 8 samples
    N2 = np.linspace(10, 100, 10, dtype=int) # 10 to 100 with 10 samples
    N3 = np.linspace(100, 1000, 10, dtype=int) # 100 to 1000 with 10 samples
    N = np.concatenate((N1, N2, N3))
    Rho = np.zeros_like(N, dtype="float")
    for (i, Ni) in enumerate(N): 
        Rho[i] = sigcor.get_rs(Ni, alpha)
    return (N, Rho)

def get_curve_rho_fisher(alpha): 
    N1 = np.linspace(2, 9, 8, dtype=int) # 2 to 9 with 8 samples
    N2 = np.linspace(10, 100, 10, dtype=int) # 10 to 100 with 10 samples
    N3 = np.linspace(100, 1000, 10, dtype=int) # 100 to 1000 with 10 samples
    N = np.concatenate((N1, N2, N3))
    Rho = np.zeros_like(N, dtype="float")
    for (i, Ni) in enumerate(N): 
        if Ni >= 4: 
            Rho[i] = sigcor.core.get_rs_fisher(Ni, alpha)
        else: 
            Rho[i] = np.nan
    return (N, Rho)


def get_curve_rho_exp(alpha, p=100): 
    N1 = np.linspace(2, 9, 8, dtype=int) # 2 to 9 with 8 samples
    N2 = np.linspace(10, 100, 10, dtype=int) # 10 to 100 with 10 samples
    N3 = np.linspace(100, 1000, 10, dtype=int) # 100 to 1000 with 10 samples
    N = np.concatenate((N1, N2, N3))
    Rho = np.zeros_like(N, dtype="float")
    for (i, Ni) in enumerate(N): 
        X = np.random.randn(p, Ni)
        C = np.corrcoef(X)
        val = []
        for j in range(p): 
            val = val + C[j, j + 1:].tolist()
        histo1 = np.histogram(val, bins=100, range=(-1, 1))
        c = sigcor.ext.scipy.stats.rv_histogram(histo1)
        Rho[i] = c.ppf(1 - alpha / 2)
    return (N, Rho)


def histogram_r_theoretical(N, n_bin=100, range_=(-1, 1)): 
    """
    get the histogram of r following the theoretical probability function of the correlation coefficient of two Gaussian random variates of size n. 
    
    Inputs
    
    N: int
        number of effective samples
    n_bin, range_: same as parameter bins and range for pylab.hist 
    
    Output
    
    hx: array with shape (n_bin, )
        The theoretical values of the pdf of the correlation coefficients in the n_bin bins from range_[0] to range_[1]. 
    
    
    """
    hx = np.zeros((n_bin, ))
    dx = (range_[1] - range_[0]) / n_bin
    if N > 3: 
        for i in range(n_bin): 
            r1 = range_[0] + i * dx
            r2 = range_[0] + (i + 1) * dx
            hx1 = sigcor.get_fr(r1, N)
            hx2 = sigcor.get_fr(r2, N)
            hx[i] = (hx1 + hx2) / 2 * dx
    elif N == 3: 
        for i in range(n_bin): 
            r1 = range_[0] + i * dx
            r2 = range_[0] + (i + 1) * dx
            if (r1 == -1) or (r2 == 1): 
                hx[i] = 2 ** (1 / 2) * dx ** (1 / 2) * 1 / np.pi
            else: 
                hx1 = sigcor.get_fr(r1, N)
                hx2 = sigcor.get_fr(r1, N)
                hx[i] = (hx1 + hx2) / 2 * dx
    elif N == 2: 
        hx[0] = 1/2
        hx[-1] = 1/2
    else: 
        pass
    return hx

def plot_rs_filtered(alpha, nM, nk, has_legend=True): 
    N = np.zeros((nM,)) 
    k = np.zeros((nk, )) 
    B = np.zeros((nk, ))
    RSF = np.zeros((nM, nk)) 
    for iN in range(nM):
        N[iN] = iN + 2
    for ik in range(nk):
        k[ik] = ik + 1
        B[ik] = 1 / 2 ** (k[ik]) 
    for iN in range(nM):
        rs_f = sigcor.get_rs(N[iN], alpha)
        RSF[iN, 0] = rs_f
        for ik in range(1, nk):
            try: 
                rs_f = sigcor.get_rs_filtered(N[iN], alpha, B[ik], 1)
                RSF[iN, ik] = rs_f
            except :
                RSF[iN, ik] = np.nan
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for ik in range(nk):
        a = int(2**(k[ik]-1))
        kappa = 1 / a
        label = f"{a:3d}: {kappa:5.2f}"
        ax.plot(N, RSF[:, ik], label=label, color=(ik / nk, 0, 0))
    ax.set_xlim([0, 1500])
    ax.set_xlabel("N")
    ax.set_ylabel("$r^*$")
    if has_legend: 
        ax.legend(title=f"   a: κ", fontsize=8)
    fig.suptitle(f"α = {alpha:3.2f}")
    fig.tight_layout()
    return fig