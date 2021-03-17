import os
from itertools import combinations

import numpy as np
from numpy.random import default_rng
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

# 1st digit
# stats = Counter(sample.flatten())
# stats = [[stats[digit], benford[digit - 1]] for digit in stats.keys()]
# stats = np.array(stats)
# stats[:, 0] /= n_samples * n_dims
# err_first = mean_squared_error(stats[:, 0], stats[:, 1]) ** 0.5


def disc_benford(sample: np.ndarray):
    n_samples, n_dims = sample.shape

    # Transform to logarithm law
    sample = np.power(10, sample)
    sample = np.clip(sample, 1, 9)
    sample = sample.astype(int)

    x = np.arange(1, 10)
    benford = np.log10(1 + 1 / x)

    stats = np.bincount(sample.flatten(), minlength=10)[1:] / (n_samples *
                                                               n_dims)
    err_first = mean_squared_error(stats, benford) ** 0.5

    return err_first


def disc_benford_2d(sample: np.ndarray):
    n_samples, n_dims = sample.shape

    # Transform to logarithm law
    sample = np.power(10, sample)
    sample = np.clip(sample, 1, 9)
    sample = sample.astype(int)

    x = np.arange(10, 100)
    benford = np.log10(1 + 1 / x)

    # canonical subspaces without diagonal
    idx_dim = np.array(list(combinations(range(n_dims), 2)))
    idx_dim = np.concatenate([idx_dim, idx_dim[:, ::-1]])
    sample = np.stack([sample[:, idx_dim[:, 0]], sample[:, idx_dim[:, 1]]]).T
    sample = sample.reshape(-1, 2)

    # 2 digits concatenation
    sample = sample[:, 0] * 10 + sample[:, 1]

    stats = np.bincount(sample, minlength=100)[10:] / len(sample)
    err_first = mean_squared_error(stats, benford) ** 0.5

    return err_first


path = f'metric'
os.makedirs(path, exist_ok=True)

n_conv = 49
n_dims = 4
MIN_POWER = 4  # 4
POWER_MAX = 12  # 12
ns_gen = 2 ** np.arange(MIN_POWER, POWER_MAX + 1)

qmc_engine = qmc.Sobol(d=n_dims, scramble=True)
rng = default_rng()

err_mc = []
err_sobol = []
for n_samples in ns_gen:
    err_mc_ = 0
    err_sobol_ = 0
    for _ in range(n_conv):
        sample = rng.random((n_samples, n_dims))
        err_mc_ += disc_benford_2d(sample)

        sample = qmc_engine.random(n_samples)
        err_sobol_ += disc_benford_2d(sample)

    err_mc.append(err_mc_ / n_conv)
    err_sobol.append(err_sobol_ / n_conv)

err_mc = np.array(err_mc)
err_sobol = np.array(err_sobol)

fig, ax = plt.subplots()

# convergence rate
# ratio_1 = err_mc[0] / ns_gen[0] ** (-1/2)
# ratio_2 = err_sobol[0] / ns_gen[0] ** (-2 / 2)
# ax.plot(ns_gen, ns_gen ** (-1 / 2) * ratio_1, ls='-', c='k')
# ax.plot(ns_gen, ns_gen ** (-2/2) * ratio_2, ls='-', c='k')

# ax.plot(ns_gen, err_sobol[:, 0]/err_mc[:, 0], ls=':', label="Sobol'/MC")
ax.plot(ns_gen, err_mc, ls=':', label="MC")
ax.plot(ns_gen, err_sobol, ls='-.', label="Sobol'")

ax.set_xlabel(r'$N_s$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(ns_gen)
ax.set_xticklabels([fr'$2^{{{ns}}}$' for ns in np.arange(MIN_POWER, POWER_MAX + 1)])
ax.set_ylabel(r'$\epsilon$')
ax.legend(labelspacing=0.7, loc='best')
fig.tight_layout()
# plt.show()
fig.savefig(os.path.join(path, f'metric_2d_{n_dims}.pdf'),
            transparent=True, bbox_inches='tight')


# fig, ax = plt.subplots()
# ax.plot(ns_gen, err_sobol/err_mc, ls=':', label="Sobol'/MC")
#
# ax.set_xlabel(r'$N_s$')
# ax.set_xscale('log')
# #ax.set_yscale('log')
# ax.set_xticks(ns_gen)
# ax.set_xticklabels([fr'$2^{{{ns}}}$' for ns in np.arange(MIN_POWER, POWER_MAX + 1)])
# ax.set_ylabel(r'$\epsilon$')
# ax.legend(labelspacing=0.7, loc='best')
# fig.tight_layout()
# # plt.show()
# fig.savefig(os.path.join(path, f'ratio_metric_{n_dims}.pdf'),
#             transparent=True, bbox_inches='tight')


# rng = np.random.default_rng()
# ratio_cd = []
# ratio_nb = []
# for _ in range(100):
#     # n_samples_1, n_dims = rng.integers([100, 2], [2000, 10])
#     # n_samples_2 = rng.integers(10, 100)
#
#     # space_1 = rng.random((n_samples_1, n_dims))
#     # space_2 = rng.random((n_samples_2, n_dims))
#     # space_2 = np.concatenate([space_1, space_2])
#
#     power, n_dims = rng.integers([5, 2], [13, 10])
#
#     engine = qmc.Sobol(d=n_dims)
#     space_1 = engine.random(2**power)
#     space_2 = engine.random(2**power)
#     space_2 = np.concatenate([space_1, space_2])
#
#     ratio_cd.append(qmc.discrepancy(space_1) / qmc.discrepancy(space_2))
#     nb_ratio = disc_benford(space_1) / disc_benford(space_2)
#     ratio_nb.append(nb_ratio)
#
# fig, ax = plt.subplots()
# ax.scatter(ratio_cd, ratio_nb, marker='+')
#
# mini = np.min(np.concatenate([[1], ratio_cd, ratio_nb]))
# maxi = np.max(np.concatenate([ratio_cd, ratio_nb]))
# ax.plot([mini, maxi], [mini, maxi], c='k')
#
# ax.plot([mini, maxi], [1, 1], c='k', ls='--', alpha=0.5)
# ax.plot([1, 1], [mini, maxi], c='k', ls='--', alpha=0.5)
#
# ax.set_xlabel('CD')
# ax.set_ylabel('NB')
#
# fig.tight_layout()
# #plt.show()
# fig.savefig(os.path.join(path, f'ratio_CD_NB.pdf'),
#             transparent=True, bbox_inches='tight')
