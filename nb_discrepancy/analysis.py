"""Benford's law to measure random number generation quality.

.. note:: This script relies on a modified version of scipy. Pull Request:
          https://github.com/scipy/scipy/pull/10844

---------------------------

MIT License

Copyright (c) 2020 Pamphile Tupui ROY

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
from collections import Counter

from scipy.stats import qmc

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

sobol = False
n_conv = 99
n_dims = 10
MIN_POWER = 5  # 4
POWER_MAX = 12  # 13
ns_gen = 2 ** np.arange(MIN_POWER, POWER_MAX + 1)

digits = list(range(1, 10))
benford = [np.log10(1 + 1 / x) for x in digits]


def sampler_sobol_scrambled(ns):
    """Sobol' scrambled."""
    engine = qmc.Sobol(d=n_dims, scramble=True)
    return engine.random(ns)


datasets = []
for _ in range(n_conv):
    dataset = []
    for n_samples in ns_gen:

        if sobol:
            sample = sampler_sobol_scrambled(n_samples)
        else:
            sample = np.random.sample((n_samples, n_dims))

        # Transform to Bendford's law
        sample = np.power(10, sample)
        sample = np.clip(sample, 1, 9)
        sample = sample.astype(int).astype(str)

        # Analysis

        # digits2 = list(range(1, 100))
        # benford2 = [np.log10(1 + 1 / x) for x in digits2]
        # err_benford = {}
        # _sample = ["".join(s) for s in sample.tolist()]
        # stats = Counter(_sample)
        # stats = dict(stats)
        # for digit in stats:
        #     stats[digit] /= n_samples
        #     mse = (stats[digit] - benford2[int(digit) - 1]) ** 2
        #     err_benford[digit] = mse
        #     dataset.append([100, n_samples, digit, mse, stats[digit]])

        for d in range(n_dims):
            err_benford = {}
            stats = Counter(sample[:, d])
            stats = dict(stats)
            for digit in stats:
                stats[digit] /= n_samples
                mse = (stats[digit] - benford[int(digit) - 1]) ** 2
                err_benford[digit] = mse
                dataset.append([d, n_samples, digit, mse, stats[digit]])

    dataset = pd.DataFrame(dataset, columns=['dim', 'Ns', 'digit', 'SE',
                                             'Pr(d)'])
    datasets.append(dataset)

df_concat = pd.concat(datasets)

rmse = lambda x: (np.sum(x) / n_conv) ** 0.5
df_concat = df_concat.groupby(['dim', 'Ns', 'digit']).aggregate({'SE': rmse,
                                                                 'Pr(d)': 'mean'})
df_concat.columns = [r'$\epsilon$', "Pr(d)"]
dataset = df_concat.reset_index()

# Visualization
fname = 'Sobol/' if sobol else 'MC/'
fig, ax = plt.subplots()
sns.lineplot(x=np.array(digits) - 1, y=benford, lw=5, label='Benford', ax=ax)
sns.lineplot(x="digit", y="Pr(d)",
             hue="dim",
             ci=99,
             data=dataset[dataset['Ns'] == 32], ax=ax)
fig.savefig(fname + 'p_benford_ns_32.pdf', bbox_inches='tight', transparent=True)


fig, ax = plt.subplots()
sns.lineplot(x=np.array(digits) - 1, y=benford, lw=5, label='Benford', ax=ax)
sns.lineplot(x="digit", y="Pr(d)",
             hue="dim",
             ci=99,
             data=dataset[dataset['Ns'] == 4096], ax=ax)
fig.savefig(fname + 'p_benford_ns_4096.pdf', bbox_inches='tight',
            transparent=True)


fig, ax = plt.subplots()
sns.scatterplot(x="digit", y=r'$\epsilon$',
                hue="dim", size="Ns",
                # palette="ch:r=-.2,d=.3_r", linewidth=0,
                sizes=(20, 200),
                data=dataset, ax=ax)
fig.savefig(fname + 'scatter_global.pdf', bbox_inches='tight', transparent=True)

fig, ax = plt.subplots()
sns.boxplot(x="digit", y=r'$\epsilon$',
            data=dataset, ax=ax)
fig.savefig(fname + 'boxplot_global.pdf', bbox_inches='tight', transparent=True)

fig, ax = plt.subplots()
sns.scatterplot(x="digit", y=r'$\epsilon$', alpha=0.8,
                hue="dim", size="dim",
                sizes=(20, 200),
                data=dataset[dataset['Ns'] == 4096], ax=ax)
fig.savefig(fname + 'scatter_ns_4096.pdf', bbox_inches='tight', transparent=True)

# fig, ax = plt.subplots()
# swarm = sns.swarmplot(x="digit", y=r'$\epsilon$', hue='dim', size=3,
#                       data=dataset[dataset['Ns'] == 4096],
#                       palette="flare", ax=ax)
#
# handles, labels = swarm.get_legend_handles_labels()
# ax.legend(handles[::10], labels[::10])
# fig.savefig(fname + 'swarm_ns_4096.pdf', bbox_inches='tight', transparent=True)
