import numpy as np
import matplotlib.pyplot as plt

space_1 = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
space_1 = (2.0 * space_1 - 1.0) / (2.0 * 6.0)
space_2 = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [6, 6]])
space_2 = (2.0 * space_2 - 1.0) / (2.0 * 6.0)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

for i, sample in enumerate([space_1, space_2]):
    axs[i].scatter(sample[:, 0], sample[:, 1])
    axs[i].set_aspect('equal')
    axs[i].set_xlabel(r'$x_1$')
    axs[i].set_ylabel(r'$x_2$')
    axs[i].grid()
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_xlim(0, 1)
    axs[i].set_ylim(0, 1)

plt.tight_layout()
# plt.show()
fig.savefig('lhs.pdf',
            transparent=True, bbox_inches='tight')
