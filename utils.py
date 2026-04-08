import itertools
import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lrux
import quantax as qtx
# import torch


# class to implement the Hubbard Hamiltonian in real space
class HamiltonianOperator:
    def __init__(self, L: int, n: int, diff: int, t: int, U: int):
        self.L = L
        self.n = n
        self.diff = diff
        self.t = t
        self.U = U

        # build H
        self.H = self.build_H().astype(np.complex64)

    def generate_basis(self):
        # diff = n_spin_up - n_spin_down
        # diff and n should have the same parity
        n_spin_up = (self.n + self.diff) // 2
        n_spin_down = (self.n - self.diff) // 2

        positions_up = []
        positions_down = []
        for i, j in itertools.product(range(self.L), repeat=2):
            positions_up.append((i, j, 1))
            positions_down.append((i, j, 0))

        basis = []
        for up, down in itertools.product(
            itertools.combinations(positions_up, n_spin_up),
            itertools.combinations(positions_down, n_spin_down),
        ):
            state = [*up, *down]

            # sort it
            basis.append(
                sorted(state, key=lambda x: ((2 * self.L) * x[0]) + (2 * x[1]) + x[2])
            )
        # lookup table
        self.state_to_index = {tuple(state): idx for idx, state in enumerate(basis)}
        # convert basis to PyTorch Tensor
        self.basis = np.array(basis, dtype=np.int32)

    # function to implement the diagonal interaction term
    def U_interaction(self, state: tuple):
        # state should be tuple of tuples containing (x, y, \sigma)
        # loop over all lattice sites
        count = 0
        for i_x, i_y in itertools.product(range(self.L), repeat=2):
            if sum((pos[0] == i_x and pos[1] == i_y) for pos in state) == 2:
                count += 1
        return self.U * count

    # helper "creation"/annihilation operators
    def c_op(self, pos: tuple, state: tuple):
        index = state.index(pos)
        return index, state[:index] + state[index + 1 :]

    def c_op_dagger(self, pos: tuple, state: tuple):
        state = tuple(
            sorted(
                state + (pos,), key=lambda x: ((2 * self.L) * x[0]) + (2 * x[1]) + x[2]
            )
        )
        return state.index(pos), state

    # the function to build the matrix representation of the Hamiltonian in real space
    def build_H(self):
        self.generate_basis()
        H = np.zeros((self.basis.shape[0], self.basis.shape[0]))
        # loop over all the basis states
        for state, idx in self.state_to_index.items():
            # first compute the diagonal entries
            H[idx, idx] += self.U_interaction(state)
            # now for the off-diagonal entries
            for i_x, i_y, sigma in itertools.product(
                range(self.L), range(self.L), (0, 1)
            ):
                # calculate adjacent lattice site
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for dx, dy in directions:
                    j_x = (i_x + dx) % self.L
                    j_y = (i_y + dy) % self.L
                    i_pos = (i_x, i_y, sigma)
                    j_pos = (j_x, j_y, sigma)
                    # to track the phase
                    exponent = 0
                    # first term
                    if j_pos not in state:
                        continue
                    # annihilate j
                    temp, new_state = self.c_op(j_pos, state)
                    exponent += temp
                    if i_pos in new_state:
                        continue
                    # create i
                    temp, new_state = self.c_op_dagger(i_pos, new_state)
                    exponent += temp
                    # calculate phase
                    if exponent % 2 == 0:
                        phase = 1
                    else:
                        phase = -1
                    # get index
                    H[idx, self.state_to_index[new_state]] += phase * (-self.t)
        return H
