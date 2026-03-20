import itertools
import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
from flax import nnx
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

# now implement the NN ansatz

# generic MLP helper class
class MLP(nnx.Module):
    def __init__(
        self,
        hidden_layers: int,
        input_dim: int,
        dim_feedforward: int,
        output_dim: int,
        rngs: nnx.Rngs,
        activation=nnx.gelu,
    ):
        # make it
        net = [
            nnx.Linear(input_dim, dim_feedforward, rngs=rngs),
            activation,
        ]
        for _ in range(hidden_layers):
            net.append(nnx.Linear(dim_feedforward, dim_feedforward, rngs=rngs))
            net.append(activation)
        net.append(nnx.Linear(dim_feedforward, output_dim, rngs=rngs))
        self.MLP = nnx.Sequential(*net)

    def __call__(self, x: jax.Array):
        return self.MLP(x)


# quick helper function
def view_as_complex(x: jax.Array):
    return x[..., 0] + (1j * x[..., 1])


class log_psi_2D_spinful(nnx.Module):
    def __init__(
        self,
        L: int,
        n: int,
        hidden_layers: int,
        dim_feedforward: int,
        rngs: nnx.Rngs,
        activation=nnx.gelu,
    ):
        self.L = L
        self.n = n
        # weights, drawn from a Gaussian distribution (for now)
        self.w_s = nnx.Param(rngs.normal((2 * (L**2),)))
        # instantiate MLPs
        self.f = MLP(hidden_layers, 5, dim_feedforward, 2, rngs, activation)
        self.g_1_prime = MLP(hidden_layers, 1, dim_feedforward, 2, rngs, activation)
        self.g_2_prime = MLP(hidden_layers, 1, dim_feedforward, 2, rngs, activation)

    # antisymmetric factor $F: \widetilde{\Lambda}^n \rightarrow \mathbb{C}$
    def F_antisymmetric(self, x: jax.Array):
        # x should be integers of shape (batch_size, n, 3) with entries from \widetilde{\Lambda}
        # contains (x, y, \sigma)
        # convert spatial coords to complex numbers e^{2\pi i x/L}
        z_x = jnp.stack(
            [
                jnp.cos((2 * jnp.pi * x[:, :, 0]) / self.L),
                jnp.sin((2 * jnp.pi * x[:, :, 0]) / self.L),
                jnp.cos((2 * jnp.pi * x[:, :, 1]) / self.L),
                jnp.sin((2 * jnp.pi * x[:, :, 1]) / self.L),
                (2 * x[:, :, 2]) - 1,  # encode spin as ±1
            ],
            axis=-1,
        )  # should be (batch_size, n, 5) now
        # pass through f MLP
        f_x = view_as_complex(self.f(z_x))  # complex (batch_size, n)
        vandermonde_matrix = jax.vmap(partial(jnp.vander, increasing=True))(
            f_x
        )  # should be of shape (batch_size, n, n)
        # take the determinant
        sign, logabsdet = jnp.linalg.slogdet(vandermonde_matrix)
        return (
            sign,  # complex
            logabsdet,  # real
        )  # both should be vectors of shape (batch_size)

    def eta_symmetric(self, x: jax.Array):
        # x should be integers of shape (batch_size, n, 3) with entries from \widetilde{\Lambda}
        # contains (x, y, \sigma)
        # first flatten the last dimension with an injective map
        x = (
            ((2 * self.L) * (x[:, :, 0] % self.L))
            + (2 * (x[:, :, 1] % self.L))
            + x[:, :, 2]
        )
        N_s = (
            jax.nn.one_hot(x, num_classes=2 * (self.L**2)).sum(axis=1).astype("float32")
        )  # (batch_size, 2L^2)
        # take matrix-vector product with w_s
        eta = jnp.matmul(N_s, self.w_s)
        return eta  # should be a vector of shape (batch_size)

    def __call__(self, occ_num: jax.Array):
        # occ_num should be of shape (batch_size, 2*L^2) in an occupation-number basis
        # need to convert to shape (batch_size, n, 3) with entries from \widetilde{\Lambda}
        _, nonzero_indices = jax.lax.top_k(occ_num, k=self.n, axis=-1)
        y_indices = nonzero_indices % self.L
        temp = nonzero_indices // self.L
        x_indices = temp % self.L
        spin_indices = temp // self.L
        x = jnp.stack([x_indices, y_indices, spin_indices], axis=-1)
        # now contains (x, y, \sigma)
        # compute the antisymmetric function F
        sign, logabsdet = self.F_antisymmetric(
            x
        )  # complex/real vectors of shape (batch_size)
        # compute the symmetric function g
        # first calculate eta
        eta = jnp.expand_dims(
            self.eta_symmetric(x), axis=-1
        )  # shape (batch_size, 1), dtype=float
        # now combine them together via:
        # \Psi = F_1 g_1 + F_2 g_2
        g_1 = view_as_complex(self.g_1_prime(eta))  # shape (batch_size), dtype=complex
        g_2 = view_as_complex(self.g_2_prime(eta))
        # natural log for the final result
        log_psi = logabsdet + jnp.log(
            (sign.real * g_1) + (sign.imag * g_2)
        )  # shape (batch_size), dtype=complex
        return log_psi

# other utilities
def count_params(model):
    params = nnx.state(model, nnx.Param)
    leaves = jax.tree_util.tree_leaves(params)
    return sum(x.size for x in leaves)