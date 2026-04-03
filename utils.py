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

# now implement the NN ansatz

# ### Hidden Fermion Pfaffian State (HFPS)

# %%
class CNN_Block(eqx.Module):
    """Residual convolution block"""

    conv1: eqx.nn.Conv
    conv2: eqx.nn.Conv
    # static because we use it for an if-else condition
    block_idx: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        block_idx: int,
        total_blocks: int,
        dtype=jnp.float32,
    ):
        lattice = qtx.get_lattice()  # should be of shape (1, L, L)
        # periodic boundary conditions
        padding_mode = "CIRCULAR"

        def new_layer(is_first_layer: bool, is_last_layer: bool) -> eqx.nn.Conv:
            if is_first_layer:
                in_channels = lattice.shape[0]  # should be 1
                if lattice.particle_type == qtx.PARTICLE_TYPE.spinful_fermion:
                    in_channels *= 2  # so first layer has in_channels = 2*1 = 2
            else:
                in_channels = channels
            # should generate a new key each time
            key = qtx.get_subkeys()
            conv = eqx.nn.Conv(
                num_spatial_dims=lattice.ndim,  # should be 2
                in_channels=in_channels,
                out_channels=channels,  # out_channels is always the same
                kernel_size=kernel_size,
                padding="SAME",
                use_bias=not is_last_layer,  # i.e. bias for every layer except the last one
                padding_mode=padding_mode,
                dtype=dtype,
                key=key,
            )
            # Notes: stride = dilation = groups = 1 by default
            conv = qtx.nn.apply_he_normal(key, conv)
            return conv

        self.conv1 = new_layer(block_idx == 0, False)
        self.conv2 = new_layer(False, block_idx == (total_blocks - 1))
        self.block_idx = block_idx

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x.copy()
        x /= jnp.sqrt(self.block_idx + 1)

        if self.block_idx == 0:
            x /= jnp.sqrt(2)
        else:
            x = jax.nn.gelu(x)
        x = self.conv1(x)
        x = jax.nn.gelu(x)
        x = self.conv2(x)

        if x.shape[0] > residual.shape[0]:
            residual = jnp.repeat(residual, x.shape[0] // residual.shape[0], axis=0)
        return x + residual


class CNN(qtx.nn.Sequential):
    """Deep convolutional residual network."""

    nblocks: int
    channels: int
    kernel_size: int
    layers: Tuple[Callable, ...]
    dtype: jnp.dtype
    out_dtype: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        nblocks: int,
        channels: int,
        kernel_size: int,
        dtype=jnp.float32,
        out_dtype=jnp.float32,
    ):
        self.nblocks = nblocks
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.out_dtype = out_dtype

        # check complex/real stuff
        pair_cpl = False
        if jnp.isdtype(dtype, "complex floating") and jnp.isdtype(
            out_dtype, "complex floating"
        ):
            holomorphic = True
        else:
            holomorphic = False
            if jnp.isdtype(out_dtype, "complex floating"):
                pair_cpl = True
                channels *= 2
        self.channels = channels

        blocks = [
            CNN_Block(channels, kernel_size, i, nblocks, dtype) for i in range(nblocks)
        ]

        def final_layer(x):
            x /= jnp.sqrt(nblocks + 1)
            if pair_cpl is True:
                x = qtx.nn.pair_cpl(x)
            x = x.astype(out_dtype)
            return x

        layers = [qtx.nn.ReshapeConv(dtype), *blocks, final_layer]

        super().__init__(layers, holomorphic)

# %%
class HFPS(eqx.Module):
    CNN: Callable
    N_tilde_x2: int
    M: int
    N_occ: int
    F_vv: Union[jax.Array, Tuple[jax.Array, jax.Array]]
    F_hh: Union[jax.Array, Tuple[jax.Array, jax.Array]]
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        n_hidden_fermions: int,
        num_blocks: int,
        kernel_size: int,
        jastrow_channels: int,
        N_occ: int,
        F_vv: jax.Array = None,
        dtype=jnp.float32,
    ):
        assert (N_occ + n_hidden_fermions) % 2 == 0, (
            "Pfaffian requires an even-dimensional matrix!"
        )
        self.N_tilde_x2 = 2 * n_hidden_fermions
        channels = self.N_tilde_x2 + jastrow_channels  # factor of 2 for spin
        self.CNN = CNN(
            num_blocks, channels, kernel_size, dtype=jnp.float32, out_dtype=dtype
        )
        self.M = 2 * qtx.get_lattice().Nsites  # should be 2L^2
        self.N_occ = N_occ
        self.dtype = dtype
        if jnp.isdtype(dtype, "complex floating"):
            sqrt_2 = jnp.sqrt(2)
            real_dtype = jnp.array(0, dtype).real.dtype
            complex_mode = True
        else:
            complex_mode = False

        if F_vv is None:
            vv_key_1 = qtx.get_subkeys()
            if complex_mode is True:
                vv_key_2 = qtx.get_subkeys()
                self.F_vv = (
                    jr.normal(vv_key_1, (self.M, self.M), real_dtype) / sqrt_2,
                    jr.normal(vv_key_2, (self.M, self.M), real_dtype) / sqrt_2,
                )
            else:
                self.F_vv = jr.normal(vv_key_1, (self.M, self.M), dtype)
        else:
            if complex_mode is True:
                self.F_vv = (F_vv.real.astype(real_dtype), F_vv.imag.astype(real_dtype))
            else:
                self.F_vv = F_vv.real.astype(dtype)

        hh_key_1 = qtx.get_subkeys()
        if complex_mode is True:
            hh_key_2 = qtx.get_subkeys()
            self.F_hh = (
                jr.normal(hh_key_1, (n_hidden_fermions, n_hidden_fermions), real_dtype)
                / sqrt_2,
                jr.normal(hh_key_2, (n_hidden_fermions, n_hidden_fermions), real_dtype)
                / sqrt_2,
            )
        else:
            self.F_hh = jr.normal(
                hh_key_1, (n_hidden_fermions, n_hidden_fermions), dtype
            )

    def __call__(self, x: jax.Array):
        # x has entries ±1 representing occupied/unoccupied orbitals
        CNN_output = self.CNN(x)  # should be of shape (2*N^tilde + k, L, L)
        # occupation number
        n = jnp.nonzero(x == 1, size=self.N_occ)[0]
        log_J_n = CNN_output[self.N_tilde_x2 :, :, :].sum()
        F_vh = CNN_output[: self.N_tilde_x2, :, :].reshape(-1, self.M).T
        # convert to complex if needed
        if jnp.isdtype(self.dtype, "complex floating"):
            F_vv = (self.F_vv[0] + (1j * self.F_vv[1])).astype(self.dtype)
            F_hh = (self.F_hh[0] + (1j * self.F_hh[1])).astype(self.dtype)
        else:
            F_vv = self.F_vv
            F_hh = self.F_hh
        # antisymmetrize F_vv and F_hh
        # NOTE: can optimize order-of-ops further
        F_vv = 0.5 * (F_vv - F_vv.T)
        F_vv = F_vv[n, :][:, n]
        F_hh = 0.5 * (F_hh - F_hh.T)
        # take the pfaffian
        pfaffian_matrix = jnp.block([[F_vv, F_vh[n, :]], [-F_vh.T[:, n], F_hh]])
        sign, logabspf = lrux.slogpf(pfaffian_matrix)
        # NOTE: may have to change J_n logic if observe numerical instability
        # return (
        #     qtx.utils.LogArray(sign, logabspf)
        #     * jnp.exp(log_J_n)
        #     * qtx.nn.fermion_inverse_sign(x)
        # )
        return (
            qtx.utils.LogArray(jnp.cos(jnp.angle(sign) + log_J_n.imag), logabspf + log_J_n.real)
            * qtx.nn.fermion_inverse_sign(x)
        )   # change to output only real values

# turn it into a MxM matrix
def get_J_matrix(N_occ: int) -> jax.Array:
    assert N_occ % 2 == 0, "N_occ must be even to have a non-zero Pfaffian!"

    # example of a 2x2 antisymmetric matrix with pfaffian=1
    base_block = jnp.array([[0.0, 1.0], [-1.0, 0.0]])

    # make an identity matrix
    identity = jnp.eye(N_occ // 2)

    # Kronecker product to repeat along the diagonal
    J = jnp.kron(identity, base_block)

    return J