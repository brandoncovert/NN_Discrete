import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
import quantax as qtx
from utils import HFPS, get_J_matrix
import json
from quantax.operator import number_u, number_d

# choose real or complex mode
DTYPE = jnp.float32
print(f"{DTYPE=}")

# Model parameters
with open("/blue/yujiabin/awwab.azam/NN_Discrete/NN_Discrete/params.json", "r") as f:
    data = json.load(f)

L=data["L"]
n=data["n"]
diff=data["diff"]
t=data["t"]
U=data["U"]
E_per_site = data["E_per_site"]

# diff = n_spin_up - n_spin_down
# diff and n should have the same parity
n_spin_up = (n + diff) // 2
n_spin_down = (n - diff) // 2

lattice = qtx.sites.Square(
    L,
    particle_type=qtx.PARTICLE_TYPE.spinful_fermion,
    Nparticles=(n_spin_up, n_spin_down),
)
print(f"{lattice.shape=}")

# define the Hamiltonian
H = qtx.operator.Hubbard(U, t)

# VMC Optimization

# The energy for this model
E = E_per_site * lattice.Nsites
print(f"AFQMC energy = {E}")

# do Fermion mean-field first
if U < 0:
    print(f"Attractive Hubbard model detected. Initializing with BCS state...")

    # chemical potential
    mu = -2
    opN = sum(number_u(i) + number_d(i) for i in range(lattice.Nsites))
    H_attractive = H - (mu * opN)

    bcs_state = qtx.state.SingletPairState(qtx.model.SingletPair(dtype=jnp.float32, out_dtype=DTYPE))
    params, static = eqx.partition(bcs_state.model, eqx.is_inexact_array)
    loss_fn = bcs_state.get_loss_fn(H_attractive)
    solver = optx.BFGS(1e-8, 1e-12)
    out = optx.minimise(loss_fn, solver, params, max_steps=10000)
    model = eqx.combine(out.value, static)
    bcs_state = qtx.state.SingletPairState(model)
    print(f"Particle number: {bcs_state.expectation(opN)}")
    print(f"Mean-field energy: {bcs_state.expectation(H)}")

    # Extract the 64x64 spatial pairing matrix
    f_spatial = bcs_state.model.F_full

    # Build the 128x128 antisymmetric spin-orbital matrix
    zeros = jnp.zeros_like(f_spatial)
    F_vv = jnp.block([
        [zeros, f_spatial],
        [-f_spatial.T, zeros]
    ])
else:
    print(f"Repulsive Hubbard model detected. Initializing with Slater determinant state...")
    det_state = qtx.state.GeneralDetState(qtx.model.GeneralDet(dtype=jnp.float32, out_dtype=DTYPE))
    params, static = eqx.partition(det_state.model, eqx.is_inexact_array)
    loss_fn = det_state.get_loss_fn(H)
    solver = optx.BFGS(1e-8, 1e-12)
    out = optx.minimise(loss_fn, solver, params, max_steps=10000)
    E_meanfield = loss_fn(out.value)
    model = eqx.combine(out.value, static)
    det_state = qtx.state.GeneralDetState(model)
    print(f"Mean-field energy: {E_meanfield}")

    J = get_J_matrix(n)
    u = det_state.model.U_full
    F_vv = u @ J @ u.T

# instantiate NN ansatz
model = HFPS(
    n_hidden_fermions=14,
    num_blocks=8,
    kernel_size=3,
    jastrow_channels=4,
    N_occ=n,
    F_vv=F_vv,
    dtype=DTYPE,
)

state = qtx.state.Variational(model, max_parallel=8192)
sampler = qtx.sampler.ParticleHop(state, 8192)
optimizer = qtx.optimizer.SR(state, H)

energy = qtx.utils.DataTracer()
VarE = qtx.utils.DataTracer()

training_rate = 0.01

for i in range(10_000):
    samples = sampler.sweep()
    step = optimizer.get_step(samples)
    state.update(step * training_rate)

    energy.append(optimizer.energy)
    VarE.append(optimizer.VarE)

    # Print progress to the .log file instead of the screen
    if i % 10 == 0:
        print(f"Step {i} | Energy: {optimizer.energy:.6f} | VarE: {optimizer.VarE:.6f}")

nqs_energy = energy[-1]
rel_err = jnp.abs((nqs_energy - E) / E)
print(f"NQS energy: {nqs_energy}, relative error: {rel_err}")

print(f"Number of parameters: {state.nparams}")

# Save the model for future use (maybe)
state.save(f"/blue/yujiabin/awwab.azam/NN_Discrete/NN_Discrete/models/Hubbard_L_{L}_n_{n}_U_{U}_t_{t}_diff_{diff}_real_test.eqx")