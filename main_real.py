import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
import quantax as qtx
from utils import HFPS, get_J_matrix

# choose real or complex mode
DTYPE = jnp.float32
print(f"{DTYPE=}")

# Model parameters
L = 4
n = 10
diff = 0
t = 1.0
U = 4.0

# diff = n_spin_up - n_spin_down
# diff and n should have the same parity
n_spin_up = (n + diff) // 2
n_spin_down = (n - diff) // 2

# Exact Diagonalization

lattice = qtx.sites.Square(
    L,
    particle_type=qtx.PARTICLE_TYPE.spinful_fermion,
    Nparticles=(n_spin_up, n_spin_down),
)
print(f"{lattice.shape=}")

# define the Hamiltonian
H = qtx.operator.Hubbard(U, t)

# VMC Optimization

# The ED energy for this model
E = -1.223808595 * lattice.Nsites
print(f"ED energy = {E}")

# do Fermion mean-field first
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

state = qtx.state.Variational(model, max_parallel=2048)
sampler = qtx.sampler.ParticleHop(state, 2048)
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
state.save("/blue/yujiabin/awwab.azam/NN_Discrete/NN_Discrete/models/Hubbard_real_test.eqx")