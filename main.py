import netket as nk
from flax import nnx
from netket.experimental.operator import FermiHubbardJax
from utils import log_psi_2D_spinful, count_params

# Model parameters
L = 12
n = 120
diff = 0
t = 1.0
U = 4.0

# diff = n_spin_up - n_spin_down
# diff and n should have the same parity
n_spin_up = (n + diff) // 2
n_spin_down = (n - diff) // 2

# make the lattice
lattice = nk.graph.Square(L, pbc=True)
num_lattice_sites = lattice.n_nodes
print(f"{num_lattice_sites = }")

# make the Hilbert Space
hi = nk.hilbert.SpinOrbitalFermions(
    num_lattice_sites, s=1 / 2, n_fermions_per_spin=(n_spin_down, n_spin_up)
)

# define the Hamiltonian
H = FermiHubbardJax(hilbert=hi, t=t, U=U, graph=lattice)
print(f"{H.max_conn_size=}")

# Instantiate the NN model
model = log_psi_2D_spinful(L, n, hidden_layers=3, dim_feedforward=128, rngs=nnx.Rngs(0))
print(model)

# Define a Metropolis exchange sampler
exchange_graph = nk.graph.disjoint_union(lattice, lattice)
print(f"Exchange graph size: {exchange_graph.n_nodes}")
sa = nk.sampler.MetropolisExchange(hi, graph=exchange_graph)
print(sa)

# Define an optimizer
op = nk.optimizer.Sgd(learning_rate=0.01)

# Create a variational state
vstate = nk.vqs.MCState(sa, model, n_samples=512, n_discard_per_chain=100)

# Create a Variational Monte Carlo with SR driver
gs = nk.driver.VMC_SR(
    H,
    op,
    variational_state=vstate,
    diag_shift=0.1,
    linear_solver=nk.optimizer.solver.pinv_smooth,
)

# Construct the logger to visualize the data later on
NN_log = nk.logging.RuntimeLog()

# Run the optimization for 500 iterations
gs.run(n_iter=500, out=NN_log)

E_gs = -1.0751 * num_lattice_sites
sd_energy = vstate.expect(H)
error = abs((sd_energy.mean - E_gs) / E_gs)

print(f"Optimized energy : {sd_energy}")
print(f"Relative error   : {error}")