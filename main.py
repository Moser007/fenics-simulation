# Parameters of the material (example for Alumina)
material_properties = {
    'Alumina': {
        'rho': 3960,          # Density (kg/m³)
        'cp': 880,            # Specific heat capacity (J/(kg·K))
        'k': 30,              # Thermal conductivity (W/(m·K))
        'E': 370e9,           # Young's modulus (Pa)
        'nu': 0.22,           # Poisson's ratio
        'alpha': 8.5e-6       # Thermal expansion coefficient (/K)
    },
    # Add other materials as needed
}

# Choose the material for simulation
material_name = 'Alumina'
props = material_properties[material_name]

# Dimensions of the piece
length = 0.01       # Length (m)
thickness = 0.001   # Thickness (m)

# Mesh
nx = 50
ny = 10
mesh = RectangleMesh(Point(0, 0), Point(length, thickness), nx, ny)

# Function space for temperature
V = FunctionSpace(mesh, 'P', 1)

# Initial and boundary conditions
T0 = Constant(300)   # Initial temperature (K)
T_hot = Constant(623)  # Temperature at the heated face (K)

# Define subdomains for boundary conditions
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)

class BottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0) and on_boundary

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], thickness) and on_boundary

bottom_boundary = BottomBoundary()
top_boundary = TopBoundary()

bottom_boundary.mark(boundary_markers, 1)
top_boundary.mark(boundary_markers, 2)

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

# Define boundary conditions
bc_bottom = DirichletBC(V, T_hot, bottom_boundary)
bc_top = DirichletBC(V, T0, top_boundary)
bcs = [bc_bottom, bc_top]

# Functions and variables
u = TrialFunction(V)
v = TestFunction(V)
u_n = Function(V)       # Solution at previous time step
u_n.interpolate(T0)

# Time parameters
dt = 0.01                # Time step (s)
t_end = 1.0              # Total simulation time (s)
num_steps = int(t_end / dt)

# Material properties
rho = props['rho']
cp = props['cp']
k = props['k']

# Define the variational form
F = rho * cp * (u - u_n) / dt * v * dx + k * dot(grad(u), grad(v)) * dx

# Compile the variational form
a, L = lhs(F), rhs(F)

# Function to store the solution
u = Function(V)

# Time-stepping loop
t = 0
for n in range(num_steps):
    t += dt
    # Solve
    solve(a == L, u, bcs)
    
    # Update previous solution
    u_n.assign(u)
    
    # (Optional) Real-time visualization
    # plot(u)
    # plt.pause(0.01)

# Final temperature visualization
plt.figure()
p = plot(u)
plt.colorbar(p)
plt.title('Temperature at t = {} s'.format(t_end))
plt.xlabel('Length (m)')
plt.ylabel('Thickness (m)')
plt.show()

# Thermal stress analysis

# Function space for displacements
V_u = VectorFunctionSpace(mesh, 'P', 1)

# Mechanical parameters
E = props['E']
nu = props['nu']
alpha = props['alpha']
lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

# Define the variational form for elasticity
def epsilon(u):
    return sym(grad(u))

def sigma(u_disp, T):
    return lambda_ * tr(epsilon(u_disp)) * Identity(2) + 2 * mu * epsilon(u_disp) - \
           2 * mu * alpha * (u - T0) * Identity(2)

u_disp = TrialFunction(V_u)
v_disp = TestFunction(V_u)

a_elastic = inner(sigma(u_disp, u), epsilon(v_disp)) * dx
L_elastic = Constant((0, 0)) * v_disp * dx

# Apply mechanical boundary conditions (fix displacements on one edge)
def left_boundary(x, on_boundary):
    return near(x[0], 0) and on_boundary

bc_disp = DirichletBC(V_u, Constant((0, 0)), left_boundary)

# Solve the elastic problem
u_sol = Function(V_u)
solve(a_elastic == L_elastic, u_sol, bc_disp)

# Compute stresses
stress = sigma(u_sol, u)
V_s = TensorFunctionSpace(mesh, 'P', 1)
stress_proj = project(stress, V_s)

# Visualization of stresses (e.g., von Mises stress)
from ufl import sqrt, inner
s = stress_proj - (1./3) * tr(stress_proj) * Identity(2)  # Deviatoric stress tensor
von_Mises = sqrt(3./2 * inner(s, s))
V_mises = FunctionSpace(mesh, 'P', 1)
von_Mises_proj = project(von_Mises, V_mises)

plt.figure()
p = plot(von_Mises_proj)
plt.colorbar(p)
plt.title('Von Mises Stress (Pa)')
plt.xlabel('Length (m)')
plt.ylabel('Thickness (m)')
plt.show()
