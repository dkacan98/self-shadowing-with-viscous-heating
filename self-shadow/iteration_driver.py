import numpy as np
import subprocess
from disk_single_run import radmc3d
from vertical_hydrostatics import solve_vertical_equilibrium

# === Parameters ===
dirname = "/home/dkacan/research/radmc3d-2.0-master/self-shadow/iterations/"
max_iter = 10
tol = 1e-2

# === Step 1: Initialize model ===
model = radmc3d(dirname_out=dirname)  # sets up grid, density, opacities, etc.
model.paramsinit()
model.make_wavelengths()
model.grid_edge_center()
model.getR_Theta()
model.density_calc()
model.heatrate_per_volume_calc()
model.write_grid()
model.write_heatrate()
model.write_density()
model.write_wavelengths()
model.write_starsinp()
model.run_optool()
model.write_dustopac()
model.write_radmc3dinp()

R_vals = model.rr[:, 0, 0]            # shape (nr,)
theta_vals = model.tt[0, :, 0]    # shape (ntheta,)    #gotta change this line
Sigma_R = model.surface_density                # surface density per R 
M_star = model.mstar   #in grams
mu = model.mu   #2.3

# === Step 2: Iteration loop ===
for i in range(max_iter):
    print(f"\n=== Iteration {i+1} ===",flush=True)

    # Run RADMC-3D radiative transfer
    subprocess.run(f"cd {dirname} && radmc3d mctherm", shell=True)

    # Read updated temperature
    T_rtheta = model.read_temperature(f"{dirname}/dust_temperature.dat")   #gotta check this structure (200,150) shape

    # Solve vertical hydrostatic equilibrium     #rho_new (r,theta)
    rho_new = solve_vertical_equilibrium(
        R_vals=R_vals,
        theta_vals=theta_vals,
        T_rtheta=T_rtheta,
        Sigma_R=Sigma_R,
        M_star=M_star,
        mu=mu
    )

    # Check convergence
    model_rho=model.rhod[0].squeeze()   #(200,100,1)--->(200,100)
    delta_rho = np.max(np.abs((rho_new - model_rho) / model_rho))
    print(f"Max fractional density change: {delta_rho:.3e}",flush=True)
    if delta_rho < tol:
        print("Converged.",flush=True)
        break

    # Update density and write new input files
    model.rhod[0] = rho_new[:,:,None] #(we make rho shape as 200,100,1)
    model.write_density()
    model.heatrate_per_volume_calc()
    model.write_heatrate()

print("\n✅ Iteration complete. Final model written to:", dirname)
