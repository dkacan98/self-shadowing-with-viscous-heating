import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid as cumtrapz

# Physical constants (cgs)
k_B = 1.380649e-16   # erg/K
G = 6.67430e-8       # cm^3/g/s^2
m_H = 1.6735575e-24  # g
sigma = 5.670374419e-5  # erg/s/cm^2/K^4

def solve_vertical_equilibrium(R_vals, theta_vals, T_rtheta, Sigma_R, M_star, mu):
    """
    Solves vertical hydrostatic equilibrium at each R using full T(R,theta),
    in cylindrical coordinates, and returns rho(R, theta).

    Parameters:
        R_vals      : (nr,) array of radial positions in cm
        theta_vals  : (ntheta,) array of polar angles in radians
        T_rtheta    : (nr, ntheta) array of temperature in K
        Sigma_R     : (nr,) array of surface densities in g/cm^2
        M_star      : stellar mass in grams
        mu          : mean molecular weight

    Returns:
        rho_rtheta  : (nr, ntheta) array of densities in g/cm^3
    """
    nr = len(R_vals)
    ntheta = len(theta_vals)
    rho_rtheta = np.zeros((nr, ntheta))

    for i in range(nr):
        R = R_vals[i]
        theta_grid = theta_vals
        T_theta = T_rtheta[i, :]

        # Convert theta to z
        z_grid = R * np.cos(theta_grid)

        # Sort by increasing z (required for integration)
        sort_idx = np.argsort(z_grid)
        z_sorted = z_grid[sort_idx]
        T_sorted = T_theta[sort_idx]

        # Compute dT/dz
        dT_dz = np.gradient(T_sorted, z_sorted)

        # Integrand for dln(rho)/dz
        integrand = - (dT_dz + (G * M_star * mu * m_H * z_sorted) / (k_B * (R**2 + z_sorted**2)**(3/2))) / T_sorted
        ln_rho = cumtrapz(integrand, z_sorted, initial=0.0)
        rho = np.exp(ln_rho)

        # Normalize to match Sigma(R)
        Sigma_calc = np.trapz(rho, z_sorted)
        rho_scaled = rho * (Sigma_R[i] / Sigma_calc)

        # Interpolate rho(z) back to original theta grid
        z_to_rho = interp1d(z_sorted, rho_scaled, bounds_error=False, fill_value=0.0)
        z_original = R * np.cos(theta_grid)
        rho_theta = z_to_rho(z_original)

        rho_rtheta[i, :] = rho_theta

    return rho_rtheta
