# Aug19,2020

import matplotlib.pyplot as plt
import numpy as np
import os

def Ts(s, sigma_g):
    """
    Get the dimensionless stopping time
    s: particle size in mm
    sigma_g: gas surface density in g cm^-2
    """
    dust_density = 1.675 # g/cm^3
    return 1.57e-3 * dust_density * s * (100./(sigma_g+1e-60))

def Hd_Hg(alpha, Ts):
    """
    Get the ratio between dust scale height and gas scale height
    """
    return  1. / (1. + Ts/alpha * (1.+2.*Ts)/(1.+Ts))**0.5 +1e-60

def Tmidr(rAU, Lstar_Lsun=1., phi=0.02):
    au = 1.496e+11
    Lsun = 3.828e26
    sigmaSB = 5.670367e-8
    Lstar = Lstar_Lsun * Lsun
    r = rAU * au
    Tmid = (phi*Lstar/(8.*np.pi*r**2*sigmaSB))**(0.25)
    return Tmid

def calculate_hr(T, mstar = 0.85, rgap = 1):
    
    G = 6.67259e-8       # in cgs
    ms  = 1.98892e33     # Solar mass              [g]
    au  = 1.49598e13     # Astronomical Unit       [cm]
    mmw = mu  =  2.3                # mean molecular weight 
    Rconst  =  8.31446261815324e7 # ideal gas constant in cgs
    
    Cs  = np.sqrt(Rconst/mmw*T)
    v_k = np.sqrt(G*(mstar*ms)/ (rgap*au))
    hr  = Cs/v_k
    return hr

def calculate_init_hr(Lstar = 0.35, mstar = 0.85, f = 0.1, rgap = 1):    #heigt at the first 
    
    G = 6.67259e-8       # in cgs
    ms  = 1.98892e33     # Solar mass              [g]
    au  = 1.49598e13     # Astronomical Unit       [cm]
    mmw = mu      =  2.3                # mean molecular weight 
    Rconst  =  8.31446261815324e7 # ideal gas constant in cgs
    
    T = Tmidr(rgap, Lstar, f)
    Cs  = np.sqrt(Rconst/mmw*T)
    v_k = np.sqrt(G*(mstar*ms)/ (rgap*au))
    hr  = Cs/v_k
    return hr

def write_density(rhod, params): 
    nr = params["nr"]; ntheta = params["ntheta"]; nphi = params["nphi"]; 
    nspecies = len(params["opacfile"])
    with open('dust_density.inp','w+') as f:
        f.write('1\n')                       # Format number
        f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
        f.write('{}\n'.format(nspecies))    # Nr of dust species
        
        for i in range(nspecies):
            data = rhod[i].ravel(order='F')    # Create a 1-D view, fortran-style indexing
            data.tofile(f, sep='\n', format="%13.6e")
            f.write('\n')

def read_density( params, filename = "dust_density.inp"):
    nr = params["nr"]; ntheta = params["ntheta"]; nphi = params["nphi"]; 
    ntotal = nr*ntheta*nphi
    nspecies = len(params["opacfile"])
    rhod = np.fromfile(filename, sep='\n', dtype=np.float64)[3:].\
    reshape((nr,ntheta,nphi, nspecies),order='F')[:,:,0,:]
    rhod = np.moveaxis(rhod, (0, 1, 2),  (1, 2, 0))
    return rhod

def read_temperature(params, filename = "dust_temperature.dat"):
    nr = params["nr"]; ntheta = params["ntheta"]; nphi = params["nphi"]; 
    ntotal = nr*ntheta*nphi
    temperature =  np.fromfile(filename, sep='\n',dtype=np.float64)[3:3+ntotal].\
    reshape((nr,ntheta,nphi),order='F')[:,:,0]
    return temperature

def get_ri_thetai_rc_thetac_phic(params):
    """
    return R and theta in meshgrid indexing='ij'
    theta is measured from the mid-plane
    """
    rin = params["rin"]
    rout = params["rout"]
    nr  = params["nr"]
    ntheta  = params["ntheta"]
    nphi  = params["nphi"]
    thetaup = params["thetaup"]

    ri       = np.logspace(np.log10(rin),np.log10(rout),nr+1)
    thetai   = np.linspace(thetaup,0.5e0*np.pi,ntheta+1)
    phii     = np.linspace(0.e0,np.pi*2.e0,nphi+1)
    rc       = 0.5 * ( ri[0:nr] + ri[1:nr+1] )
    thetac   = 0.5 * ( thetai[0:ntheta] + thetai[1:ntheta+1] )
    phic     = 0.5 * ( phii[0:nphi] + phii[1:nphi+1] )
   
    return ri, thetai, rc, thetac, phic


def init_write(params):
    """
    rin in au
    rout in au
    thetaup in radian
    
    sigmag0, sigma gas at R_c
    
    hr0 , h/r at 1 au!!!
    
    """
    
    nphot, nr, ntheta, nphi, rin, rout, thetaup, sigmag0, sigmad0, plsig, hr0, plh, R_c,\
    mstar, rstar, tstar, opacfile = params["nphot"], params["nr"], params["ntheta"],\
                        params["nphi"], params["rin"], params["rout"], params["thetaup"],\
                        params["sigmag0"], params["sigmad0"], params["plsig"], params["hr0"], params["plh"],\
                        params["R_c"], params["mstar"], params["rstar"],\
                        params["tstar"], params["opacfile"]
    
    
    # Some natural constants
    #
    au  = 1.49598e13     # Astronomical Unit       [cm]
    pc  = 3.08572e18     # Parsec                  [cm]
    ms  = 1.98892e33     # Solar mass              [g]
    ts  = 5.78e3         # Solar temperature       [K]
    ls  = 3.8525e33      # Solar luminosity        [erg/s]
    rs  = 6.96e10        # Solar radius            [cm]
    #
    
    rin      = rin * au
    rout     = rout* au
    
    mstar    = mstar * ms
    rstar    = rstar * rs
    pstar    = np.array([0.,0.,0.])
    
    
        # Make the coordinates
    
    ri, thetai, rc, thetac, phic =  get_ri_thetai_rc_thetac_phic(params)
    
    ri       = ri * au
    phii     = np.linspace(0.e0,np.pi*2.e0,nphi+1)
    rc       = rc * au
    #
    # Make the grid
    #
    qq       = np.meshgrid(rc,thetac,phic,indexing='ij')
    rr       = qq[0]
    tt       = qq[1]
    zr       = np.pi/2.e0 - qq[1]
    #
    # Make the dust density model
    
    
    nspecies = len(params["opacfile"])
    
    rhod = np.zeros((nspecies, nr, ntheta, nphi))
    
    if params["expcutoff"]:
        sigmag   = sigmag0 * (rr/au)**plsig * np.exp(-(rr/au/R_c)**params["s_exp"])
    else: 
        sigmag   = sigmag0 * (rr/au)**plsig
    
    if params["expinner"]: 
        sigmag   = np.exp(-1./(rr/au)) * sigmag
    
    # calculate the smallest dust coupled to the gas
    if params["expcutoff"]:
        sigmad   = sigmad0[0] * (rr/au)**plsig * np.exp(-(rr/au/R_c)**params["s_exp"])
    else:
        sigmad   = sigmad0[0] * (rr/au)**plsig
        
    if params["expinner"]: 
        sigmad   = np.exp(-1./(rr/au)) * sigmad
    
        
    if params["gaussianbump"]:
            sig = params["sig"]
            mu_peak  = params["mu_peak"]
            ampp = params["ampp"]

            bkg = sigmad
            bkg[rr/au < mu_peak] = 0
            amp  = ampp * np.max(bkg[rr/au >= mu_peak])
            gaussian = amp * np.exp(-0.5*((rr/au-mu_peak)/sig)**2)
            sigmad = np.max([gaussian, bkg], axis = 0)
            
    if params["gas_profile"]:
        sigmag = params["gas_profile"](rr/au)
        sigmad = params["dust_profile"][0](rr/au)
        
        
    hhr      = hr0 * (rr/au)**plh
    hh       = hhr * rr
    rhod[0]  = ( sigmad / (np.sqrt(2.e0*np.pi)*hh) ) * np.exp(-(zr**2/hhr**2)/2.e0)
    
    # calculate larger dusts decoupled to the gas
    alpha    = params["alpha"]
    for i in range(1, nspecies):
        dust_size = params["dustsizes"][i] # mm
        if params["expcutoff"]:
            sigmad   = sigmad0[i] * (rr/au)**plsig * np.exp(-(rr/au/R_c)**params["s_exp"])
        else:
            sigmad   = sigmad0[i] * (rr/au)**plsig
        
        if params["gaussianbump"]:
            sig = params["sig"]
            mu_peak  = params["mu_peak"]
            ampp = params["ampp"]

            bkg = sigmad
            bkg[rr/au < mu_peak] = 0
            amp  = ampp * np.max(bkg[rr/au >= mu_peak])
            gaussian = amp * np.exp(-0.5*((rr/au-mu_peak)/sig)**2)
            sigmad = np.max([gaussian, bkg], axis = 0)
        
        if params["expinner"]: 
            sigmad   = np.exp(-1./(rr/au)) * sigmad
            
        if params["gas_profile"]:
            sigmad = params["dust_profile"][i](rr/au)
         
        hhr      = hr0 * (rr/au)**plh * Hd_Hg(alpha, Ts(dust_size, sigmag))
        hh       = hhr * rr
        rhod[i]  = ( sigmad / (np.sqrt(2.e0*np.pi)*hh) ) * np.exp(-(zr**2/hhr**2)/2.e0)
        
        # if the disk is too thin (h/r < dtheta), then fill the grid closest to the midplane as an integral value
        from scipy.special import erf
        coeff = np.sqrt(np.pi/2.) * hhr[:,-1, :] * erf(np.cos(thetai[-2]) / (np.sqrt(2) * hhr[:,-1, :] ) ) / np.cos(thetai[-2])
        rhod[i][:,-1, :] = ( sigmad[:,-1, :] / (np.sqrt(2.e0*np.pi)*hh[:,-1, :]) ) * coeff
    
    #
    # Write the wavelength_micron.inp file
    #
    lam1     = 0.1e0
    lam2     = 7.0e0
    lam3     = 25.e0
    lam4     = 1.0e4
    n12      = 20
    n23      = 100
    n34      = 30
    lam12    = np.logspace(np.log10(lam1),np.log10(lam2),n12,endpoint=False)
    lam23    = np.logspace(np.log10(lam2),np.log10(lam3),n23,endpoint=False)
    lam34    = np.logspace(np.log10(lam3),np.log10(lam4),n34,endpoint=True)
    lam      = np.concatenate([lam12,lam23,lam34])
    nlam     = lam.size
    #
        #writes the wavelength file used to do the thermal mcmc. 
    #
    with open('wavelength_micron.inp','w+') as f:
        f.write('%d\n'%(nlam))
        for value in lam:
            f.write('%13.6e\n'%(value))
    #
    #
    # Write the stars.inp file
    #
    with open('stars.inp','w+') as f:
        f.write('2\n')
        f.write('1 %d\n\n'%(nlam))
        f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n'%(rstar,mstar,pstar[0],pstar[1],pstar[2]))
        for value in lam:
            f.write('%13.6e\n'%(value))
        f.write('\n%13.6e\n'%(-tstar))   
    #
    # Write the grid file
    #
    with open('amr_grid.inp','w+') as f:
        f.write('1\n')                       # iformat
        f.write('0\n')                       # AMR grid style  (0=regular grid, no AMR)
        f.write('100\n')                     # Coordinate system: spherical
        f.write('0\n')                       # gridinfo
        f.write('1 1 0\n')                   # Include r,theta coordinates
        f.write('%d %d %d\n'%(nr,ntheta,nphi))  # Size of grid     #nphi will be 1.
        for value in ri:
            f.write('%13.6e\n'%(value))      # r coordinates (cell walls)
        for value in thetai:
            f.write('%13.6e\n'%(value))      # theta coordinates (cell walls)
        for value in phii:
            f.write('%13.6e\n'%(value))      # phi coordinates (cell walls)
    #
    # Write the density file
    #
    write_density(rhod, params)
    
    #
    # Dust opacity control file
    # opactfile = "mix_2species_60silicates_40carbons_amax01"
	
    with open('dustopac.inp','w+') as f:
        f.write('2               Format number of this file\n')
        f.write('{}               Nr of dust species\n'.format(nspecies))
        f.write('============================================================================\n')
        for i in range(nspecies):
            f.write('1               Way in which this dust species is read\n')
            f.write('0               0=Thermal grain\n')
            f.write('{}     Extension of name of dustkappa_***.inp file\n'.format(opacfile[i]))
            f.write('----------------------------------------------------------------------------\n')

    #
    # Write the radmc3d.inp control file
    #
    with open('radmc3d.inp','w+') as f:
        f.write('nphot = %d\n'%(nphot))
        f.write('scattering_mode_max = 1\n')
        f.write('iranfreqmode = 1\n')
        f.write('itempdecoup = 0\n')
        f.write('modified_random_walk = {}\n'.format(int(params["MRW"])))
        
        
def getR_Theta(params):
    """
    return R and theta in meshgrid indexing='ij'
    theta is measured from the mid-plane
    """
    ri, thetai, rc, thetac, phic = get_ri_thetai_rc_thetac_phic(params)
    
    # Make the grid
    #
    qq       = np.meshgrid(rc,thetac,phic,indexing='ij')
    rr       = qq[0][:,:,0]
    tt       = qq[1][:,:,0]
    zr       = np.pi/2.e0 - qq[1][:,:,0]
    #
    return rr, zr

def iterate_density(params, temperature, rhod):
    """
    
    """
    #-----------------------------------------------------------------------------------
    au  = 1.49598e13     # Astronomical Unit       [cm]
    pc  = 3.08572e18     # Parsec                  [cm]
    ms  = 1.98892e33     # Solar mass              [g]
    ts  = 5.78e3         # Solar temperature       [K]
    ls  = 3.8525e33      # Solar luminosity        [erg/s]
    rs  = 6.96e10        # Solar radius            [cm]
    mu = 2.3
    Rconst = 8.31446261815324e7 # in cgs
    G = 6.67259e-8       # in cgs
    
    
    
    nspecies = len(params["opacfile"])
    
    #-------------------------preparations--------------------------------------------------
    
    nr = params["nr"]; ntheta = params["ntheta"]; nphi = params["nphi"];
    mstar = params["mstar"]
    new_rhod = np.zeros((nspecies, nr, ntheta))
    
    ri, thetai, rc, thetac, phic = get_ri_thetai_rc_thetac_phic(params)
    # surface density profile
    if params["expcutoff"]:
        sigmad   = params["sigmad0"][0] * (rc)**params["plsig"] * np.exp(-(rc/params["R_c"])**params["s_exp"])
    else:
        sigmad   = params["sigmad0"][0] * (rc)**params["plsig"]
        
    if params["expinner"]: 
        sigmad   = np.exp(-1./(rc)) * sigmad
        
    if params["gaussianbump"]:
        sig = params["sig"]
        mu_peak  = params["mu_peak"]
        ampp = params["ampp"]
        
        bkg = sigmad
        bkg[rc < mu_peak] = 0
        amp  = ampp * bkg[rc >= mu_peak][0]
        gaussian = amp * np.exp(-0.5*((rc-mu_peak)/sig)**2)
        sigmad = np.max([gaussian, bkg], axis = 0)
        
    if params["gas_profile"]:
        sigmad = params["dust_profile"][0](rc)
       
 
    sin_new_thetac = np.cos(thetac)       # change 0 point to midplane
    sinthetac_diff = -np.diff(np.cos(thetai))

    dtheta = np.diff(thetai[:2])[0]
    blockarea = (ri[1:]**2 - ri[:-1]**2)/2. * dtheta
    blocklength = np.diff(ri)
    
    R, Theta = getR_Theta(params)
    X = R*np.cos(Theta); Y = R*np.sin(Theta)
    ## surface_density = np.zeros(nr, dtype=np.float64)
    
    pressure = 1./mu*Rconst*rhod[0]*temperature     
    Cs = np.sqrt(Rconst*temperature/mu)
    Omega = np.sqrt(G*(mstar*ms)/(R*au)**3)
    scale_height = Cs/Omega/au        # in au 
    r_over_H    = R/scale_height
  
    
    #------------------------------------------------------------------------------------------
    for i in range(1,pressure.shape[1]):
        exponent = r_over_H[:,-i]**2 * sin_new_thetac[-i] * sinthetac_diff[-i]
        pressure[:, -i-1] = pressure[:, -i] * np.exp( - exponent)
        
    new_rhod[0] = pressure/(Rconst/mu*temperature)
    sigmad_new = surface_density(params, new_rhod[0])
    new_rhod[0] = new_rhod[0] * (sigmad / sigmad_new)[ : , None ]
    
    #for i in range(len(ri)-1):
    #    indices = np.where((ri[i] <= X) & (X < ri[i+1]))
    #    surface_density[i] = np.sum(new_rhod[0][indices]*blockarea[indices[0]])/blocklength[i]*au * 2.
    #    new_rhod[0][indices] = new_rhod[0][indices] * sigmad[i]/surface_density[i]
    #-------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------
    # Make the grid
    #
    qq       = np.meshgrid(rc,thetac,phic,indexing='ij')
    rr       = qq[0][:,:,0]
    tt       = qq[1][:,:,0]
    zr       = np.pi/2.e0 - qq[1][:,:,0]
    #
    if params["expcutoff"]:
        sigmag   = params["sigmag0"] * (rr)**params["plsig"] * np.exp(-(rr/params["R_c"])**params["s_exp"])
    else:
        sigmag   = params["sigmag0"] * (rr)**params["plsig"]
        
    if params["expinner"]: 
        sigmag   = np.exp(-1./(rr)) * sigmag
        
    if params["gas_profile"]:
        sigmag = params["gas_profile"](rr)
        
    temp_mid = np.mean(temperature[:,-10:], axis=1)
    temp_mid = np.repeat(temp_mid, ntheta).reshape(nr, ntheta)
    
    alpha    = params["alpha"]
    for i in range(1, nspecies):
        dust_size = params["dustsizes"][i] # mm
        if params["expcutoff"]:
            sigmad   = params["sigmad0"][i] * (rr)**params["plsig"] * np.exp(-(rr/params["R_c"])**params["s_exp"])
        else:
            sigmad   = params["sigmad0"][i] * (rr)**params["plsig"]
        
        if params["expinner"]: 
            sigmad   = np.exp(-1./ rr) * sigmad
       
        if params["gaussianbump"]:
            sig = params["sig"]
            mu_peak  = params["mu_peak"]
            ampp = params["ampp"]

            bkg = sigmad
            bkg[rr < mu_peak] = 0
            amp  = ampp * np.max(bkg[rr >= mu_peak])
            gaussian = amp * np.exp(-0.5*((rr-mu_peak)/sig)**2)
            sigmad = np.max([gaussian, bkg], axis = 0)
    
        
        if params["gas_profile"]:
            sigmad = params["dust_profile"][i](rr)
    
        
        hhr = calculate_hr(temp_mid, mstar=params["mstar"], rgap = rr)  * Hd_Hg(alpha, Ts(dust_size, sigmag))
        hh       = hhr * rr 
        new_rhod[i]  = ( sigmad / (np.sqrt(2.e0*np.pi)*hh*au) ) * np.exp(-(zr**2/hhr**2)/2.e0)
        from scipy.special import erf
        coeff = np.sqrt(np.pi/2.) * hhr[:,-1] * erf(np.cos(thetai[-2]) / (np.sqrt(2) * hhr[:,-1] ) ) / np.cos(thetai[-2])
        new_rhod[i][:,-1] = ( sigmad[:,-1] / (np.sqrt(2.e0*np.pi)*hh[:,-1]*au) ) * coeff
            
    
    
    return new_rhod

def total_mass(params, rhod):
    au  = 1.49598e13     # Astronomical Unit       [cm]
    ms  = 1.98892e33
    ri, thetai, rc, thetac, phic = get_ri_thetai_rc_thetac_phic(params)
    cubic_r = (ri[1:]**3 - ri[:-1]**3)         # rout**3 - rin**3
    costheta_d = -(np.cos(thetai[1:]) - np.cos(thetai[:-1])) #
    Cubic_r, Costheta_d = np.meshgrid(cubic_r, costheta_d, indexing='ij')
    blockvolume = 2./3. * np.pi * Cubic_r * Costheta_d # the formula in the website
    return np.sum(blockvolume * rhod) / ms * au**3  * 2


def surface_density(params, rhod):
    #-----------------------------------------------------------------------------------
    au  = 1.49598e13     # Astronomical Unit       [cm]
    pc  = 3.08572e18     # Parsec                  [cm]
    ms  = 1.98892e33     # Solar mass              [g]
    ts  = 5.78e3         # Solar temperature       [K]
    ls  = 3.8525e33      # Solar luminosity        [erg/s]
    rs  = 6.96e10        # Solar radius            [cm]
    mu = 2.3
    Rconst = 8.31446261815324e7 # in cgs
    G = 6.67259e-8       # in cgs
    
    
    
    nspecies = len(params["opacfile"])
    
    #-------------------------preparations--------------------------------------------------
    
    nr = params["nr"]; ntheta = params["ntheta"]; nphi = params["nphi"];
    mstar = params["mstar"]
    ri, thetai, rc, thetac, phic = get_ri_thetai_rc_thetac_phic(params)
    # surface density profile


    sin_new_thetac = np.cos(thetac)       # change 0 point to midplane
    sinthetac_diff = -np.diff(np.cos(thetai))

    dtheta = np.diff(thetai[:2])[0]
    blockarea = (ri[1:]**2 - ri[:-1]**2)/2. * dtheta
    blocklength = np.diff(ri)
    
    R, Theta = getR_Theta(params)
    X = R*np.cos(Theta); Y = R*np.sin(Theta)
    surface_density = np.zeros(nr, dtype=np.float64)
    
 
    for i in range(len(ri)-1):
        indices = np.where((ri[i] <= X) & (X < ri[i+1]))
        surface_density[i] = np.sum(rhod[indices]*blockarea[indices[0]])/blocklength[i]*au * 2.
    
    return surface_density


def averaged_temperature(params, rhod, temperature):
    #-----------------------------------------------------------------------------------
    au  = 1.49598e13     # Astronomical Unit       [cm]
    pc  = 3.08572e18     # Parsec                  [cm]
    ms  = 1.98892e33     # Solar mass              [g]
    ts  = 5.78e3         # Solar temperature       [K]
    ls  = 3.8525e33      # Solar luminosity        [erg/s]
    rs  = 6.96e10        # Solar radius            [cm]
    mu = 2.3
    Rconst = 8.31446261815324e7 # in cgs
    G = 6.67259e-8       # in cgs
    
    
    
    nspecies = len(params["opacfile"])
    
    #-------------------------preparations--------------------------------------------------
    
    nr = params["nr"]; ntheta = params["ntheta"]; nphi = params["nphi"];
    mstar = params["mstar"]
    ri, thetai, rc, thetac, phic = get_ri_thetai_rc_thetac_phic(params)
    # surface density profile


    sin_new_thetac = np.cos(thetac)       # change 0 point to midplane
    sinthetac_diff = -np.diff(np.cos(thetai))

    dtheta = np.diff(thetai[:2])[0]
    blockarea = (ri[1:]**2 - ri[:-1]**2)/2. * dtheta
    blocklength = np.diff(ri)
    
    R, Theta = getR_Theta(params)
    X = R*np.cos(Theta); Y = R*np.sin(Theta)
    surface_density = np.zeros(nr, dtype=np.float64)
    surface_density_temperature = np.zeros(nr, dtype=np.float64)
    
 
    for i in range(len(ri)-1):
        indices = np.where((ri[i] <= X) & (X < ri[i+1]))
        surface_density_temperature[i] = np.sum(rhod[indices]*temperature[indices]*blockarea[indices[0]])/blocklength[i]*au * 2.
        surface_density[i] = np.sum(rhod[indices]*blockarea[indices[0]])/blocklength[i]*au * 2.
    
    return surface_density_temperature/surface_density
