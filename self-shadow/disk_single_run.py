#######################################
# imports
import pandas as pd
import glob
import os
import numpy as np
import logging
import sys
import time
import subprocess

from astropy import units as u
from astropy import constants as c
from scipy.interpolate import griddata
from scipy.optimize import brenth
from optool_functions import runoptool
from logformat import CustomFormatter
from radmc3dPy import *
#######################################
## main class ##
class radmc3d():
	"""
	
	inputs

	"""
	au  = 1.49598e13     # Astronomical Unit       [cm]
	pc  = 3.08572e18     # Parsec                  [cm]
	ms  = 1.98892e33     # Solar mass              [g]
	ts  = 5.78e3         # Solar temperature       [K]
	ls  = 3.8525e33      # Solar luminosity        [erg/s]
	rs  = 6.96e10        # Solar radius            [cm]
	mu = 2.3	     # Mean molec weight H2+He+Metals
	Rconst = 8.31446261815324e7 # in cgs
	G = 6.67259e-8       # in cgs
    	
	##instance inputs go here##
	# dirname_out:path to output directory
	def __init__(self, 
		dirname_out = None,
		verbose = True, scat = 0):    #scat 0 indicates isotropic scattering. if 1, anisotropic

		#load inits to self

		self.dirname_out = dirname_out 
		self.verbose = verbose
		self.scat = scat
		self.incl = 90  # Set inclination angle
		self.npix = 300  # Number of pixels in the output image (300x300)   #1 pixel ~ 1 AU   
		self.params =  {
	"nphot": 1e5,
	"nr": 200,
	"ntheta": 150,
	"nphi": 1,
	"rin": 1,
	"rout": 100.0,
	"thetaup": 0.6,
	"sigmag0": 100.0,
	"sigmad0": 10.0,
	"plsig": -1.0,
	"hr0": 0.0425,
	"plh": 2/7,
	"R_c": 50.0,
	"mstar": 1.0,
	"rstar": 2.0,
	"tstar": 4000.0,
	"opacfile": "dustkappa_dust.inp",
	"mdot": 1e-8,
	}
		self.paramsinit()
	def paramsinit(self):
		"""
		rin in au
		rout in au
		thetaup in radian

		sigmad0, sigma dust at R_c(au)

		hr0 , h/r at 1 au!!!

		"""

		nphot, nr, ntheta, nphi, rin, rout, thetaup, sigmag0, sigmad0, plsig, hr0, plh, R_c,\
		mstar, rstar, tstar, opacfile, mdot = self.params["nphot"], self.params["nr"], self.params["ntheta"],\
				self.params["nphi"], self.params["rin"], self.params["rout"], self.params["thetaup"],\
				self.params["sigmag0"], self.params["sigmad0"], self.params["plsig"], self.params["hr0"], self.params["plh"],\
				self.params["R_c"], self.params["mstar"], self.params["rstar"],\
				self.params["tstar"], self.params["opacfile"], self.params["mdot"]

		#rin = rin * u.au
		#rout = rout * u.au
		mstar = mstar * self.ms    #mstar is given as solar mass unit. now we convert it to grams.
		rstar = rstar * self.rs    #rstar is given as solar radius unit. now we convert it to cm.
		pstar = np.array([0.,0.,0.])

		self.nphot = nphot
		self.nr = nr
		self.ntheta = ntheta 
		self.nphi = nphi
		self.rin = rin
		self.rout = rout
		self.thetaup = thetaup
		self.sigmag0 = sigmag0
		self.sigmad0 = sigmad0
		self.plsig = plsig
		self.hr0 = hr0
		self.plh = plh 
		self.R_c = R_c
		self.mstar = mstar 
		self.rstar = rstar 
		self.tstar = tstar
		self.pstar = pstar		
		self.mdot = mdot
		self.opacfile = opacfile
		nspecies = len([self.opacfile])
		self.nspecies = nspecies if isinstance(self.opacfile, str) else len(self.opacfile)

		return self

	def make_wavelengths(self,n12=20,n23=100,n34=30,lam1=0.1e0,lam2=7.0e0,lam3=25.e0,lam4=1.0e4):

	#makes log spaced wavelengths, in three intervals

		lam12    = np.logspace(np.log10(lam1),np.log10(lam2),n12,endpoint=False)
		lam23    = np.logspace(np.log10(lam2),np.log10(lam3),n23,endpoint=False)
		lam34    = np.logspace(np.log10(lam3),np.log10(lam4),n34,endpoint=True)
		lam      = np.concatenate([lam12,lam23,lam34])
		nlam     = lam.size
		self.lam = lam
		self.nlam = nlam
	
		return self

	def grid_edge_center(self):

	#creates log spaced grid edges based on given inner and outer radius and number of grids. 

		ri       = np.logspace(np.log10(self.rin),np.log10(self.rout),self.nr+1)
		thetai   = np.linspace(self.thetaup,np.pi-self.thetaup,self.ntheta+1)
		thetai[0] = 0
		thetai[-1] = np.pi
		#Manually puts a grid edge on the midplane (pi/2) (#needed for midplane symmetry required by radmc3d)
		mid_idx = np.argmin(np.abs(thetai - np.pi/2))
		thetai[mid_idx] = np.pi / 2.0  # Force it to be exactly π/2
		phii     = np.linspace(0.e0,np.pi*2.e0,self.nphi+1)
		rc       = 0.5 * ( ri[:-1] + ri[1:] )
		thetac   = 0.5 * ( thetai[:-1] + thetai[1:] )
		phic     = 0.5 * ( phii[:-1] + phii[1:] )

		ri = ri * self.au    #rin given as au value. hence calculated ri is in au. but with this line now in the cm unit
		rc = rc * self.au
		self.ri = ri
		self.thetai = thetai
		self.phii = phii
		self.rc = rc
		self.thetac = thetac
		self.phic = phic

		print(self.thetai)
		# Check if pi/2 is exactly in the grid
		midplane_exists = np.any(np.isclose(self.thetai, np.pi/2, atol=1e-12))
		print(f"Midplane at π/2 exists in the grid: {midplane_exists}")

		return self


	def getR_Theta(self):
		"""
		return R and theta in meshgrid indexing='ij'. meshgrid form is needed to make r and t(zr) coordinate values to be compatible to each other for calculations.
		"""

		# Make the grid
		#
		qq       = np.meshgrid(self.rc,self.thetac,self.phic,indexing='ij')   #creates other dimensions for rc,thetac,phic to make them all (200,100,1) shaped meshgrids. Their values repeat on other dimensions to fill the shape.
		rr       = qq[0]  # meshgrid of (200,100,1) where a row gives r values in each column and all other rows repeat the same values.
		tt       = qq[1]  # meshgrid of (200,100,1) where a column gives theta values in each row and all other columns repeat the same values.
		zr       = np.pi/2.e0 - qq[1]  #theta changed to being measured from the mid-plane
		#
		self.rr = rr
		self.tt = tt
		self.zr = zr
		return self

#this creates [nr,ntheta,1] matrix for rr,tt,pp                        
							#rr=[1,2,3,4,5,....] 	             tt=[7,8,9,0,....]   pp=[pi] np=1
					                #rr=[1,1,1,1,1,....] (ntheta)	     tt=[7,8,9,0,....]    (ntheta) 
						        #    [2,2,2,2,2,....]		        [7,8,9,0,....]
					                #    [3,3,3,3,3,....] 	 	        [7,8,9,0,....]
			                                #     .				        [7,8,9,0,....]
 						        #     .				          .
						        #     . 				  .
							#    (nr)			         (nr)
#rr*tt operations are elementwise. So these rr tt matrix forms are good for that purpose.

	def density_calc(self):
        #calculating surface density (sigmad) and volume density (rhod)
		rhod = np.zeros((self.nspecies, self.nr, self.ntheta, self.nphi))
		sigmad   = self.sigmad0 * (self.rr/self.au)**self.plsig     #sigmad0 is sigma dust at R_c   plsig = powerlaw sigma 
		print(sigmad.shape)
		
		hhr = self.hr0 * (self.rr/self.au)**self.plh     #hr0  h/r at 1 AU     plh = powerlaw height
		hh = hhr * self.rr
		# self.rr in cm units. hh in cm units too.
		rhod[0]  = ( sigmad / (np.sqrt(2.e0*np.pi)*hh) ) * np.exp(-(self.zr**2/hhr**2)/2.e0)
		print(rhod.shape)
		self.sigmad = sigmad
		self.rhod = rhod

		r=self.rr[:,0,0]    #taking the relevant r and h values out of meshgrids of (200,100,1), to plot r vs sigma and r vs h
		surface_density=sigmad[:,0,0]
		self.surface_density=surface_density
		h=hh[:,0,0]
		r=r/self.au
		h=h/self.au
		
		p=pd.DataFrame(data={"col1":r,"col2":surface_density})
		p.to_csv("/scratch/bell/dkacan/surfacedensity.csv",sep=',',index=False)

		f=pd.DataFrame(data={"col1":r,"col2":h})
		f.to_csv("/scratch/bell/dkacan/rvsh.csv",sep=',',index=False)
		

	def heatrate_per_volume_calc(self):
        # calculating the heat rate per volume
		v_k = np.sqrt(self.G*(self.mstar) / (self.rr))     #all in cgs units
		heat_area = 3/(4*np.pi) * (v_k)**2 * self.mdot      #give mdot as g/s 
		heat_volume = heat_area / self.sigmad * self.rhod  
		print(heat_volume.shape)
		self.heat_volume = heat_volume

	def write_heatrate(self):
        #writes the heat rate per volume in each grid to the radmc input file.
		with open(os.path.join(self.dirname_out,'heatsource.inp'),'w+') as f:
			f.write('1\n')
			f.write('%d\n'%(self.nr*self.ntheta*self.nphi))
			for i in range(self.nspecies):
				data = self.heat_volume[i].ravel(order='F')
				data.tofile(f, sep='\n', format="%13.6e")
				f.write('\n')
			
	def write_wavelengths(self):
	#writes the wavelength file used to do the thermal mcmc. 
		with open(os.path.join(self.dirname_out,'wavelength_micron.inp'),'w+') as f:
			f.write('%d\n'%(self.nlam))
			for value in self.lam:
		    		f.write('%13.6e\n'%(value))

	def write_starsinp(self):
	#writes stellar radiation field to the radmc input file
		with open(os.path.join(self.dirname_out,'stars.inp'),'w+') as f:
			f.write('2\n')
			f.write('1 %d\n\n'%(self.nlam))
			f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n'%(self.rstar,self.mstar,self.pstar[0],self.pstar[1],self.pstar[2]))
			for value in self.lam:
				f.write('%13.6e\n'%(value))
			f.write('\n%13.6e\n'%(-self.tstar))     #blackbody spectrum with a temperature equal to the effective temperature of the star

	def write_grid(self):
	#writes grid to the radmc input file
		with open(os.path.join(self.dirname_out,'amr_grid.inp'),'w+') as f: 
			f.write('1\n')                       # iformat
			f.write('0\n')                       # AMR grid style  (0=regular grid, no AMR)
			f.write('100\n')                     # Coordinate system: spherical
			f.write('0\n')                       # gridinfo
			f.write('1 1 0\n')                   # Include r,theta coordinates
			f.write('%d %d %d\n'%(self.nr,self.ntheta,self.nphi))  # Size of grid
			for value in self.ri:
			    f.write('%13.12e\n'%(value))      # X coordinates (cell walls)
			for value in self.thetai:
			    f.write('%13.12e\n'%(value))      # Y coordinates (cell walls)
			for value in self.phii:
			    f.write('%13.12e\n'%(value))      # Z coordinates (cell walls)

	def write_density(self):
	#write dust density to the radmc input file
		with open(os.path.join(self.dirname_out,'dust_density.inp'),'w+') as f:
			f.write('1\n')                       # Format number
			f.write('%d\n'%(self.nr*self.ntheta*self.nphi))     # Nr of cells
			f.write('{}\n'.format(self.nspecies))    # Nr of dust species. {} is the placeholder. nspecies value will be put there.
			for i in range(self.nspecies):
			    data = self.rhod[i].ravel(order='F')    # Create a 1-D view, fortran-style indexing
			    data.tofile(f, sep='\n', format="%13.6e") 
			    f.write('\n') 

	def run_optool(self):
        #runs optool (from the script of optool_functions.py) to calculate opacity parameters based on given grain size and composition and outputs dustkappa_dust.inp file.
		scat = '-scat' if self.scat else ''     #if True (scat=1(anisotropic scattering))
		runoptool(f'optool -c pyr-mg80 0.85\
                           -c c 0.15\
                           -p 0.25 -dhs -radmc dust\
                           -a 0.01 100\
                           {scat} -lmin 1e-2 -chop 3 -nlam 1000 -na 30 -o {self.dirname_out}',verbose = self.verbose)
		return self

	def write_dustopac(self):
        #puts the output of optool into a format that radmc wants.
		with open(os.path.join(self.dirname_out,'dustopac.inp'),'w+') as f:
			f.write('2               Format number of this file\n')
			f.write('{}               Nr of dust species\n'.format(self.nspecies))
			f.write('============================================================================\n')
			for i in range(self.nspecies):
				f.write('1               Way in which this dust species is read\n')
				f.write('0               0=Thermal grain\n')
				f.write('dust     Extension of name of dustkappa_***.inp file\n')
				f.write('----------------------------------------------------------------------------\n')

	def write_radmc3dinp(self):
        #a reqired input file for radmc runs that gives a couple of settings. more can be added. 
		with open(os.path.join(self.dirname_out,'radmc3d.inp'),'w+') as f:
			f.write('nphot = %d\n'%(self.nphot))
			f.write('scattering_mode_max = 1\n')
			f.write('iranfreqmode = 1\n')
			f.write('itempdecoup = 0\n')     #each dust species are forced to have the same temperature.
			f.write('setthreads = 32\n')
			f.write('istar_sphere = 1\n')

	def read_density(self, filename = "dust_density.inp"):
	#not using this yet.
	    ntotal = self.nr*self.ntheta*self.nphi
	    rhod = np.fromfile(filename, sep='\n', dtype=np.float64)[3:].\
	    reshape((self.nr,self.ntheta,self.nphi,self.nspecies),order='F')[:,:,0,:]
	    rhod = np.moveaxis(self.rhod, (0, 1, 2),  (1, 2, 0))
	    return rhod

	def read_temperature(self, filename = "dust_temperature.dat"):
        #not using this yet.
	    ntotal = self.nr*self.ntheta*self.nphi
	    temperature =  np.fromfile(filename, sep='\n',dtype=np.float64)[3:3+ntotal].\
	    reshape((self.nr,self.ntheta,self.nphi),order='F')[:,:,0]
	    return temperature

'''
def run_simulation():
# run function to run all needed functions in order. 
	sim = radmc3d(dirname_out="/home/dkacan/research/radmc3d-2.0-master/self-shadow/Chaing_Goldreich_scale_height/")
	sim.paramsinit()
	sim.make_wavelengths()
	sim.grid_edge_center()
	sim.getR_Theta()
	sim.density_calc()
	sim.heatrate_per_volume_calc()
	sim.write_grid()
	sim.write_heatrate()
	sim.write_density()
	sim.write_wavelengths()
	sim.write_starsinp()
	sim.run_optool()
	sim.write_dustopac()
	sim.write_radmc3dinp()

	#run radmc thermal to get temperature structure
	run_statement = f'cd {sim.dirname_out} && radmc3d mctherm'      #uses wavelength_micron.inp
	shut_up = subprocess.DEVNULL if not sim.verbose else None
	proces = subprocess.run([run_statement],shell=True, stderr=shut_up, stdout=shut_up)
	#run radmc to get image cube
	run_statement = f'cd {sim.dirname_out} && radmc3d image incl {sim.incl} lambdarange 0.1 1000. nlam 50 npix {sim.npix} sloppy'
	shut_up = subprocess.DEVNULL if not sim.verbose else None
	proces = subprocess.run([run_statement],shell=True, stderr=shut_up, stdout=shut_up)
#execution of the running function
run_simulation()
'''
