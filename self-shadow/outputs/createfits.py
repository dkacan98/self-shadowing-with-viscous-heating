## this file creates fits file from image.out output of radmc3d, using methods in output_processing script. They also consider distance to the source (currently taken as 185 parsec) and measure the calibrated flux based on that.
#reddening of the light due to extinction is applied with reddenim2d. it's output fits file is named redden.fits.

import sys
sys.path.append('/home/dkacan/research/radmc3d-2.0-master/RIPPED')
from tqdm import tqdm
from astropy.io import fits
from astropy import units as u
from astropy import constants as c
from utilities import *
import output_processing as op

radmcdata1=op.radmcdata(fname = '/home/dkacan/research/radmc3d-2.0-master/self-shadow/outputs/')
radmcdata1.radmcim_to_fits()
radmcdata1.reddenim2d()
