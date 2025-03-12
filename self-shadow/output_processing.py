###########################################imports############################
from tqdm import tqdm

from astropy.io import fits
from astropy import units as u
from astropy import constants as c

from utilities import *
###########################################imports############################

class radmcdata():
    def __init__(self, fname, dist = 185, hp = False,
         comp = 'total', hpind = 1, rewrite=False):
        """dist in pc"""
        
        self.fname = fname
        if (dist*u.dimensionless_unscaled).unit.is_equivalent(u.pc):
            dist = dist.to(u.pc).value
        self.dist = dist
        self.rewrite = rewrite
        
        if hp:
            self.comp = comp
            self.hpind = hpind
            self.hp_to_fits()
        else:
            self.radmcim_to_fits()     

       
#        self.getsed()

        #some booleans to remember whether the cube is already convolved and rebinned etc.
        self.reb_b      = False
        self.conv_b     = False
        self.redden2d_b = False
        self.redden_b   = False

#        self.conv()
#        self.nirspecrebin()
#        self.reddenim2d()
#        self.getsed()
#        self.reddensed()

    def hp_to_fits(self):
        mo = ModelOutput(os.path.join(self.fname,'hyperion_setup.rtout'))
        img = mo.get_image(distance=self.dist * pc, units='mJy', component = self.comp)
                         
        self.im2d = np.rollaxis(img.val[self.hpind],-1)
        self.imlam = img.wav
                         
        self.nl, self.ny, self.nx = self.im2d.shape
        self.xc = self.nx//2
        self.yc = self.ny//2
                                  
        self.spatres = (self.imlam*u.micron/(6.5*u.m)).to(" ").value*206265
       
        self.dx = ((img.x_max - img.x_min)/self.nx*u.cm).to(u.au).value
        self.dy = ((img.y_max - img.y_min)/self.ny*u.cm).to(u.au).value

        self.imfreq = (c.c/(self.imlam*u.micron)).to('Hz')        
        
    def radmcim_to_fits(self):
        fname = os.path.join(self.fname,'image.out')
        
        #if fits file already exists, use that, else make one
        if (not self.rewrite) * (os.path.exists(os.path.join(self.fname,"im2d_large.fits"))):
            im2d,header = fits.getdata(os.path.join(self.fname,"im2d_large.fits"),header=True)
            try:
                self.nl = header['nl']
            except:
                self.nl = header['nlam']
            self.nx = header['nx']
            self.ny = header['ny']
            self.dx = header['dx']
            self.dy = header['dy']
            self.imlam = np.loadtxt(fname, skiprows = 4, max_rows = self.nl)
            self.imfreq = (c.c/(self.imlam*u.micron)).to('Hz')
        else:
            #print("im2d not found. Are going to create one, but this might take a while")
            with open(fname, 'r') as fin:
                header = [line.strip('\n') for line in fin.readlines()[:4]]
                self.nl = int(header[2])
                self.nx,self.ny = [int(par) for par in header[1].split()]
                dx,dy = [float(par) for par in header[3].split()]
                self.dx = (dx*u.cm).to(u.au).value
                self.dy = (dy*u.cm).to(u.au).value
                
                self.imlam = np.loadtxt(fname, skiprows = 4, max_rows = self.nl)   #skip first 4 lines in the image.out file and take all lines until taking the next 180 (nl) line. So we only take wavelength values here assigning them to self.imlam.
                self.imfreq = (c.c/(self.imlam*u.micron)).to('Hz')

            header = {'nx':self.nx,'ny':self.ny,'dx':self.dx,'dy':self.dy,'nl':self.nl}
            im2d = (np.loadtxt(fname, skiprows = 4+self.nl+1).reshape((self.nl, self.nx, self.ny)))     #all flux values for 300x300 pixels and for each 180 wavelengths. Reshaping all the flux values in a shape that we have 180(nl) arrays each having nx rows and nx columns.
            normfac = (((self.dx/self.dist)**2 * u.arcsec**2).to(u.sr).value) * 1e23 * 1000 #to mJy
            im2d = normfac*im2d
            wfits(im2d, os.path.join(self.fname, "im2d_large.fits"), he=header)

        #normfac = (((self.dx/self.dist)**2 * u.arcsec**2).to(u.sr).value) * 1e23 * 1000 #to mJy     #converting erg s-1 cm-2 Hz-1 sr-1 to mJansky unit.
        #im2d *= normfac
        
        self.im2d = im2d
        self.xc = self.nx//2
        self.yc = self.ny//2
        telescope_size = 6.5*u.m # size of JWST in meters to get a handle on the resolution
        self.spatres = (self.imlam*u.micron/(telescope_size)).to(" ").value*206265
        return self
	
    def FFT(self,A):
        from numpy.fft import fft2,fftshift,ifft2,ifftshift
        fourierA = fftshift(fft2(ifftshift(A)))
        return fourierA

    def IFFT(self,fourierA):
        from numpy.fft import fft2,fftshift,ifft2,ifftshift
        A = fftshift(ifft2(ifftshift(fourierA)))
        return A

    def convolve(self,f,g):
        return self.IFFT(self.FFT(f)*self.FFT(g)[::-1])
    
    def conv(self):
        if self.conv_b:
            return self

        if (not self.rewrite) * (os.path.exists(os.path.join(self.fname,"im2d_conv_newinterp.fits"))):
            self.im2d = fits.getdata(os.path.join(self.fname,"im2d_conv_newinterp.fits"))
	    
        else:
            psf_wavel = fits.getdata("/home/dkacan/research/wavelengthcombined.fits")       #wavelengths that are used for psf
            psf_tot = fits.getdata("/home/dkacan/research/combined_file.fits")     #psfed values 

            convmask = (self.imlam >= psf_wavel[0]) * (self.imlam <= psf_wavel[-1])

            conv = self.im2d.copy()

            for i, wav in tqdm(enumerate(self.imlam)):
                if (wav >= psf_wavel[0]) * (wav <= psf_wavel[-1]): 
                    conv[i] = (self.convolve(self.im2d[i], psf_tot[np.argmin(abs(psf_wavel-wav))]))
                                
            self.im2d = conv[convmask]
            header = {'nx':self.nx,'ny':self.ny,'dx':self.dx,'dy':self.dy,'nl':self.nl}
            wfits(self.im2d, os.path.join(self.fname, "im2d_conv_newinterp.fits"), he=header)

        self.imlam = self.imlam[(self.imlam >=0.5917) * (self.imlam <= 28.5)]    # it was 28.7417 i changed it for my case
        self.conv_b = True
        return self
    
    def conv_arbitrarybeam(self,frame,dx,bmaj,dy = None, bmin = None,bpa = 0):
        from scipy.signal import convolve2d
        from astropy.convolution import Gaussian2DKernel

        """
        convolution with a gaussian to mimic ALMA observations
        dx and bmaj should be in same units
        """

        if bmin is None:
            bmin = bmaj
        if dy is None:
            dy = dx

        x_stddev = bmaj/dx/(2*np.sqrt(2*np.log(2)))
        y_stddev = bmin/dy/(2*np.sqrt(2*np.log(2)))
        theta = 90+bpa

        gaussian_2D_kernel = Gaussian2DKernel(x_stddev=x_stddev, y_stddev=y_stddev, theta = theta/180*np.pi)
        frame_conv = convolve2d(frame, gaussian_2D_kernel, mode='same')
        return frame_conv
        
    def nirspecrebin(self):
        """rebins the spectrum to pixel size of nirspec (0.1 x 0.1 arcsec) """

        #rebin only once, if asked multiple times
        if self.reb_b:
            return self
        
        nirspecres = .1
        shp = self.im2d.shape
        npix = int(nirspecres/self.dy)
        shave = int((self.ny * self.dy)%(nirspecres * self.dist)//2)
        ny = int(np.ceil(((shp[1] - 2 * shave) * self.dy)/(nirspecres * self.dist))) 
        nx = int(np.ceil(((shp[2] - 2 * shave) * self.dx)/(nirspecres * self.dist)))
        shpin = shp[0], ny, nx
        if shave > 0:
            self.im2d = rebin2d(self.im2d[:,shave:-shave,shave:-shave],shpin)
        else:
            self.im2d = rebin2d(self.im2d,shpin)
        self.im2d = self.im2d  
        self.dx = self.dx * (self.nx - 2 * shave) / nx
        self.dy = self.dy * (self.ny - 2 * shave) / ny
        self.nx = nx
        self.ny = ny
        self.xc = self.nx//2
        self.yc = self.ny//2
        header = {'nx':self.nx,'ny':self.ny,'dx':self.dx,'dy':self.dy,'nl':self.nl}
        wfits(self.im2d, os.path.join(self.fname, "im2d_reb.fits"), he=header)
        self.reb_b = True
        return self

    def reddensed(self):
        if self.redden_b:
            return self

        if "Av" not in self.__dict__:
            print("Av is missing in this model. Assuming you use an old model for HH48, with Av 5")
            self.Av = 0.5
        self.sed = deredden(self.imlam,self.sed,self.Av,redden=True)
        self.redden_b = True
        return self

    def reddenim2d(self):
        if self.redden2d_b:
            return self

        if "Av" not in self.__dict__:
            print("Av is missing in this model. Assuming you use an old model for HH48, with Av 5")
            self.Av = 5

        header = {'nx':self.nx,'ny':self.ny,'dx':self.dx,'dy':self.dy,'nl':self.nl}
        self.im2d = deredden(self.imlam,self.im2d,self.Av,redden=True)
        self.redden2d_b = True
        wfits(self.im2d, os.path.join(self.fname, "redden.fits"), he=header)
        return self

    def getsed(self):
        self.sed = np.sum(self.im2d,axis=(1,2))*u.mJy
        return self    #don't forget i changed this line for my purposes. it was return self

    def calc_peaktau(self,spec):
        
        self.conv().nirspecrebin().reddenim2d()
        shp = self.im2d.shape
        peaktau = np.zeros((shp[1],shp[2]))
        for i in range(shp[1]):
            for j in range(shp[2]):
                fl = ipm.loc[spec,'featurelim']
                tt1,cont1,data1,tau1 = continuumfitting(self.imlam,self.im2d[:,i,j],
                                                        xmin = fl[0], 
                                                        xmax = fl[1], 
                                                        range_limits = ipm.loc[spec,'fitlim'],
                                                        order=ipm.loc[spec,'fitorder'],
                                                        origwav = False)
                peaktau[i,j] = np.nanmax(tau1)
        return peaktau
