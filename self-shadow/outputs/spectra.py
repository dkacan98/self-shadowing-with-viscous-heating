from radmc3dPy import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits  
cube=fits.open('redden.fits')
cube.info()
data1=cube[0].data
#print(np.shape(data1))
header=cube[0].header
NAXIS1=header['NAXIS1']
NAXIS2=header['NAXIS2']
NAXIS3=header['NAXIS3']
#print(header,NAXIS1,NAXIS2,NAXIS3)

wavelength=np.linspace(0.1,1000,num=50)
print(wavelength)
flux=[]
for i in range(50):
   #print(data1[i,:,:])            #function that creates fits files made the shape of the data this way. number of arrays=number of wavelengths, number of rows for each array=number of nx pixel, number of columns for each array=number of ny pixel
   image=data1[i,:,:].ravel()
   #print(image)
   flux.append(sum(image))    #summing up Jansky/pixel flux values
def wavtofreq(x):
	return 1e4*natconst.cc/x
nu=list(map(wavtofreq,wavelength))
arr1=np.array(nu)
arr2=np.array(flux)
nuflux=np.multiply(arr1,arr2)
flux=[number*1000 for number in flux]

plt.plot(wavelength,flux,label='Total SED')
#plt.step(wavelength,flux,where='mid',label='Total SED')
plt.legend()
plt.xlabel('$\lambda\; [\mu \mathrm{m}$]')
plt.ylabel('mJansky $[\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}]$')
plt.xlim(0.1,1000)
#plt.ylim(0,7)
plt.xticks(np.arange(0,1000,50))
plt.show()
#plt.savefig('spectraneww.png')




