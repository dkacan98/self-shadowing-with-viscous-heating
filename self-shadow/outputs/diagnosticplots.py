from radmc3dPy import *
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from astropy.io import fits 
AU  =  1.4960e13 	# cm
AMU = 1.660538921e-24 	# atomic mass unit [g]


#reading the outputs of radmc3d run. 
data = analyze.readData(ddens=True)
def sph2cart(az, el, r):
    rsin_theta = np.outer(r,np.sin(el))  
    x = rsin_theta * np.cos(az)
    y = rsin_theta * np.sin(az)
    z = np.outer(r,np.cos(el))   
#(x[:,None]*y)
    return x, y, z
x,y,z=sph2cart(np.pi,data.grid.y, data.grid.x)
print(data.grid.y)

##plots the dust density with given colormap scales
c = plb.contourf(x/natconst.au,z/natconst.au,np.log10(data.rhodust[:,:,0,0]),cmap='BuPu',levels=np.linspace(-22,-14,100),extend='both')
#	E =[*range(-22,-14,1)]
#	cf = plt.contour(x/natconst.au,z/natconst.au,np.log10(data.rhodust[:,:,0,i]))
#	plt.clabel(cf,fontsize='smaller')
#	lines,labels = cf.legend_elements()
plb.xlabel('x [AU]')
plb.ylabel('z [AU]')
plb.xlim(-150,0)
plb.ylim(-150,150)
cb = plb.colorbar(c)
cb.set_ticks(np.arange(-23,-13))
cb.set_label(r'$\log_{10}{\rho}$', rotation=270.,verticalalignment='baseline')
plb.title('dust density ')
plb.savefig('dust density contour.png')
plb.clf()

##read opacity file and plot absorption opacity of grains with respect to wavelength
opac = analyze.readOpac(scatmat=False,idust=0)
plb.loglog(opac.wav[0], opac.kabs[0])
plb.xlabel(r'$\lambda$ [$\mu$m]')
plb.ylabel(r'$\kappa_{\rm abs}$ [cm$^2$/g]')
plb.legend("dust")
plb.title('Dust Opacities')
plb.savefig('dust opacity.png')
plb.clf()
data.getTau(wav=0.5)
taux=data.taux
print(np.shape(taux))
print(taux)
'''
#c = plb.contour(data.grid.x/natconst.au, np.pi/2.-data.grid.y, data.taux[:,:,0].T, [1.0],  colors='w', linestyles='solid')
c = plb.contour(x/natconst.au,z/natconst.au, data.taux[:,:,0], [1.0],  colors='w', linestyles='solid')
plb.clabel(c, inline=1, fontsize=10)
plb.savefig('optical depth.png')
'''

## Dust temperature color plot
plb.clf()
data2 = analyze.readData(dtemp=True)  #reading dust temperature (The readData() function is only an interface to the methods of the radmc3dData class.)
print(np.shape(data2.dusttemp))
c = plb.contourf(x/natconst.au,z/natconst.au,np.log10(data2.dusttemp[:,:,0,0]), 100)
plb.xlabel('x [AU]')
plb.ylabel('z [AU]')
cb = plb.colorbar(c)
cb.set_label(r'$\log_{10}T(K)$', rotation=270.,verticalalignment='baseline')
plb.title('dust temperature')
plb.savefig('dust temperature contour.png')
plb.clf()
	

###Combined figure of dust density + temperature distribution 
data2 = analyze.readData(dtemp=True)  #reading dust temperature (The readData() function is only an interface to the methods of the radmc3dData class.)

V = [20,47,80,96,128,155]      #for temperature contour lines
c = plb.contourf(x/natconst.au,z/natconst.au,np.log10(data.rhodust[:,:,0,0]),100,cmap='BuPu',levels=np.linspace(-22,-14,100),extend='both')
cb = plb.colorbar(c)
cb.set_ticks(np.arange(-23,-13))
cb.set_label(r'$\log_{10}{\rho}$', rotation=270.,verticalalignment='baseline')
plb.ylim(-150,150)
plb.xlim(-150,0)
plb.xticks(np.arange(-150,0,50))
plb.yticks(np.arange(-150,150,25))
con=plt.contour(x/natconst.au,z/natconst.au,data2.dusttemp[:,:,0,0],V)	
plt.clabel(con,fontsize='smaller',manual=True)
lines,labels = con.legend_elements()
plt.legend(lines,['CO','CH4','CO2','NH3','CH3OH','H2O'])
plt.xlabel('x [AU]')
plt.ylabel('z [AU]')
plt.title('disk dust density')
plt.show()
#plt.savefig('dust density with temperature %s.png'% t)
plt.clf()



