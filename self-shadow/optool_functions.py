from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

def readkappa(fname,skiprows=2):
    lines = "".join([line for line in open(os.path.join(outputpath,fname))\
             if not line.startswith('#')])
    return pd.read_csv(StringIO(lines),delim_whitespace=True,skiprows=skiprows,
            names = ['wav','kabs','ksca','gg'])

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
        
def runoptool(osline,verbose = True):
    shut_up = subprocess.DEVNULL if not verbose else None
    proces = subprocess.run([f'{osline}'],shell=True, stderr=shut_up, stdout=shut_up)

    
def make_kgfig(fnames,labels = [],xlim=(1,100),ylim=(10,1e3),log=True):
    colors = ['black','red','darkblue','darkorange','green','purple','pink']
    if len(fnames)>len(colors):
        colors = plt.cm.jet(np.linspace(0,1,len(daliname)))
    
    for i in range(len(fnames)-len(labels)): labels.append(None)
    
    ######## Figure
    fig,axes = plt.subplots(1,3,figsize=(15,4))

    ######## k abs
    ax = axes[0]
    for i,fname in enumerate(fnames):
        fl = readkappa(fname)
        ax.plot(fl.wav,fl.kabs,label=labels[i],color=colors[i])

    if log:
        ax.loglog()
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_ylabel('k$_{\\rm abs}$')
    ax.set_xlabel('$\lambda$ ($\mu$m)')

    ######## k scat
    ax = axes[1]
    for i,fname in enumerate(fnames):
        fl = readkappa(fname)
        ax.plot(fl.wav,fl.ksca,label=labels[i],color=colors[i])
    if log:
        ax.loglog()
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_ylabel('k$_{\\rm scat}$')
    ax.set_xlabel('$\lambda$ ($\mu$m)')

    ######## gg
    ax = axes[2]
    for i,fname in enumerate(fnames):
        fl = readkappa(fname)
        ax.plot(fl.wav,fl.gg,label=labels[i],color=colors[i])
    if log:
        ax.semilogx()
        
    ax.set_ylim(0,1)
    ax.set_xlim(xlim)
    ax.set_ylabel('asymmetry g')
    ax.set_xlabel('$\lambda$ ($\mu$m)')

    if len([i for i in labels if i]):
        ax.legend()
    return fig,axes
