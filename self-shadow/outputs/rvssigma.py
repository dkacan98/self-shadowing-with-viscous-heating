import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('surfacedensity.csv',sep=',',header=None,skiprows=[0],dtype=np.float64)
df.dropna(inplace=True)
df.rename(columns={0:'r',1:'sigma'},inplace=True)
r=df['r'].tolist()
sigma=df['sigma'].tolist()
plt.plot(r,sigma)
plt.xlabel('r [AU]')
plt.ylabel('\u2211')
plt.show()
