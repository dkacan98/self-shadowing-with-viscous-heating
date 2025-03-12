import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('rvsh.csv',sep=',',header=None,skiprows=[0],dtype=np.float64)
df.dropna(inplace=True)
df.rename(columns={0:'r',1:'h'},inplace=True)
r=df['r'].tolist()
h=df['h'].tolist()
plt.plot(r,h)
plt.xlabel('r [AU]')
plt.ylabel('h [AU]')
plt.show()
