import numpy as np

a =np.array( [[1, 2, 0, 4, 5], [2, 3, 5, 3, 4]])
b=a[0,:].copy()
b[b!=0]=1
count=sum(b)
print(count)
print(a[0,:].reshape(1,5))