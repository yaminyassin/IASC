import numpy as np 


array = np.array([ [1,1], [1,1], [1,1]])


media = np.insert(array, 0, range(1, len(array)+1), axis=1)

print(media)