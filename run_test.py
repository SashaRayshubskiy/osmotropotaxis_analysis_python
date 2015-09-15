__author__ = 'sasha'

# Load the data into the environment
import osmotropotaxis_main
d = osmotropotaxis_main.osmotropotaxis_main()
d.run()

# run test case 1
import osmotropotaxis_test
dd = osmotropotaxis_test.osmotropotaxis_test()
dd.run(d.bdata, d.exp_mdata)


import matplotlib.pylab as plt
import numpy as np

xx = np.random.rand(12,)
yy = np.random.rand(12,)
plt.figure()
plt.plot(xx,yy)
plt.show()
