import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd

x=[20, 40, 50, 80, 100, 200, 400]

p4 = pd.read_csv('4.csv')
p8 = pd.read_csv('8.csv')

plt.errorbar(x, p4['Gauss'], yerr=p4['Error_Gauss'], fmt='.k');
plt.errorbar(x, p4['MC'], yerr=p4['Error_MC'], fmt='.r');
plt.errorbar(x, p4['FP'], yerr=p4['Error_FP'], fmt='.b');
plt.ylabel('Distance in state space between solution and truth')
plt.xlabel('Iterations')
plt.legend(['Gauss', 'Monte-Carlo', 'Forwards Proposal'])
plt.title('step size = 0.4')
plt.show()

plt.errorbar(x, p8['Gauss'], yerr=p8['Error_Gauss'], fmt='.k');
plt.errorbar(x, p8['MC'], yerr=p8['Error_MC'], fmt='.r');
plt.errorbar(x, p8['FP'], yerr=p8['Error_FP'], fmt='.b');
plt.ylabel('Distance in state space between solution and truth')
plt.xlabel('Iterations')
plt.legend(['Gauss', 'Monte-Carlo', 'Forwards Proposal'])
plt.title('step size = 0.8')
plt.show()

