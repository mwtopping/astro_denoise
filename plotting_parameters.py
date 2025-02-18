import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (7,6)
myfontsize=14
plt.rcParams['xtick.labelsize'] = myfontsize
plt.rcParams['ytick.labelsize'] = myfontsize
plt.rcParams['axes.labelpad']=5

plt.rcParams['xtick.major.size'] = 7
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.major.pad'] = 7

plt.rcParams['ytick.major.size'] = 7
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['ytick.minor.width'] = 2
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.pad'] = 4

plt.rcParams['legend.frameon']=True
plt.rcParams['legend.fontsize']=myfontsize
plt.rcParams['legend.loc']='upper right'
plt.rcParams['legend.numpoints']=1

plt.rcParams['axes.titlesize']=myfontsize+4
plt.rcParams['axes.labelsize']=myfontsize+2
plt.rcParams['axes.linewidth']=2
plt.rcParams['figure.facecolor']='white'

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = ['dejavu serif']
plt.rcParams['font.serif'] = ['Times']
