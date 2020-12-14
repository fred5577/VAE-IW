import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotinpy as pnp
from matplotlib.patches import Patch
import numpy as np
import numpy.ma as ma

sns.set()
sns.set_style("ticks")
sns.palplot(sns.color_palette("Paired"))
data = pd.read_csv("barplotstuff.csv") 
data = data.sort_values(by=["Human"]).set_index("Domain")
#fig = plt.figure(figsize=(10,15))


f,(ax) = plt.subplots(1,1,sharey=True,figsize=(3,18))
# f.subplots_adjust(left=0.2)
#ax = fig.add_subplot(1,1,1)
ax.set_xlim(0,1100)
#ax2.set_xlim(2000,13000)


#ax.axvline(100,c='black')
ax.axhline(11.5,c='black')
# ax2.axhline(11.5,c='black')


ax.set_facecolor('white')
#ax2.set_facecolor('white')

indices = np.arange(len(data))

mask1 = ma.where(data.Human >= data.BPROST)
mask2 = ma.where(data.Human <= data.BPROST)
maskNames1 = data.index[mask1].tolist()
maskNames2 = data.index[mask2].tolist()
#print(data.index[mask1].tolist())
#print(data.loc[maskNames,"Human"].tolist())
ax.barh(data.index,data.col, height=1, color='turquoise', alpha=1)
ax.barh(maskNames1,data.loc[maskNames1,"Human"].tolist(), height=0.90, color='turquoise', alpha=1, linewidth=0)
ax.barh(data.index,data.BPROST, height=0.90, color='gray', alpha=1, linewidth=0)
ax.barh(maskNames2,data.loc[maskNames2,"Human"].tolist(), height=0.90, color='turquoise', alpha=1, linewidth=0)
# ax.legend(
#     [
#         Patch(facecolor="turquoise"),
#         Patch(facecolor="gray")
#     ], ["DQN", "B-PROST"],loc=4
# )
#p3 = plt.bar(data.BPROST[mask2], color='r', alpha=1, edgecolor='none',linewidth=0,width=0.5, log=False)

#data.col.plot(kind='barh',width=1, ax=ax,alpha=0.4)
#data.query("Human < BPROST").Human.plot(kind='barh',width=1, color="turquoise", ax=ax,alpha=0.4)
#data.query("Human < BPROST").BPROST.plot(kind='barh',width=1, color="gray", ax=ax,alpha=0.6)
#data.query("Human < BPROST").BPROST.plot(kind='barh',width=1, x=["B-PROST"], color="gray", ax=ax,alpha=0.6)
#data.query("Human > BPROST").Human.plot(kind='barh', x=["VAE-IW"],width=1, color="turquoise", ax=ax,alpha=0.4)
#data.col.plot(kind='barh',width=1, ax=ax,alpha=0.4)

#plt.yticks(rotation=30)
plt.xticks([0,100,200,400,600,800,1000])
# plt.xticks([2500,7500,12500])

#kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
#data.Human.plot(kind='barh',width=1, ax=ax2,alpha=0.5)
#data.BPROST.plot(kind='barh',width=1, ax=ax2,alpha=0.5)

#ax.set_xscale("log")
#ax2.set_xscale("log")

# ax.spines['bottom'].set_visible(True)
# ax2.spines['left'].set_visible(False)
# ax.yaxis.tick_left()
# ax.tick_params(bottom=True) # don't put tick labels at the top
# ax2.yaxis.tick_bottom()

# Make the spacing between the two axes a bit smaller
#plt.subplots_adjust(wspace=0.15)
sns.despine()

ax.text(2500, 12-0.2, "Above human level", color='black')
ax.text(2500, 11-0.2, "Below human level", color='black')

for i, v in enumerate(data.Human):
   ax.text(10, i-0.25, str(round(v)) + "%", color='black')

# d = .015 # how big to make the diagonal lines in axes coordinates
# # arguments to pass plot, just so we don't keep repeating them
# kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((1-d,1+d),(-d,+d), **kwargs) # top-left diagonal
# ax.plot((1-d,1+d),(1-d,1+d), **kwargs) # bottom-left diagonal

# kwargs.update(transform=ax2.transAxes) # switch to the bottom axes
# ax2.plot((-d,d),(-d,+d), **kwargs) # top-right diagonal
# ax2.plot((-d,d),(1-d,1+d), **kwargs) # bottom-right diagonal


f.savefig("first_new_1.png")