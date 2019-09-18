import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.read_csv("C:\\Users\\xinran\\Desktop\\boxplot.csv")
#plt.rc('font',family='SimHei',size=13)

sns.set(style="ticks")
ax=sns.boxplot(x="group",y="value",data = df,whis=1,saturation=1,palette=sns.color_palette("husl",n_colors=3), width=0.75, fliersize=2, linewidth=None,hue="class")

ax.set(ylim=(2,3.5))

#ax.autoscale(enable=True, axis='both',tight=True)
#plt.setp(ax.get_xticklabels(), visible=False)
#ax.tick_params(axis=u'both', which=u'both',length=0)
plt.legend(loc='upper right')
plt.xlabel('value', fontsize=16)
#设置坐标轴范围
plt.show()