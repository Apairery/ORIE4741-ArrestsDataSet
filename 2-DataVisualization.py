import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import random
import sqlite3
import seaborn as sns
# sns.set_style("darkgrid")
# sns.set(style="darkgrid")
pd.set_option('display.max_columns', None)
matplotlib.use('TkAgg')
matplotlib.rcdefaults()

########################
# Default Setting
########################
p = matplotlib.rcParams
p["font.family"] = "Times New Roman" #"sans-serif"
# p["font.sans-serif"] = ["SimHei", "Tahoma"]
p["font.size"] = 36
p["axes.unicode_minus"] = False
p['lines.linewidth'] = 4
p['pdf.fonttype'] = 42
p['ps.fonttype'] = 42
colors=['#FF595E','#FFCA3A','#00AFB5','#8AC926']
# conn = sql.connect(r"1125-sqf-cleaned.db")

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# ​
#
#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus']=False
def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        #随便设计的y函数
        plt.plot(x, np.sin(x+i*0.5)*x)
    plt.show()
# sns.set_style(style={"font.sans-serif":['Microsoft YaHei', 'SimHei']})
# sns.set_style('darkgrid')        
# sinplot()

def draw_all_stops():
	conn = sqlite3.connect(r"sqf.db")
	df = pd.read_sql_query("select * from SQFdata_byyear ;", conn)
	df.columns = ['year','stops']

	pdf = PdfPages('final/Figure1_allstops.pdf')
	figure = plt.figure(figsize=(12,12*0.5))
	figure.tight_layout()
	plt.plot(df.year, df.stops)
	plt.annotate('685,724 stops',xy=(2011,685724),fontsize=22,xytext=(2011,685724),arrowprops=dict(headwidth=1,headlength=0.4,width=1))
	plt.annotate('45,787 stops',xy=(2014,45787),fontsize=22,xytext=(2014,45787),arrowprops=dict(headwidth=1,headlength=0.4,width=1))
	plt.xlabel('year')
	plt.ylabel('# of stops')
	plt.show()
	pdf.savefig(figure,bbox_inches='tight',dpi=figure.dpi,pad_inches=0.0)
	plt.close()
	pdf.close()
sns.set_style(style={"font.sans-serif":["Times New Roman"]})
sns.set_style('darkgrid')  
draw_all_stops()


def draw_all_stops_sns():
	conn = sqlite3.connect(r"sqf.db")
	df = pd.read_sql_query("select * from SQFdata_byyear ;", conn)
	df.columns = ['year','stops']

	pdf = PdfPages('final/Figure1_allstops.pdf')
	figure = plt.figure(figsize=(12,12*0.5))
	figure.tight_layout()
	# sns.lineplot(df.year, df.stops)
	plt.plot(df.year, df.stops)
	plt.annotate('685,724 stops',xy=(2011,685724),fontsize=22,xytext=(2011,685724),arrowprops=dict(headwidth=1,headlength=0.4,width=1))
	plt.annotate('45,787 stops',xy=(2014,45787),fontsize=22,xytext=(2014,45787),arrowprops=dict(headwidth=1,headlength=0.4,width=1))
	plt.xlabel('year')
	plt.ylabel('# of stops')
	plt.show()
	pdf.savefig(figure,bbox_inches='tight',dpi=figure.dpi,pad_inches=0.0)
	plt.close()
	pdf.close()
# draw_all_stops_sns()
def draw(df):
	pdf = PdfPages('Figure1_1.pdf')
	figure = plt.figure(figsize=(12,12*0.8))
	figure.tight_layout()
	plt.hist(df.timestop,bins=24)
	plt.xticks([i*60 for i in range(25)],rotation=90)
	plt.xlabel('time of a day (in minute)')
	plt.ylabel('number of stops')
	plt.show()
	pdf.savefig(figure,bbox_inches='tight',dpi=figure.dpi,pad_inches=0.0)
	plt.close()
	pdf.close()


############## across time
def GroupByTime(timetype): # timetype = ['year','datestop','timestop']
	pdf = PdfPages('Figure_time_%s.pdf'%timetype)
	figure = plt.figure(figsize=(12,12*0.8))
	figure.tight_layout()

	df = pd.read_csv('2014-2016-cleaned.csv')
	df.timestop = df.timestop.apply(lambda x:x//60) # minute to hour
	# print(df.head())
	# print(df.datestop.describe())
	if timetype == 'timestop':
		title = 'hour'
	elif timetype == 'datestop':
		title = 'month'
	else:
		title = timetype

	new = df.groupby(by=timetype)['arstmade'].agg(['mean','count','sum']).reset_index()

	new.columns=[title,'arrest_rate','stop','arrest']

	new['non-arrest'] = new.apply(lambda x: x.stop - x.arrest, axis=1)

	figure, ax1= plt.subplots(figsize=(12,6))
	ax2 =ax1.twinx()
	# plt.
	p1 = new.loc[:,['non-arrest','arrest']].plot.bar(stacked=True,ax=ax1, color=colors[:2])
	(left,right) = plt.xlim()

	p2 = new.loc[:,['arrest_rate']].plot(ax=ax2,secondary_y=True,xlim = (left,right),color=colors[2])
	plt.ylabel('arrest rate')
	ax1.legend(loc='lower right')
	# ax2.legend(loc='upper left')
	ax1.set_xlabel(title)
	ax1.set_ylabel('# of arrest/non-arrest')
	ax1.set_xticklabels(new[title],rotation=0)
	# plt.show()
	pdf.savefig(figure,bbox_inches='tight',dpi=figure.dpi,pad_inches=0.0)
	plt.close()
	pdf.close()

# GroupByTime('timestop')
# GroupByTime('datestop')
# GroupByTime('year')

# draw(df)
# df = pd.read_csv('sqf-2016-draw.csv')

# df = pd.read_csv('2016CompleteClean.csv')
# 
# # 
# print(df.head())

# k = df.groupby('crimsusp')['crimsusp'].agg({'num':len}).reset_index()
# print(k.columns)
# plt.scatter(k.crimsusp,k.num)
# plt.xlabel('crime type')
# plt.ylabel('number')
# print(k)

# df.boxplot('perobs','ac_incid')
# df.boxplot('perobs','ac_time')

# df2 = df.groupby(['arstmade','city']).apply(len).reset_index()
# df2.columns = ['arstmade', 'city', 'count']
# df2.plot.bar(x='city',y='count',by='arstmade')
# plt.xlabel('right:1')
# print(df2)
# plt.show()