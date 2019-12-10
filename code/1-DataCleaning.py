import pandas as pd
import numpy as np
import sqlite3 as sql

def extract_raw():
	"""
	extract all data from 2014 to 2016 and write to new db
	"""
	conn = sql.connect(r"sqf.db")
	c = conn.cursor()

	query = "select * from SQFdata where year >= 2014 and year <= 2016"

	data = pd.read_sql(query, conn)
	print('load data')
	data['id'] = range(len(data)) # add id for each oberservation

	conn.commit()
	conn.close()
	conn = sql.connect(r"1125-sqf-cleaned.db")
	data.to_sql('rawsqf_1416', conn, if_exists='replace', index=False)

# extract_raw()

def all():
	"""
	Created on Tue Oct 22 00:23:21 2019

	@author: liu
	"""
	#%%read data via sqlite3
	conn = sql.connect(r"1125-sqf-cleaned.db")
	c = conn.cursor()

	query = "select * from rawsqf_1416 "

	data = pd.read_sql(query, conn)
	print('load data')

	conn.commit()
	# conn.close()

	#%% year, month of occurence
	yrmn = data.iloc[:, [0, 3, -1]]
	yrmn.loc[:, 'datestop'] = (yrmn.loc[:, 'datestop'] / 10e5).astype('int')
	#%%suspect personal feature
	features = pd.read_excel(r".\SQF-File-Documentation\2016 SQF File Spec.xlsx", header=3)
	suspFeat = list(features.loc[80:89, "Variable"])

	dataSuspFeat = data.loc[:, suspFeat]
	dataSuspFeat = dataSuspFeat.replace(r'^\s*$', np.nan, regex=True)
	#dataSuspFeat = dataSuspFeat[dataSuspFeat['age'] != '**']
	dataSuspFeat = dataSuspFeat.replace(r'**', np.nan)
	dataSuspFeat.age = dataSuspFeat.age.astype('float64')
	dataSuspFeat.ht_feet = dataSuspFeat.ht_feet.astype('float64')
	dataSuspFeat.ht_inch = dataSuspFeat.ht_inch.astype('float64')
	dataSuspFeat['ht'] = dataSuspFeat['ht_feet']*12 + dataSuspFeat['ht_inch']
	dataSuspFeat = dataSuspFeat.drop(['ht_feet', 'ht_inch', 'dob'], 1)
	dataSuspFeat.weight = dataSuspFeat.weight.astype('float64')
	#%% location info
	location = list(features.loc[100:106, "Variable"])
	dataLoc = data.loc[:, location]
	dataLoc = dataLoc.replace(r'^\s*$', np.nan, regex=True)
	dataLoc = dataLoc.drop(['city', 'sector', 'state', 'zip', 'post', 'beat'], 1)

	#%% other info
	other = list(features.loc[41:72, "Variable"])
	other += (["explnstp", "othpers", "sumissue"])
	dataOther = data.loc[:, other]
	dataOther = dataOther.replace('Y', 1)
	dataOther = dataOther.replace('N', 0)

	dataOther1 = data.loc[:, ["perobs", "perstop", "typeofid"]]
	dataOther = dataOther.join(dataOther1, how='left')
	#%% merge all features
	dataJoin = dataLoc.join(dataSuspFeat, how='left')
	dataJoin = dataJoin.join(yrmn, how='left')
	dataJoin = dataJoin.join(dataOther, how='left')

	"""
	@author: xia
	"""
	def times_trans(x):
		time = str(x)
		if ':'in time:
			time = time.split(':')
			try:
				minute = int(time[1])
			except:
				minute = 0
			try:
				hour = int(time[0])
			except:
				hour = 0
			return hour*60+minute
		else:
			if len(time) <= 2:
				return int(x)
			else:
				minute = int(time[-2:])
				hour = int(time[:-2])
				return hour*60+minute

	def clean(data):
		data['inout'] = data['inout'].apply(lambda x: 1 if x=='I' else 0) # in :1 ; out: 0
		print('finish inout')

		data['timestop'] = data['timestop'].apply(times_trans)
		print('finish timestop')

		df = pd.read_csv('crime_list.csv')
		crime_dict = df.set_index('crime').to_dict()['code']
		crime_dict['490.10'] = 30
		crime_dict['140.10'] = 50

		data.crimsusp = data.crimsusp.apply(lambda x:crime_dict[x] if x in crime_dict else 34)
		print('finish crimsusp')

		for i in nominal:
			data[i] = data[i].apply(lambda x:1 if x=='Y' else 0)
		print('finish norminal')

		data.arstmade = data.arstmade.apply(lambda x:1 if x=='Y' else 0)
		print('finish arstmade')

		# data = data.apply(cood_a_to_b,args=("epsg:2263","epsg:4326",),axis=1)
		# del data['xcoord']
		# del data['ycoord']
		# print('finish coord')

		# print(data.head())

		# data.to_csv(output,index=False)
		return data

	nominal = ['offunif','frisked','searched','contrabn','pistol','riflshot','asltweap','knifcuti','machgun','othrweap']
	use = ['arstmade','timestop','inout','crimsusp'] #,'xcoord','ycoord'
	data = clean(data)
	data = data[['id']+use+nominal]

	# merge data from two parts, write to new db
	del dataJoin['id']
	data = data.join(dataJoin, how='left')
	data.to_sql('cleanedsqf_1416', conn, if_exists='replace', index=False)

all()

###### The following functions are used to change the coordination system
def demo_trans():
	lon, lat = -73.869426, 40.863772
	p1 = pyproj.Proj(init="epsg:4326")  

	p2 = pyproj.Proj(init="epsg:2263")  
	x1, y1 = p1(lon, lat)
	print(lon, lat)
	print(x1, y1)

	x2, y2 = pyproj.transform(p1, p2, x1, y1, radians=True)
	print(x2, y2) # 1020366.7116745672 253999.97497915442

	x1, y1 = pyproj.transform(p2, p1, x2, y2, radians=True)
	print(x1, y1)

	lon, lat = p1(x1, y1,inverse=True)
	print(lon, lat)

def cood_a_to_b(df,a,b):
	x1 = df.xcoord
	y1 = df.ycoord

	p1 = pyproj.Proj(init=a)
	p2 = pyproj.Proj(init=b)

	if x1 != ' ':
		x2, y2 = pyproj.transform(p1, p2, x1, y1, radians=True)
		lon, lat = p1(x1, y1,inverse=True)
		# print(lon, lat)
		df['lon'] = lon
		df['lat'] = lat
	else:
		df['lon'] = np.nan
		df['lat'] = np.nan	
	return df
