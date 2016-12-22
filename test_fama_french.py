import os
import pandas as pd
import numpy as np
from sklearn import linear_model


current_path = os.getcwd()
weights_path = current_path + '\\weights'
fees_path = current_path + '\\fees'

files = os.listdir(fees_path)
fund_names = list(map(lambda x: x[:-5].replace('_','&').encode('utf-8'), 
	files))

# fund of funds
funds = pd.read_excel('Fund of Funds-US Equity.xlsx', encoding='utf-8')
funds = funds[funds.columns[[0] + list(range(35,341))]]
funds['Name'] = list(map(lambda x: x.encode('utf-8'), funds['Name']))


# upper level fees
upper_fees = pd.read_excel('upper_fees.xlsx')
upper_fees['Name'] = list(map(lambda x: x.encode('utf-8'), 
	upper_fees['Name']))
upper_fees.columns.values[1:] = list(map(lambda x:x[-4:], 
	upper_fees.columns.values[1:]))
upper_fees = upper_fees.fillna(0)


def createFamaFrenchFactor():

	old_factor = pd.read_excel('FactorReturn.xlsx')
	old_factor = old_factor[old_factor.columns[:-1]]

	new_factor= pd.read_excel('Russell Factor Returns 84 to 16.xlsx')
	new_factor[new_factor.columns[0]] = new_factor[new_factor.columns[0]] // 100
	new_factor[new_factor.columns[1:3]] = new_factor[new_factor.columns[1:3]] * 100

	factor = pd.merge(old_factor, new_factor)

	return factor


def make_yymm(start_year = 1990,start_month = 7, end_year = 2015, 
	end_month = 2015):
	yymm = []
	year = start_year
	month = start_month
	while (year < end_year) or (year == end_year and month <= end_month):
		yymm.append(year * 100 + month)
		if month == 12:
			year = year + 1
			month = 1
		else:
			month += 1

	return yymm


def r2_to_adj_r2(r2, n, p):
	return 1 - (1 - r2) * (n - 1)/(n - p - 1)


def fit_ff3(data):
	lm = linear_model.LinearRegression()
	X = data[['Rm3-Rf','SMB3','HML3']]
	y = data['mret']
	lm.fit(X,y)
	intercept = lm.intercept_
	r2 = lm.score(X,y)
	n = X.shape[0]
	p = 3
	adj_r2 = r2_to_adj_r2(r2, n, p)
	return intercept, adj_r2


def fit_ff4(data):
	lm = linear_model.LinearRegression()
	X = data[['Rm3-Rf','Small-Mid','Mid-Large', 'HML3']]
	y = data['mret']
	lm.fit(X,y)
	intercept = lm.intercept_
	r2 = lm.score(X,y)
	n = X.shape[0]
	p = 4
	adj_r2 = r2_to_adj_r2(r2, n, p)
	return intercept, adj_r2


def fit_ff5(data):
	lm = linear_model.LinearRegression()
	X = data[['Mkt5-RF','SMB5','HML5','RMW5','CMA5']]
	y = data['mret']
	lm.fit(X,y)
	intercept = lm.intercept_
	r2 = lm.score(X,y)
	n = X.shape[0]
	p = 5
	adj_r2 = r2_to_adj_r2(r2, n, p)
	return intercept, adj_r2


def fit_ff6(data):
	lm = linear_model.LinearRegression()
	X = data[['Mkt5-RF','Small-Mid','Mid-Large','HML5','RMW5','CMA5']]
	y = data['mret']
	lm.fit(X,y)
	intercept = lm.intercept_
	r2 = lm.score(X,y)
	n = X.shape[0]
	p = 6
	adj_r2 = r2_to_adj_r2(r2, n, p)
	return intercept, adj_r2


def addUpperFees(data, fund_name):
	for index, row in data.iterrows():
		mret = row['mret']
		year = int(row['yymm'] // 100)
		fee = upper_fees[upper_fees['Name'] == fund_name].values[0, year - 1990 + 1] /12
		data.set_value(index, 'mret', mret + fee)


def addLowerFees(data, fund_name):
	fee = calculateLowerFees(fund_name)
	for index, row in data.iterrows():
		mret = row['mret']
		year = int(row['yymm'] // 100)
		f = fee[year - 1990]
		data.set_value(index, 'mret', mret + f)


def calculateLowerFees(fund_name):
	file_name = files[fund_names.index(fund_name)]

	weight = pd.read_excel('weights\\' + file_name)
	weight = weight[['CUSIP','Portfolio Weighting %']]
	lower = pd.read_excel('fees\\' + file_name)
	lower = lower[lower.columns.values[1:]]
	df = pd.merge(weight, lower)
	df = df.fillna(0)
	matrix = df.values

	weight = matrix[:,1]
	weight = weight.astype(np.float64)

	lower = matrix[:,2:]

	fee = lower.copy()

	for index, w in np.nditer([np.arange(len(weight)),weight]):
		fee[index,:] *= w/100 /12

	return np.sum(fee, axis = 0)



def writeTitles(w, fund_name):
	w.write('%s' %(fund_name))
	w.write(',Three Factor Adjusted R^2')
	w.write(',Three Factor Intercept')
	w.write(',Four Factor Adjusted R^2')
	w.write(',Four Factor Intercept')
	w.write(',Five Factor Adjusted R^2')
	w.write(',Five Factor Intercept')
	w.write(',Six Factor Adjusted R^2')
	w.write(',Six Factor Intercept\n')


def output():
	yymm = make_yymm()
	factor = createFamaFrenchFactor()
	w = open('result.csv', 'w+')
	for fund_name in fund_names:
		print(fund_name)
		writeTitles(w, fund_name.decode('utf-8'))
		monthly_return = funds[funds['Name'] == fund_name].values[0][1:]

		d = {'yymm': yymm, 'mret':monthly_return}
		fund_return = pd.DataFrame(d)
		fund = pd.merge(factor,fund_return)
		fund = fund.dropna()
		fund['mret'] = fund['mret'].astype(np.float64)
		fund.index = fund['yymm'].values
		
		# no fees
		w.write('None')
		intercept,r2 = fit_ff3(fund)
		w.write(',%s,%s' %(intercept, r2))
		intercept, r2 = fit_ff4(fund)
		w.write(',%s,%s' %(intercept, r2))
		intercept, r2 = fit_ff5(fund)
		w.write(',%s,%s' %(intercept, r2))
		intercept, r2 = fit_ff6(fund)
		w.write(',%s,%s' %(intercept, r2))
		w.write('\n')
		print('Finsihed None for %s' %fund_name)

		# upper fees
		w.write('Upper')
		addUpperFees(fund, fund_name)

		intercept,r2 = fit_ff3(fund)
		w.write(',%s,%s' %(intercept, r2))
		intercept, r2 = fit_ff4(fund)
		w.write(',%s,%s' %(intercept, r2))
		intercept, r2 = fit_ff5(fund)
		w.write(',%s,%s' %(intercept, r2))
		intercept, r2 = fit_ff6(fund)
		w.write(',%s,%s' %(intercept, r2))
		w.write('\n')
		print('Finsihed Upper for %s' %fund_name)

		# upper + lower fees
		w.write('Lower')
		addLowerFees(fund, fund_name)

		intercept,r2 = fit_ff3(fund)
		w.write(',%s,%s' %(intercept, r2))
		intercept, r2 = fit_ff4(fund)
		w.write(',%s,%s' %(intercept, r2))
		intercept, r2 = fit_ff5(fund)
		w.write(',%s,%s' %(intercept, r2))
		intercept, r2 = fit_ff6(fund)
		w.write(',%s,%s' %(intercept, r2))
		w.write('\n')
		print('Finsihed Lower for %s' %fund_name)
		w.write('\n')

	w.close()




def main():
	output()


if __name__ == '__main__':
	main()








