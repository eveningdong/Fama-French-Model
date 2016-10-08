import os
import pandas as pd
import numpy as np
from pandas import DataFrame
import statsmodels.api as sm
import statsmodels.formula.api as smf


# read data
def read_data():
	test = pd.read_excel('Fund of Funds-US Equity.xlsx')
	fffactor= pd.read_excel('FactorReturn.xlsx')
	newfactor= pd.read_excel('Russell Factor Returns 84 to 16.xlsx')
	raw_cpi = pd.read_csv('CPI.csv')
	
	return test, fffactor, newfactor, raw_cpi


# create Fama-French factor
def create_ff(fffactor, newfactor):
	fffactor = fffactor[fffactor.columns[:-1]]
	newfactor['yymm'] = newfactor['yymm'] // 100
	newfactor[newfactor.columns[1:3]] = newfactor[newfactor.columns[1:3]] * 100
	merged = pd.merge(fffactor, newfactor)
	
	return merged


# create cpi
def create_cpi(raw_cpi):
	raw_cpi = raw_cpi[raw_cpi.columns[:-2]]
	cpi = DataFrame(columns=('yymm', 'cpi'))

	for i in range(0, raw_cpi.shape[0]):
	    year = raw_cpi.iloc[i,0]
	    for j in range(1,13):
	        cpi_value = raw_cpi.iloc[i,j]
	        month = j
	        yymm = year * 100 + month
	        cpi.loc[i*12 + (j-1)] = [yymm, cpi_value]

	cpi['yymm'] = cpi['yymm'].astype(np.int32)
	cpi['cpi'] = cpi['cpi'] / 202.6
	cpi = cpi.dropna()

	return cpi


# create test data
def make_yymm(start_year,start_month,end_year,end_month):
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


def extract_data(row, yymm):
	fundname = row[0]
	monthly_return = row[34:347]
	d = {'yymm': yymm, 'mret':monthly_return}
	fund = pd.DataFrame(d)
	return fundname, fund


def create_test_data(fund, merged):
	test_data = pd.merge(merged,fund)
	test_data = test_data.dropna()
	test_data['mret'] = test_data['mret'].astype(np.float64)
	
	return test_data


def fit_ff3(data):
	lm = smf.ols(formula='mret ~ Rm3_Rf + SMB3 + HML3', data=data).fit()
	intercept = lm.params[0]
	r2 = lm.rsquared_adj
	return intercept, r2


def fit_ff4(data):
	lm = smf.ols(formula='mret ~ Rm3_Rf + Small_Mid + Mid_Large + HML3', data=data).fit()
	intercept = lm.params[0]
	r2 = lm.rsquared_adj
	return intercept, r2


def fit_ff5(data):
	lm = smf.ols(formula='mret ~ Mkt5_RF + SMB5 + HML5 + RMW5 + CMA5', data=data).fit()
	intercept = lm.params[0]
	r2 = lm.rsquared_adj
	return intercept, r2


def fit_ff6(data):
	lm = smf.ols(formula='mret ~ Mkt5_RF + Small_Mid + Mid_Large + HML5 + RMW5 + CMA5', data=data).fit()
	intercept = lm.params[0]
	r2 = lm.rsquared_adj
	return intercept, r2

def output(test, merged):
	yymm = make_yymm(1990,6,2016,6)
	w = open('result.csv', 'w+')
	for i in range(0, test.shape[0]):
		fundname, fund = extract_data(test.loc[i], yymm)
		w.write(fundname.encode('utf-8'))
		w.write('\n')	
		test_data = create_test_data(fund, merged)
		intercept,r2 = fit_ff3(test_data)
		w.write('Three Factor, %s, %s\n' %(intercept, r2))
		intercept, r2 = fit_ff4(test_data)
		w.write('Four Factor, %s, %s\n' %(intercept, r2))
		intercept, r2 = fit_ff5(test_data)
		w.write('Five Factor, %s, %s\n' %(intercept, r2))
		intercept, r2 = fit_ff6(test_data)
		w.write('Six Factor, %s, %s\n' %(intercept, r2))
		w.write('\n')	
	w.close()
		

def rename(merged):
	merged.rename(columns={'Rm3-Rf': 'Rm3_Rf', 'Mkt5-RF':'Mkt5_RF','Small-Mid':'Small_Mid','Mid-Large' : 'Mid_Large'}, inplace=True)
	return merged


def main():
	test, fffactor, newfactor, raw_cpi = read_data()
	merged = create_ff(fffactor, newfactor)
	merged = rename(merged)
	output(test, merged)


if __name__ == '__main__':
	main()
