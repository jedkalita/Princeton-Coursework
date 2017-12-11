import numpy as np
import matplotlib.pyplot as plt
import pandas
from cvxpy import *
from xlrd import open_workbook
import functools
import math
import cvxopt as opt
from cvxopt import blas, solvers
import numpy
from scipy.stats import norm

#load in the data
df = pandas.read_excel('Assets_4.xlsx')
#print(df.columns)

#get all 4 assets
us_equity = df['U.S. Equity'].values
t_bond = df['Treasury Bond'].values
corporate_bond = df['Corporate Bond'].values
cash = df['Cash'].values
#print(cash)

#convert r to R by adding 1
for i in range(len(us_equity)):
    us_equity[i] = us_equity[i] + 1
    t_bond[i] = t_bond[i] + 1
    corporate_bond[i] = corporate_bond[i] + 1
    cash[i] = cash[i] + 1 #R

#function to calculate monthly geometric mean of a series
def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

#calculate the monthly geometric means (R)
gmean_us_equity_monthly_R = geo_mean(us_equity)
print('Geo-mean monthly US Equities (R) = %f' % gmean_us_equity_monthly_R)
gmean_t_bond_monthly_R = geo_mean(t_bond)
print('Geo-mean monthly Treasury Bonds (R) = %f' % gmean_t_bond_monthly_R)
gmean_corporate_bond_monthly_R = geo_mean(corporate_bond)
print('Geo-mean monthly Corporate Bonds (R) = %f' % gmean_corporate_bond_monthly_R)
gmean_cash_monthly_R = geo_mean(cash)
print('Geo-mean monthly Cash (R) = %f' % gmean_cash_monthly_R)

#calculate the monthly geometric means (r)
gmean_us_equity_monthly_r = gmean_us_equity_monthly_R - 1
print('Geo-mean monthly US Equities (r) = %f' % gmean_us_equity_monthly_r)
gmean_t_bond_monthly_r = gmean_t_bond_monthly_R - 1
print('Geo-mean monthly Treasury Bonds (r) = %f' % gmean_t_bond_monthly_r)
gmean_corporate_bond_monthly_r = gmean_corporate_bond_monthly_R - 1
print('Geo-mean monthly Corporate Bonds (r) = %f' % gmean_corporate_bond_monthly_r)
gmean_cash_monthly_r = gmean_cash_monthly_R - 1
print('Geo-mean monthly Cash (r) = %f' % gmean_cash_monthly_r)

#now, calculate annual geometric means (R)
gmean_us_equity_annually_R = math.pow(gmean_us_equity_monthly_R, 12)
print('Geo-mean annually US Equities (R) = %f' % gmean_us_equity_annually_R)
gmean_t_bond_annually_R = math.pow(gmean_t_bond_monthly_R, 12)
print('Geo-mean annually Treasury Bonds (R) = %f' % gmean_t_bond_annually_R)
gmean_corporate_bond_annually_R = math.pow(gmean_corporate_bond_monthly_R, 12)
print('Geo-mean annually Corporate Bonds (R) = %f' % gmean_corporate_bond_annually_R)
gmean_cash_annually_R = math.pow(gmean_cash_monthly_R, 12)
print('Geo-mean annually Cash (R) = %f' % gmean_cash_annually_R)

#now, calculate annual geometric means (r)
gmean_us_equity_annually_r = gmean_us_equity_annually_R - 1
print('Geo-mean annually US Equities (r) = %f' % gmean_us_equity_annually_r)
gmean_t_bond_annually_r = gmean_t_bond_annually_R - 1
print('Geo-mean annually Treasury Bonds (r) = %f' % gmean_t_bond_annually_r)
gmean_corporate_bond_annually_r = gmean_corporate_bond_annually_R - 1
print('Geo-mean annually Corporate Bonds (r) = %f' % gmean_corporate_bond_annually_r)
gmean_cash_annually_r = gmean_cash_annually_R - 1
print('Geo-mean annually Cash (r) = %f' % gmean_cash_annually_r)

#now, calculate monthly standard deviation
us_equity_std_monthly = np.std(np.array(us_equity))
print("Std for US Equities monthly = %f" % us_equity_std_monthly)
t_bond_std_monthly = np.std(np.array(t_bond))
print("Std for Treasury Bonds monthly = %f" % t_bond_std_monthly)
corporate_bond_std_monthly = np.std(np.array(corporate_bond))
print("Std for Corporate Bonds monthly = %f" % corporate_bond_std_monthly)
cash_std_monthly = np.std(np.array(cash))
print("Std for Cash monthly = %f" % cash_std_monthly)

#now, calculate monthly volatility (variance)
us_equity_var_monthly = math.pow(us_equity_std_monthly, 2)
print("Variance (volatility) for US Equities monthly = %f" % us_equity_var_monthly)
t_bond_var_monthly = math.pow(t_bond_std_monthly, 2)
print("Variance (volatility) for Treasury Bonds monthly = %f" % t_bond_var_monthly)
corporate_bond_var_monthly = math.pow(corporate_bond_std_monthly, 2)
print("Variance (volatility) for Corporate Bonds monthly = %f" % corporate_bond_var_monthly)
cash_var_monthly = math.pow(cash_std_monthly, 2)
print("Variance (volatility) for Cash monthly = %f" % cash_var_monthly)

#now, calculate annualized standard deviation
us_equity_std_annually = math.sqrt(12) * us_equity_std_monthly
print("Std for US Equities annually = %f" % us_equity_std_annually)
t_bond_std_annually = math.sqrt(12) * t_bond_std_monthly
print("Std for Treasury Bonds annually = %f" % t_bond_std_annually)
corporate_bond_std_annually = math.sqrt(12) * corporate_bond_std_monthly
print("Std for Corporate Bonds annually = %f" % corporate_bond_std_annually)
cash_std_annually = math.sqrt(12) * cash_std_monthly
print("Std for Cash annually = %f" % cash_std_annually)

#now, calculate annualized volatility (variance)
us_equity_var_annually = math.pow(us_equity_std_annually, 2)
print("Variance (volatility) for US Equities annually = %f" % us_equity_var_annually)
t_bond_var_annually = math.pow(t_bond_std_annually, 2)
print("Variance (volatility) for Treasury Bonds annually = %f" % t_bond_var_annually)
corporate_bond_var_annually = math.pow(corporate_bond_std_annually, 2)
print("Variance (volatility) for Corporate Bonds annually = %f" % corporate_bond_var_annually)
cash_var_annually = math.pow(cash_std_annually, 2)
print("Variance (volatility) for Cash annually = %f" % cash_var_annually)

#now, make the monthly correlation matrix
c = list()
c.append(np.array(us_equity))
c.append(np.array(t_bond))
c.append(np.array(corporate_bond))
c.append(np.array(cash))
correlation_mat_monthly = np.corrcoef(np.array(c))

#annualized correlation matrix
correlation_mat_annually = 12 * correlation_mat_monthly


#now, make the monthly covariance matrix
covariance_mat_monthly = np.cov(np.array(c))

#annualized covariance matrix
covariance_mat_annually = 12 * covariance_mat_monthly


#projected monthly returns for the 4 assets
us_equity_projected_monthly_R = 1.00565
t_bond_projected_monthly_R = 1.00206
corporate_bond_monthly_R = 1.00368
cash_monthly_R = 1.00021

#store projected monthly returns (R) for the 4 assets in an array
mu_R = np.array([us_equity_projected_monthly_R, t_bond_projected_monthly_R,
                 corporate_bond_monthly_R, cash_monthly_R])
#print(mu_R)
#store projected monthly returns (r) for the 4 assets in an array
mu_r = np.array([us_equity_projected_monthly_R - 1, t_bond_projected_monthly_R - 1,
                 corporate_bond_monthly_R - 1, cash_monthly_R - 1])
#print(mu_r)

#now, generate 10000 scenarios each of 120 periods (10 years) for 4 assets based on R
scenario_matrix_R = np.random.multivariate_normal(mu_R, covariance_mat_monthly, (10000, 120))
#now, generate 10000 scenarios each of 120 periods (10 years) for 4 assets based on r
scenario_matrix_r = np.random.multivariate_normal(mu_r, covariance_mat_monthly, (10000, 120))
#10000 scenarios x 120 periods x 4 assets (shape of the matrix)
#print(scenario_matrix.shape)


h = 0.05 #the loss tolerance for VaR and CVaR calculation

#CVaR and VaR from sample - US Equity
#historical
number = len(us_equity) #find the total number of samples there is
us_equity_array = np.sort(np.array(us_equity)) #get the sorted list
#print('printing US equity')
var_point = int(h * number) #get the 5th percentile point
VaR_us_equity_sample = us_equity_array[var_point]
print('VaR of historical US Equity from sample = %f' % VaR_us_equity_sample)
#CVar calculation - average out until VaR point
sum = 0
for i in range(var_point):
    sum = float(sum + us_equity_array[i])
sum = float(sum / var_point)
print('CVaR of historical US Equity from sample = %f' % sum)

#simulated
us_equity_period_1_R = scenario_matrix_R[:, 0, 0]
us_equity_array_s = np.sort(np.array(us_equity_period_1_R)) #get the sorted list
VaR = us_equity_array_s[500] #since 0.05 * 10000 = 500
print('VaR of simulated US Equity from sample = %f' % VaR)
#CVar calculation - average out until VaR point
sum = 0
for i in range(500):
    sum = float(sum + us_equity_array_s[i])
sum = float(sum / 500)
print('CVaR of simulated US Equity from sample = %f' % sum)



#CVaR and VaR from sample - Treasury Bonds
number = len(t_bond) #find the total number of samples there is
t_bond_array = np.sort(np.array(t_bond)) #get the sorted list
'''print('printing T-Bond')
print(t_bond_array)'''
var_point = int(h * number) #get the 5th percentile point
VaR_t_bond_sample = t_bond_array[var_point]
print('VaR of historical Treasury Bond from sample = %f' % VaR_t_bond_sample)
#CVar calculation - average out until VaR point
sum = 0
for i in range(var_point):
    sum = float(sum + t_bond_array[i])
sum = float(sum / var_point)
print('CVaR of historical Treasury Bond from sample = %f' % sum)

#simulated
t_bond_period_1_R = scenario_matrix_R[:, 0, 1]
t_bond_array_s = np.sort(np.array(t_bond_period_1_R)) #get the sorted list
VaR = t_bond_array_s[500] #since 0.05 * 10000 = 500
print('VaR of simulated Treasury Bond from sample = %f' % VaR)
#CVar calculation - average out until VaR point
sum = 0
for i in range(500):
    sum = float(sum + t_bond_array_s[i])
sum = float(sum / 500)
print('CVaR of simulated Treasury Bond from sample = %f' % sum)


#CVaR and VaR from sample - Corporate Bonds
number = len(corporate_bond) #find the total number of samples there is
corporate_bond_array = np.sort(np.array(corporate_bond)) #get the sorted list
'''print('printing T-Bond')
print(t_bond_array)'''
var_point = int(h * number) #get the 5th percentile point
VaR_corporate_bond_sample = corporate_bond_array[var_point]
print('VaR of historical Corporate Bond from sample = %f' % VaR_corporate_bond_sample)
#CVar calculation - average out until VaR point
sum = 0
for i in range(var_point):
    sum = float(sum + corporate_bond_array[i])
sum = float(sum / var_point)
print('CVaR of historical Corporate Bond from sample = %f' % sum)

#simulated
corporate_bond_period_1_R = scenario_matrix_R[:, 0, 2]
corporate_bond_array_s = np.sort(np.array(corporate_bond_period_1_R)) #get the sorted list
VaR = corporate_bond_array_s[500] #since 0.05 * 10000 = 500
print('VaR of simulated Corporate Bond from sample = %f' % VaR)
#CVar calculation - average out until VaR point
sum = 0
for i in range(500):
    sum = float(sum + corporate_bond_array_s[i])
sum = float(sum / 500)
print('CVaR of simulated Corporate Bond from sample = %f' % sum)


#CVaR and VaR from sample - cash
number = len(cash) #find the total number of samples there is
cash_array = np.sort(np.array(cash)) #get the sorted list
'''print('printing T-Bond')
print(t_bond_array)'''
var_point = int(h * number) #get the 5th percentile point
VaR_cash_sample = cash_array[var_point]
print('VaR of historical Cash from sample = %f' % VaR_cash_sample)
#CVar calculation - average out until VaR point
sum = 0
for i in range(var_point):
    sum = float(sum + cash_array[i])
sum = float(sum / var_point)
print('CVaR of historical Cash from sample = %f' % sum)

#simulated
cash_period_1_R = scenario_matrix_R[:, 0, 3]
cash_array_s = np.sort(np.array(cash_period_1_R)) #get the sorted list
VaR = cash_array_s[500] #since 0.05 * 10000 = 500
print('VaR of simulated Cash from sample = %f' % VaR)
#CVar calculation - average out until VaR point
sum = 0
for i in range(500):
    sum = float(sum + cash_array_s[i])
sum = float(sum / 500)
print('CVaR of simulated Cash from sample = %f' % sum)



#get all the 1st period US Equity R's
us_equity_period_1_R = scenario_matrix_R[:, 0, 0]
#print(len(us_equity_period_1_R))
#calculate the geometric mean of all the 1st period US Equity R's
gmean_us_equity_period_1_R = geo_mean(us_equity_period_1_R)
#print(gmean_us_equity_period_1_R)
#calculate the geometric mean of all the 1st period US Equity r's
gmean_us_equity_period_1_r = gmean_us_equity_period_1_R - 1
#print(gmean_us_equity_period_1_r)
#calculate the monthly standard deviation of all the 1st period US Equity R's
us_equity_std_monthly_period_1 = np.std(np.array(us_equity_period_1_R))
#print(us_equity_std_monthly_period_1)
#now, calculate cvar and var for 1st period of US Equities from 10000 scenarios
CVaR_us_equity_period_1 = (h ** -1) * norm.pdf(norm.ppf(h)) * us_equity_std_monthly_period_1 \
                          - gmean_us_equity_period_1_r
CVaR_us_equity_period_1 = CVaR_us_equity_period_1 * 100 #convert it to %
print('CVaR for 1st period US Equities from mean and variance = %f' % CVaR_us_equity_period_1)
VaR_us_equity_period_1 = norm.ppf(1 - h) * us_equity_std_monthly_period_1 - gmean_us_equity_period_1_r
VaR_us_equity_period_1 = VaR_us_equity_period_1 * 100 #convert it to %
print('VaR for 1st period US Equities from mean and variance = %f' % VaR_us_equity_period_1)

#calculate CVaR and VaR for US Equities historically
CVaR_us_equity_historical = (h ** -1) * norm.pdf(norm.ppf(h)) * us_equity_std_monthly \
                          - gmean_us_equity_monthly_r
CVaR_us_equity_historical = CVaR_us_equity_historical * 100 #convert it to %
print('CVaR for historical US Equities from mean and variance = %f' % CVaR_us_equity_historical)
VaR_us_equity_historical = norm.ppf(1 - h) * us_equity_std_monthly - gmean_us_equity_monthly_r
VaR_us_equity_historical = VaR_us_equity_historical * 100 #convert it to %
print('VaR for historical US Equities from mean and variance = %f' % VaR_us_equity_historical)

#get all the 1st period T-bond R's
t_bond_period_1_R = scenario_matrix_R[:, 0, 1]
#print(len(t_bond_period_1_R))
#calculate the geometric mean of all the 1st period T-bond R's
gmean_t_bond_period_1_R = geo_mean(t_bond_period_1_R)
#print(gmean_t_bond_period_1_R)
#calculate the geometric mean of all the 1st period T-bond r's
gmean_t_bond_period_1_r = gmean_t_bond_period_1_R - 1
#print(gmean_t_bond_period_1_r)
#calculate the monthly standard deviation of all the 1st period T-bond R's
t_bond_std_monthly_period_1 = np.std(np.array(t_bond_period_1_R))
#print(t_bond_std_monthly_period_1)
#now, calculate cvar and var for 1st period of T-bond from 10000 scenarios
CVaR_t_bond_period_1 = (h ** -1) * norm.pdf(norm.ppf(h)) * t_bond_std_monthly_period_1 \
                          - gmean_t_bond_period_1_r
CVaR_t_bond_period_1 = CVaR_t_bond_period_1 * 100 #convert it to %
print('CVaR for 1st period Treasury Bonds from mean and variance = %f' % CVaR_t_bond_period_1)
VaR_t_bond_period_1 = norm.ppf(1 - h) * t_bond_std_monthly_period_1 - gmean_t_bond_period_1_r
VaR_t_bond_period_1 = VaR_t_bond_period_1 * 100 #convert it to %
print('VaR for 1st period Treasury Bonds from mean and variance = %f' % VaR_t_bond_period_1)

#calculate CVaR and VaR for Treasury Bonds historically
CVaR_t_bond_historical = (h ** -1) * norm.pdf(norm.ppf(h)) * t_bond_std_monthly \
                          - gmean_t_bond_monthly_r
CVaR_t_bond_historical = CVaR_t_bond_historical * 100 #convert it to %
print('CVaR for historical Treasury Bonds from mean and variance = %f' % CVaR_t_bond_historical)
VaR_t_bond_historical = norm.ppf(1 - h) * t_bond_std_monthly - gmean_t_bond_monthly_r
VaR_t_bond_historical = VaR_t_bond_historical * 100 #convert it to %
print('VaR for historical Treasury Bonds from mean and variance = %f' % VaR_t_bond_historical)

#get all the 1st period Corporate-bond R's
corporate_bond_period_1_R = scenario_matrix_R[:, 0, 2]
#print(len(corporate_bond_period_1_R))
#calculate the geometric mean of all the 1st period Corporate-bond R's
gmean_corporate_bond_period_1_R = geo_mean(corporate_bond_period_1_R)
#print(gmean_corporate_bond_period_1_R)
#calculate the geometric mean of all the 1st period Corporate-bond r's
gmean_corporate_bond_period_1_r = gmean_corporate_bond_period_1_R - 1
#print(gmean_corporate_bond_period_1_r)
#calculate the monthly standard deviation of all the 1st period Corporate-bond R's
corporate_bond_std_monthly_period_1 = np.std(np.array(corporate_bond_period_1_R))
#print(corporate_bond_std_monthly_period_1)
#now, calculate cvar and var for 1st period of Corporate-bond from 10000 scenarios
CVaR_corporate_bond_period_1 = (h ** -1) * norm.pdf(norm.ppf(h)) * corporate_bond_std_monthly_period_1 \
                          - gmean_corporate_bond_period_1_r
CVaR_corporate_bond_period_1 = CVaR_corporate_bond_period_1 * 100 #convert it to %
print('CVaR for 1st period Corporate Bonds from mean and variance = %f' % CVaR_corporate_bond_period_1)
VaR_corporate_bond_period_1 = norm.ppf(1 - h) * corporate_bond_std_monthly_period_1 - gmean_corporate_bond_period_1_r
VaR_corporate_bond_period_1 = VaR_corporate_bond_period_1 * 100 #convert it to %
print('VaR for 1st period Corporate Bonds from mean and variance = %f' % VaR_corporate_bond_period_1)

#calculate CVaR and VaR for Corporate Bonds historically
CVaR_corporate_bond_historical = (h ** -1) * norm.pdf(norm.ppf(h)) * corporate_bond_std_monthly \
                          - gmean_corporate_bond_monthly_r
CVaR_corporate_bond_historical = CVaR_corporate_bond_historical * 100 #convert it to %
print('CVaR for historical Corporate Bonds from mean and variance = %f' % CVaR_corporate_bond_historical)
VaR_corporate_bond_historical = norm.ppf(1 - h) * corporate_bond_std_monthly - gmean_corporate_bond_monthly_r
VaR_corporate_bond_historical = VaR_corporate_bond_historical * 100 #convert it to %
print('VaR for historical Corporate Bonds from mean and variance = %f' % VaR_corporate_bond_historical)

#get all the 1st period Cash R's
cash_period_1_R = scenario_matrix_R[:, 0, 3]
#print(len(cash_period_1_R))
#calculate the geometric mean of all the 1st period Cash R's
gmean_cash_period_1_R = geo_mean(cash_period_1_R)
#print(gmean_cash_period_1_R)
#calculate the geometric mean of all the 1st period Cash r's
gmean_cash_period_1_r = gmean_cash_period_1_R - 1
#print(gmean_cash_period_1_r)
#calculate the monthly standard deviation of all the 1st period Cash R's
cash_std_monthly_period_1 = np.std(np.array(cash_period_1_R))
#print(cash_std_monthly_period_1)
#now, calculate cvar and var for 1st period of Cash from 10000 scenarios
CVaR_cash_period_1 = (h ** -1) * norm.pdf(norm.ppf(h)) * cash_std_monthly_period_1 \
                          - gmean_cash_period_1_r
CVaR_cash_period_1 = CVaR_cash_period_1 * 100 #convert it to %
print('CVaR for 1st period Cash from mean and variance = %f' % CVaR_cash_period_1)
VaR_cash_period_1 = norm.ppf(1 - h) * cash_std_monthly_period_1 - gmean_cash_period_1_r
VaR_cash_period_1 = VaR_cash_period_1 * 100 #convert it to %
print('VaR for 1st period Cash from mean and variance = %f' % VaR_cash_period_1)

#calculate CVaR and VaR for Cash historically
CVaR_cash_historical = (h ** -1) * norm.pdf(norm.ppf(h)) * cash_std_monthly \
                          - gmean_cash_monthly_r
CVaR_cash_historical = CVaR_cash_historical * 100 #convert it to %
print('CVaR for historical Cash from mean and variance = %f' % CVaR_cash_historical)
VaR_cash_historical = norm.ppf(1 - h) * cash_std_monthly - gmean_cash_monthly_r
VaR_cash_historical = VaR_cash_historical * 100 #convert it to %
print('VaR for historical Cash from mean and variance = %f' % VaR_cash_historical)


#now, for time period = 5 years, simulate policies A, B, C
wealth = 100000 #the initial wealth
prob_scenario = float(1/10000) #the probability of one scenario being selected

#Policy (A) simulator
us_equity_A_weight = 0.0
t_bond_A_weight = 0.9
corporate_bond_A_weight = 0.0
cash_A_weight = 0.1

#print(scenario_matrix_R[2, :, :])

time_limit = 60 #5 years = 60 monthly periods
#now, we will make a matrix which will be of size (10000, 1) and will store the wealth accumulated
#at the end of time_limit for each scenario
wealth_time_limit_scenario_A = list()
for i in range(10000): #for each scenario
    for j in range(time_limit): #for each period in the 60 month period
        #get return for period j
        X_us_equity_A = scenario_matrix_R[i, j, 0] * us_equity_A_weight * wealth
        X_t_bond_A = scenario_matrix_R[i, j, 1] * t_bond_A_weight * wealth
        #print(X_t_bond_A)
        X_corporate_bond_A = scenario_matrix_R[i, j, 2] * corporate_bond_A_weight * wealth
        X_cash_A = scenario_matrix_R[i, j, 3] * cash_A_weight * wealth
        #now, calculate the wealth generated at the end of time period j
        wealth = X_us_equity_A + X_t_bond_A + X_corporate_bond_A + X_cash_A
        '''if (i == 0 and j == 4):
            print(wealth)'''
        #print(wealth)
        #now, rebalance the portfolio after each month (period)
        '''us_equity_A_weight = float (X_us_equity_A / wealth)
        t_bond_A_weight = float(X_t_bond_A / wealth)
        corporate_bond_A_weight = float(X_corporate_bond_A / wealth)
        cash_A_weight = float(X_cash_A / wealth)'''

    #wealth = X_us_equity_A + X_t_bond_A + X_corporate_bond_A + X_cash_A
    #now, at the end of 60 months, now add the wealth accumulated for this scenario to wealth_time_limit_scenario
    wealth_time_limit_scenario_A.append(wealth)
    wealth = 100000 #again make wealth to be 10000


#create an np array for wealth_time_limit_scenario
wealth_time_limit_scenario_A_array = np.array(wealth_time_limit_scenario_A)
#now, take the expectation and add it to find probability of policy A
wealth_A = np.sum(prob_scenario * wealth_time_limit_scenario_A_array)
print('Policy A generated wealth after 5 years : %f' % wealth_A)
#create the indicator array
indicator_wealth_A = list()
for i in range(10000):
    if wealth_time_limit_scenario_A_array[i] >= 120000:
        indicator_wealth_A.append(1)
    else:
        indicator_wealth_A.append(0)
indicator_wealth_array_A = np.array(indicator_wealth_A)
#now calculate the probability of success
prob_A = np.sum(prob_scenario * indicator_wealth_array_A)
print('Policy A probability of success after 5 years: %f' % prob_A)


#Policy (B) simulator
wealth = 100000 #the initial wealth
us_equity_B_weight = 0.4
t_bond_B_weight = 0.3
corporate_bond_B_weight = 0.2
cash_B_weight = 0.1

#now, we will make a matrix which will be of size (10000, 1) and will store the wealth accumulated
#at the end of time_limit for each scenario
wealth_time_limit_scenario_B = list()
for i in range(10000): #for each scenario
    for j in range(time_limit): #for each period in the 60 month period
        #get return for period j
        X_us_equity_B = scenario_matrix_R[i, j, 0] * us_equity_B_weight * wealth
        X_t_bond_B = scenario_matrix_R[i, j, 1] * t_bond_B_weight * wealth
        X_corporate_bond_B = scenario_matrix_R[i, j, 2] * corporate_bond_B_weight * wealth
        X_cash_B = scenario_matrix_R[i, j, 3] * cash_B_weight * wealth
        #now, calculate the wealth generated at the end of time period j
        wealth = X_us_equity_B + X_t_bond_B + X_corporate_bond_B + X_cash_B
        #now, rebalance the portfolio after each month (period)
        '''us_equity_B_weight = float (X_us_equity_B / wealth)
        t_bond_B_weight = float(X_t_bond_B / wealth)
        corporate_bond_B_weight = float(X_corporate_bond_B / wealth)
        cash_B_weight = float(X_cash_B / wealth)'''

    #wealth = X_us_equity_B + X_t_bond_B + X_corporate_bond_B + X_cash_B
    #now, at the end of 60 months, now add the wealth accumulated for this scenario to wealth_time_limit_scenario
    wealth_time_limit_scenario_B.append(wealth)
    wealth = 100000 #again make wealth to be 10000

#create an np array for wealth_time_limit_scenario
wealth_time_limit_scenario_B_array = np.array(wealth_time_limit_scenario_B)
#now, take the expectation and add it to find probability of policy A
wealth_B = np.sum(prob_scenario * wealth_time_limit_scenario_B_array)
print('Policy B generated wealth after 5 years : %f' % wealth_B)
#create the indicator array
indicator_wealth_B = list()
for i in range(10000):
    if wealth_time_limit_scenario_B_array[i] >= 120000:
        indicator_wealth_B.append(1)
    else:
        indicator_wealth_B.append(0)
indicator_wealth_array_B = np.array(indicator_wealth_B)
#now calculate the probability of success
prob_B = np.sum(prob_scenario * indicator_wealth_array_B)
print('Policy B probability of success after 5 years: %f' % prob_B)


#Policy (C) simulator
wealth = 100000 #the initial wealth
us_equity_C_weight = 0.25
t_bond_C_weight = 0.25
corporate_bond_C_weight = 0.25
cash_C_weight = 0.25

#now, we will make a matrix which will be of size (10000, 1) and will store the wealth accumulated
#at the end of time_limit for each scenario
wealth_time_limit_scenario_C = list()
for i in range(10000): #for each scenario
    for j in range(time_limit): #for each period in the 60 month period
        #get return for period j
        X_us_equity_C = scenario_matrix_R[i, j, 0] * us_equity_C_weight * wealth
        X_t_bond_C = scenario_matrix_R[i, j, 1] * t_bond_C_weight * wealth
        X_corporate_bond_C = scenario_matrix_R[i, j, 2] * corporate_bond_C_weight * wealth
        X_cash_C = scenario_matrix_R[i, j, 3] * cash_C_weight * wealth
        #now, calculate the wealth generated at the end of time period j
        wealth = X_us_equity_C + X_t_bond_C + X_corporate_bond_C + X_cash_C
        #now, rebalance the portfolio after each month (period)
        '''us_equity_C_weight = float (X_us_equity_C / wealth)
        t_bond_C_weight = float(X_t_bond_C / wealth)
        corporate_bond_C_weight = float(X_corporate_bond_C / wealth)
        cash_C_weight = float(X_cash_C / wealth)'''

    #wealth = X_us_equity_C + X_t_bond_C + X_corporate_bond_C + X_cash_C
    #now, at the end of 60 months, now add the wealth accumulated for this scenario to wealth_time_limit_scenario
    wealth_time_limit_scenario_C.append(wealth)
    wealth = 100000 #again make wealth to be 10000

#create an np array for wealth_time_limit_scenario
wealth_time_limit_scenario_C_array = np.array(wealth_time_limit_scenario_C)
#now, take the expectation and add it to find probability of policy A
wealth_C = np.sum(prob_scenario * wealth_time_limit_scenario_C_array)
print('Policy C generated wealth after 5 years : %f' % wealth_C)
#create the indicator array
indicator_wealth_C = list()
for i in range(10000):
    if wealth_time_limit_scenario_C_array[i] >= 120000:
        indicator_wealth_C.append(1)
    else:
        indicator_wealth_C.append(0)
indicator_wealth_array_C = np.array(indicator_wealth_C)
#now calculate the probability of success
prob_C = np.sum(prob_scenario * indicator_wealth_array_C)
print('Policy C probability of success after 5 years: %f' % prob_C)







#now, for time period = 10 years, simulate policies A, B, C
wealth = 100000 #the initial wealth
prob_scenario = float(1/10000) #the probability of one scenario being selected

#Policy (A) simulator
us_equity_A_weight = 0.0
t_bond_A_weight = 0.9
corporate_bond_A_weight = 0.0
cash_A_weight = 0.1

time_limit = 120 #10 years = 120 monthly periods
#now, we will make a matrix which will be of size (10000, 1) and will store the wealth accumulated
#at the end of time_limit for each scenario
wealth_time_limit_scenario_A = list()
for i in range(10000): #for each scenario
    for j in range(time_limit): #for each period in the 60 month period
        #get return for period j
        X_us_equity_A = scenario_matrix_R[i, j, 0] * us_equity_A_weight * wealth
        X_t_bond_A = scenario_matrix_R[i, j, 1] * t_bond_A_weight * wealth
        #print(X_t_bond_A)
        X_corporate_bond_A = scenario_matrix_R[i, j, 2] * corporate_bond_A_weight * wealth
        X_cash_A = scenario_matrix_R[i, j, 3] * cash_A_weight * wealth
        #now, calculate the wealth generated at the end of time period j
        wealth = X_us_equity_A + X_t_bond_A + X_corporate_bond_A + X_cash_A
        #now, rebalance the portfolio after each month (period)
        '''us_equity_A_weight = float (X_us_equity_A / wealth)
        t_bond_A_weight = float(X_t_bond_A / wealth)
        corporate_bond_A_weight = float(X_corporate_bond_A / wealth)
        cash_A_weight = float(X_cash_A / wealth)'''

    #wealth = X_us_equity_A + X_t_bond_A + X_corporate_bond_A + X_cash_A
    #now, at the end of 60 months, now add the wealth accumulated for this scenario to wealth_time_limit_scenario
    wealth_time_limit_scenario_A.append(wealth)
    wealth = 100000 #again make wealth to be 10000
#print(wealth_time_limit_scenario_A)

#create an np array for wealth_time_limit_scenario
wealth_time_limit_scenario_A_array = np.array(wealth_time_limit_scenario_A)
#now, take the expectation and add it to find probability of policy A
wealth_A = np.sum(prob_scenario * wealth_time_limit_scenario_A_array)
print('Policy A generated wealth after 10 years : %f' % wealth_A)
#create the indicator array
indicator_wealth_A = list()
for i in range(10000):
    if wealth_time_limit_scenario_A_array[i] >= 120000:
        indicator_wealth_A.append(1)
    else:
        indicator_wealth_A.append(0)
indicator_wealth_array_A = np.array(indicator_wealth_A)
#now calculate the probability of success
prob_A = np.sum(prob_scenario * indicator_wealth_array_A)
print('Policy A probability of success after 10 years: %f' % prob_A)


#Policy (B) simulator
wealth = 100000 #the initial wealth
us_equity_B_weight = 0.4
t_bond_B_weight = 0.3
corporate_bond_B_weight = 0.2
cash_B_weight = 0.1

#now, we will make a matrix which will be of size (10000, 1) and will store the wealth accumulated
#at the end of time_limit for each scenario
wealth_time_limit_scenario_B = list()
for i in range(10000): #for each scenario
    for j in range(time_limit): #for each period in the 60 month period
        #get return for period j
        X_us_equity_B = scenario_matrix_R[i, j, 0] * us_equity_B_weight * wealth
        X_t_bond_B = scenario_matrix_R[i, j, 1] * t_bond_B_weight * wealth
        X_corporate_bond_B = scenario_matrix_R[i, j, 2] * corporate_bond_B_weight * wealth
        X_cash_B = scenario_matrix_R[i, j, 3] * cash_B_weight * wealth
        #now, calculate the wealth generated at the end of time period j
        wealth = X_us_equity_B + X_t_bond_B + X_corporate_bond_B + X_cash_B
        #now, rebalance the portfolio after each month (period)
        '''us_equity_B_weight = float (X_us_equity_B / wealth)
        t_bond_B_weight = float(X_t_bond_B / wealth)
        corporate_bond_B_weight = float(X_corporate_bond_B / wealth)
        cash_B_weight = float(X_cash_B / wealth)'''

    #wealth = X_us_equity_B + X_t_bond_B + X_corporate_bond_B + X_cash_B
    #now, at the end of 60 months, now add the wealth accumulated for this scenario to wealth_time_limit_scenario
    wealth_time_limit_scenario_B.append(wealth)
    wealth = 100000 #again make wealth to be 10000

#create an np array for wealth_time_limit_scenario
wealth_time_limit_scenario_B_array = np.array(wealth_time_limit_scenario_B)
#now, take the expectation and add it to find probability of policy A
wealth_B = np.sum(prob_scenario * wealth_time_limit_scenario_B_array)
print('Policy B generated wealth after 10 years : %f' % wealth_B)
#create the indicator array
indicator_wealth_B = list()
for i in range(10000):
    if wealth_time_limit_scenario_B_array[i] >= 120000:
        indicator_wealth_B.append(1)
    else:
        indicator_wealth_B.append(0)
indicator_wealth_array_B = np.array(indicator_wealth_B)
#now calculate the probability of success
prob_B = np.sum(prob_scenario * indicator_wealth_array_B)
print('Policy B probability of success after 10 years: %f' % prob_B)


#Policy (C) simulator
wealth = 100000 #the initial wealth
us_equity_C_weight = 0.25
t_bond_C_weight = 0.25
corporate_bond_C_weight = 0.25
cash_C_weight = 0.25

#now, we will make a matrix which will be of size (10000, 1) and will store the wealth accumulated
#at the end of time_limit for each scenario
wealth_time_limit_scenario_C = list()
for i in range(10000): #for each scenario
    for j in range(time_limit): #for each period in the 60 month period
        #get return for period j
        X_us_equity_C = scenario_matrix_R[i, j, 0] * us_equity_C_weight * wealth
        X_t_bond_C = scenario_matrix_R[i, j, 1] * t_bond_C_weight * wealth
        X_corporate_bond_C = scenario_matrix_R[i, j, 2] * corporate_bond_C_weight * wealth
        X_cash_C = scenario_matrix_R[i, j, 3] * cash_C_weight * wealth
        #now, calculate the wealth generated at the end of time period j
        wealth = X_us_equity_C + X_t_bond_C + X_corporate_bond_C + X_cash_C
        #now, rebalance the portfolio after each month (period)
        '''us_equity_C_weight = float (X_us_equity_C / wealth)
        t_bond_C_weight = float(X_t_bond_C / wealth)
        corporate_bond_C_weight = float(X_corporate_bond_C / wealth)
        cash_C_weight = float(X_cash_C / wealth)'''

    #now, at the end of 60 months, now add the wealth accumulated for this scenario to wealth_time_limit_scenario
    wealth_time_limit_scenario_C.append(wealth)
    wealth = 100000 #again make wealth to be 10000

#create an np array for wealth_time_limit_scenario
wealth_time_limit_scenario_C_array = np.array(wealth_time_limit_scenario_C)
#now, take the expectation and add it to find probability of policy A
wealth_C = np.sum(prob_scenario * wealth_time_limit_scenario_C_array)
print('Policy C generated wealth after 10 years : %f' % wealth_C)
#create the indicator array
indicator_wealth_C = list()
for i in range(10000):
    if wealth_time_limit_scenario_C_array[i] >= 120000:
        indicator_wealth_C.append(1)
    else:
        indicator_wealth_C.append(0)
indicator_wealth_array_C = np.array(indicator_wealth_C)
#now calculate the probability of success
prob_C = np.sum(prob_scenario * indicator_wealth_array_C)
print('Policy C probability of success after 10 years: %f' % prob_C)


#CVaR and VaR calculation for 10 year period - policy A
#sort the wealth array for policy A
wealth_time_limit_scenario_A_array_sorted = np.sort(wealth_time_limit_scenario_A_array,
                                                    kind='quicksort')
#print(wealth_time_limit_scenario_A_array_sorted)
#since h = 0.05, thus 501th element (index = 501) will be VaR of wealth
#print(wealth_time_limit_scenario_A_array_sorted[500])
VaR_A_goal = float(120000 - wealth_time_limit_scenario_A_array_sorted[500]) #VaR wrt goal
print('VaR with respect to the goal for policy A = %f' % VaR_A_goal)
#now, for CVaR, take the average of all the wealth upto index 499
sum = 0
for i in range(500):
    sum = float(sum + wealth_time_limit_scenario_A_array_sorted[i])
sum = float(sum / 500)
CVaR_A_goal = float(120000 - sum)
print('CVaR with respect to the goal for policy A = %f' % CVaR_A_goal)


#CVaR and VaR calculation for 10 year period - policy B
#sort the wealth array for policy B
wealth_time_limit_scenario_B_array_sorted = np.sort(wealth_time_limit_scenario_B_array,
                                                    kind='quicksort')
#print(wealth_time_limit_scenario_B_array_sorted)
#since h = 0.05, thus 501th element (index = 501) will be VaR of wealth
#print(wealth_time_limit_scenario_B_array_sorted[500])
VaR_B_goal = float(120000 - wealth_time_limit_scenario_B_array_sorted[500]) #VaR wrt goal
print('VaR with respect to the goal for policy B = %f' % VaR_B_goal)
#now, for CVaR, take the average of all the wealth upto index 499
sum = 0
for i in range(500):
    sum = float(sum + wealth_time_limit_scenario_B_array_sorted[i])
sum = float(sum / 500)
CVaR_B_goal = float(120000 - sum)
print('CVaR with respect to the goal for policy B = %f' % CVaR_B_goal)


#CVaR and VaR calculation for 10 year period - policy C
#sort the wealth array for policy C
wealth_time_limit_scenario_C_array_sorted = np.sort(wealth_time_limit_scenario_C_array,
                                                    kind='quicksort')
#print(wealth_time_limit_scenario_C_array_sorted)
#since h = 0.05, thus 501th element (index = 501) will be VaR of wealth
#print(wealth_time_limit_scenario_C_array_sorted[500])
VaR_C_goal = float(120000 - wealth_time_limit_scenario_C_array_sorted[500]) #VaR wrt goal
print('VaR with respect to the goal for policy C = %f' % VaR_C_goal)
#now, for CVaR, take the average of all the wealth upto index 499
sum = 0
for i in range(500):
    sum = float(sum + wealth_time_limit_scenario_C_array_sorted[i])
sum = float(sum / 500)
CVaR_C_goal = float(120000 - sum)
print('CVaR with respect to the goal for policy C = %f' % CVaR_C_goal)
















