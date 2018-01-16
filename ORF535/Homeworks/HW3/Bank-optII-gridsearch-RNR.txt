import numpy as np
import matplotlib.pyplot as plt
import pandas
import cvxpy as cvx
from xlrd import open_workbook
import functools
import math
import cvxopt as opt
from cvxopt import blas, solvers
import numpy
from scipy.stats import norm
import scipy as scipy
import cvxopt as cvxopt
import matplotlib.pyplot as plt
import copy
from scipy.optimize import *
import itertools

def is_sum_one(tup):
    sum = 0
    for i in range(len(tup)):
        sum += tup[i]
    if sum == 1:
        return True
    else:
        return False

#function to calculate monthly geometric mean of a series
def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))


#load in the data
df_AGG = pandas.read_excel('Project_Data_Part1.xlsx', sheetname='AGG') #AGG adj close data
df_SPY = pandas.read_excel('Project_Data_Part1.xlsx', sheetname='SPY') #APY adj close data
df_TBill = pandas.read_excel('Project_Data_Part1.xlsx', sheetname='13W Treasury') #13WTreasury r (%)
df_RNR = pandas.read_excel('RNR-2.xlsx') #RNR data
df_RNR = df_RNR['Adj Close'].values[102:270] #since the last row is a day of the month other than 1st

df_AGG = df_AGG['Adjusted Close'].values
df_SPY = df_SPY['Adjusted Close'].values
df_TBill = df_TBill['Rate (annualized %)'].values



#now for the adj. close and r data, get the monthly returns (R) - ((r_t+1 - r_t) / 100) + 1
AGG_R = list() #return (R) of Feb 2004 to December 2017 - AGG
SPY_R = list() #return (R) of Feb 2004 to December 2017 - SPY
TBill_R = list() #return (R) of Feb 2004 to December 2017 - TBill
RNR_R = list() #return (R) RNR
for i in range(len(df_AGG) - 1):
    R_AGG = float((df_AGG[i + 1] - df_AGG[i]) / 100) + 1
    AGG_R.append(R_AGG)
    R_SPY = float((df_SPY[i + 1] - df_SPY[i]) / 100) + 1
    SPY_R.append(R_SPY)
    R_TBill = float((df_TBill[i + 1] + 1) ** (1 / float(12))) #converting from annualized to monthly
    TBill_R.append(R_TBill)
    R_RNR = float((df_RNR[i + 1] - df_RNR[i]) / 100) + 1
    RNR_R.append(R_RNR)

#R's for TBills which would essentially be the returns at the beginning of each period
R_TBill_beg = list()
for i in range(len(df_TBill)):
    TBill_beg = float((df_TBill[i] + 1) ** (1 / float(12))) #converting from annualized to monthly
    R_TBill_beg.append(TBill_beg)

ys = np.linspace(0.0, 5.0, num=100, endpoint=True) #grid search spacings - for the sake of practicality,
#print(ys)

#we are considering 0 to 5 with both endpoints included as the range of leverage
ws = np.linspace(0.1, 1.0, num=10, endpoint = True) #grid search spacings for weights
#print(ws)
w1 = list()
w2 = list()
w3 = list()
for i in range(ws.size):
    w1.append(float(round(ws[i], 2)))
    w2.append(float(round(ws[i], 2)))
    w3.append(float(round(ws[i], 2)))
'''w1.append(1)
w2.append(2)
w3.append(3)'''

w = list()
w.append(w1)
w.append(w2)
w.append(w3)
# print(w)
d = list(itertools.product(*w))

sum_to_one = list()
for i in range(len(d)):
    if (is_sum_one(d[i]) is True):
        sum_to_one.append(d[i])
    else:
        continue

ws2 = list()
ws2.append(np.array(sum_to_one))
#print(ws2)
w_s = np.array(np.array(ws2))
#print(w_s)
#print(w_s.shape[1])
#print(w_s[0][4798])
#now, just join ys and weights that add to 1
x = list()
for i in range(len(ys)):
    if (ys[i] == 0): #don't take y = 0
        continue
    for j in range(w_s.shape[1]): #for each of the 4801 weights that add to 1
        t = list()
        t.append(ys[i])
        for k in range(3): #since 3 assets
            t.append(w_s[0][j][k])
        #print(t)
        x.append(t)
'''print(len(x))
print(x[42998])'''
#print(x)

#now, the list 'x' stores all possible combinations of decision variable for leverage 'y'
#and weights of each individual asset that add up to 1
#CVaR constraint
def CVaR_constraint(capital_per_month1):
    h = 0.01  # tolerance level is 1%
    #print(len(capital_per_month1))
    VaR_point = int(h * len(capital_per_month1))
    capital_per_month_sorted = np.sort(capital_per_month1)
    # CVaR at h = 0.01 (1%)
    CVaR = 0
    for i in range(VaR_point):
        CVaR = float(CVaR + capital_per_month_sorted[i])
    # print(VaR_point)
    CVaR = float(CVaR / VaR_point)
    return CVaR
#NOTE: asset1 - SPY, asset2 - AGG, asset3 - TBill
#we will iterate through the entire list and check for constraints and optimize for y, w1, w2, w3
y = 0.0 #initialization of the optimization variable
max_geometric_R = 0.0 #this will be the optimizer
min_CVaR = 0.0 #best CVaR so far
w1_best = 0.0
w2_best = 0.0
w3_best = 0.0
'''print(len(x))
print(x)
print(x[0])
print(x[1])
print(x[1][0])
print(x[1][1])
print(x[1][2])
print(x[1][3])'''
#mean = []
for j in range(len(x)):
    #print('Iteration number = %d' % j)
    y_i = x[j][0] #the y value
    w1_i = x[j][1] #the w1 value
    w2_i = x[j][2]  # the w2 value
    w3_i = x[j][3]  # the w3 value
    '''print('y variable = %f' % y_i)
    print('w(SPY) = %f' % w1_i)
    print('w(AGG) = %f' % w2_i)
    print('w(TBill) = %f' % w3_i)'''
    capital_per_month = list()  # the amount of capital available at the beginning of each month
    asset_investments = list()  # beginning of each month how much money is invested - long
    asset_investments_rnr = list()  # beginning of each month how much money is invested - long
    liabilities_shorted = list()  # beginning of each month how much money is shorted
    net_profit_list = list()
    capital_returns_per_month = list()  # Total Returns (R) per month -
    capital = 1.3e9  # initial capital (in Billions of $) - this is what we have in January 2004 * returns due to

    spy_investments = list()
    agg_investments = list()
    tbill_investments = list()

    red_break = 0  # check if you came out due to a red flag

    for i in range(0, len(R_TBill_beg)):
        # for first time period
        if i == 0:
            asset = y_i * capital
            asset_investments.append(asset)
            asset_investments_rnr.append(asset)
            liability = 2 * y_i * capital
            liabilities_shorted.append(liability)
            capital = capital * R_TBill_beg[0]  # for first year just focus on TBills
            capital_per_month.append(capital)
            spy = capital * w1_i
            spy_investments.append(spy)
            agg = capital * w2_i
            agg_investments.append(agg)
            tbill = capital * w3_i
            tbill_investments.append(tbill)
            if capital <= 0:  # red card
                red_break = 1
                break
        else:
            # take in the assets - liabilities from the pervious month which would add to capital flows this month
            '''print("len = %d" % len(asset_investments_tlt))
            print("i = %d" %i)'''
            assets_returns_prev_year = asset_investments_rnr[i - 1] * RNR_R[i - 1] \
                                       + asset_investments[i - 1] * AGG_R[i - 1]
            liabilities_prev_year = liabilities_shorted[i - 1] * TBill_R[i - 1]
            lev_profit = assets_returns_prev_year - liabilities_prev_year
            # returns from each asset this month
            spy_returns_prev_year = spy_investments[i - 1] * SPY_R[i - 1]
            agg_returns_prev_year = agg_investments[i - 1] * AGG_R[i - 1]
            tbill_returns_prev_year = tbill_investments[i - 1] * TBill_R[i - 1]
            net_profit = lev_profit + spy_returns_prev_year \
                         + agg_returns_prev_year + tbill_returns_prev_year
            total_profit = net_profit - capital
            net_profit_list.append(total_profit)
            prev_month_capital = capital

            # net capital from returns
            capital = net_profit
            R_this_month = float(capital / prev_month_capital)  # formula for R
            if capital <= 0:  # red card
                #print("Red Card. i = %d" % i)
                red_break = 1
                break

            # asset and liability allocation for the next month
            asset = y_i * capital
            asset_investments.append(asset)
            asset_investments_rnr.append(asset)
            liability = 2 * y_i * capital
            liabilities_shorted.append(liability)

            # for each of the invested assets - fixed mix - next month investments
            spy = capital * w1_i
            spy_investments.append(spy)
            agg = capital * w2_i
            agg_investments.append(agg)
            tbill = capital * w3_i
            tbill_investments.append(tbill)
            capital_per_month.append(capital)  # capital for this month
            capital_returns_per_month.append(R_this_month)

    # now calculate the geometric mean of the capital returns and minimize the -ve of it
    if red_break == 1: #that means a red flag was the reason, then just go to the next j iteration
        continue

    annual_geometric_return_R = math.pow(geo_mean(capital_returns_per_month), 12)
    #mean.append(annual_geometric_return_R)
    CVaR = CVaR_constraint(capital_per_month)
    '''print('outside if...')
    print('Annual geomean = %f' % annual_geometric_return_R)
    print('CVaR = %f' % CVaR)
    print('y = %f' % y_i)
    print('W1 = %f' % w1_i)
    print('W2 = %f' % w2_i)
    print('W3 = %f' % w3_i)
    print('Max geo mean so far: %f' % max_geometric_R)'''
    # update for decision variable whenever we have a higher return and CVaR constraint is satisfied
    if (annual_geometric_return_R >= max_geometric_R):
        y = y_i
        max_geometric_R = annual_geometric_return_R
        min_CVaR = CVaR
        w1_best = w1_i
        w2_best = w2_i
        w3_best = w3_i
        '''print('inside if - changing the max geomean..')
        print('y = %f' % y)
        print('w1 best = %f' % w1_best)
        print('w2 best = %f' % w2_best)
        print('w3 best = %f' % w3_best)
        print('Max geo mean (changed): %f' % max_geometric_R)
        print('CVaR (changed): %f' % min_CVaR)'''


'''print('Printing results...')
print(y)
print(w1_best)
print(w2_best)
print(w3_best)
print(max_geometric_R)
print(min_CVaR)'''
'''print(mean)
print(np.sort(np.array(mean)))'''
#now, our y will store the optimized decision variable, and w1, w2, w3 store the optimized weights.
# Now we will just conduct the analysis of risk-reward as before
num_yellow = 0
num_red = 0
inverted_yield = 0
capital = 1.3e9
yellow_mark = capital * 0.2

#At every month, y*capital - long on AGG, y*capital - short on 13W TBills, the remaining capital
# (initial capital) - invest in SPY, AGG, T-Bills
#at each month beginning February, we will add the results of long and short and see how the previous
#T-Bill investment paid off - returns of assets - losses due to liabilities
capital_per_month = list() #the amount of capital available at the beginning of each month
asset_investments = list()  # beginning of each month how much money is invested - long
asset_investments_rnr = list() #beginning of each month how much money is invested - long
liabilities_shorted = list() #beginning of each month how much money is shorted
net_profit_list = list()
capital_returns_per_month = list() #Total Returns (R) per month -

spy_investments = list()
agg_investments = list()
tbill_investments = list()
#changing the names of weight variables for clarity
w_SPY = w1_best
w_AGG = w2_best
w_TBill = w3_best

for i in range(len(R_TBill_beg)):
    if i == 0:
        # for first time period
        asset = y * capital
        asset_investments.append(asset)
        asset_investments_rnr.append(asset)
        liability = 2 * y * capital
        liabilities_shorted.append(liability)
        capital = capital * R_TBill_beg[0]  # for first year just focus on TBills
        capital_per_month.append(capital)
        spy = capital * w_SPY
        spy_investments.append(spy)
        agg = capital * w_AGG
        agg_investments.append(agg)
        tbill = capital * w_TBill
        tbill_investments.append(tbill)
    else:
        #take in the assets - liabilities from the pervious month which would add to capital flows this month
        assets_returns_prev_year = asset_investments_rnr[i - 1] * RNR_R[i - 1] \
                                       + asset_investments[i - 1] * AGG_R[i - 1]
        liabilities_prev_year = liabilities_shorted[i - 1] * TBill_R[i - 1]
        lev_profit = assets_returns_prev_year - liabilities_prev_year
        #returns from each asset this month
        spy_returns_prev_year = spy_investments[i - 1] * SPY_R[i - 1]
        agg_returns_prev_year = agg_investments[i - 1] * AGG_R[i - 1]
        tbill_returns_prev_year = tbill_investments[i - 1] * TBill_R[i - 1]
        net_profit = lev_profit + spy_returns_prev_year \
                     + agg_returns_prev_year + tbill_returns_prev_year
        total_profit = net_profit - capital
        net_profit_list.append(total_profit)
        prev_month_capital = capital

        #net capital from returns
        capital = net_profit
        R_this_month = float(capital / prev_month_capital) #formula for R

        spread = spread = (AGG_R[i - 1] + RNR_R[i - 1]) - TBill_R[i - 1] #the spread
        if spread < 0:
            inverted_yield = int(inverted_yield + 1)

        #asset and liability allocation for the next month
        asset = y * capital
        asset_investments.append(asset)
        asset_investments_rnr.append(asset)
        liability = 2 * y * capital
        liabilities_shorted.append(liability)

        #for each of the invested assets - fixed mix - next month investments
        spy = capital * w_SPY
        spy_investments.append(spy)
        agg = capital * w_AGG
        agg_investments.append(agg)
        tbill = capital * w_TBill
        tbill_investments.append(tbill)
        capital_per_month.append(capital) #capital for this month
        capital_returns_per_month.append(R_this_month)
    if capital < 0:  # we will have to recapitalize the bank
        # print('Red Flag.')
        num_red = int(num_red + 1)
        diff = float((-1 * capital + 1.3e9))

    elif capital < yellow_mark:
        # print('Yellow Flag.')
        num_yellow = int(num_yellow + 1)

#print(capital_per_month)

print('Value of optimized decision variable (y) = %f' % y)
print('Value of optimized weight variable for SP500 = %f' % w1_best)
print('Value of optimized weight variable for AGG = %f' % w2_best)
print('Value of optimized weight variable for TBill = %f' % w3_best)
print('Number of red cards = %d ' % num_red)
print('Number of yellow cards = %d ' % num_yellow)
print('Number of inverted yields = %d ' % inverted_yield)


#Calculate reward and risk measures
#find the geometric returns, monthly then convert to annualized
annual_geometric_return_R = math.pow(geo_mean(capital_returns_per_month), 12)
print('Annual geometric return (R) = %f ' % annual_geometric_return_R)
#annualized volatility of returns
annual_volatility = 12 * math.pow(np.std(np.array(capital_returns_per_month)), 2)
print('Annualized volatility = %f' % annual_volatility)
#Sharpe Ratio
'''all_TBills_R = list()
val = float((df_TBill[0] ** (1 / float(12))) + 1) #converting from annualized to monthly - for 1st month
all_TBills_R.append(val)
for i in range(len(TBill_R)):
    all_TBills_R.append(TBill_R[i])'''
all_TBills_R = list()
for i in range(len(df_TBill)):
    val = float((df_TBill[i] + 1) ** (1 / float(12)))  # converting from annualized to monthly - for each month
    all_TBills_R.append(val)
#print(all_TBills_R)
risk_free_R = math.pow(geo_mean(all_TBills_R), 12) #the risk free rate is the annualized avg of T-Bills
print('Risk Free rate of return = %f' % risk_free_R)
Sharpe_Ratio = float((annual_geometric_return_R - risk_free_R) / math.sqrt(annual_volatility))
print('Sharpe Ratio = %f' % Sharpe_Ratio)
#Sortino Ratio
#first we have to calculate the downside variance wrt risk_free_R
N = len(capital_returns_per_month)

risk_free_R_monthly = geo_mean(all_TBills_R) #monthly avg risk free return
#print(risk_free_R_monthly)
downsides = list()
for i in range(N):
    comp = risk_free_R_monthly - capital_returns_per_month[i] #using monthly avg risk free return
    if comp > 0:
        downsides.append(comp)
    else:
        downsides.append(0)
downside_variance = 12 * math.pow(np.std(np.array(downsides)), 2) #annualized from mothly
print('Annualized downside variance = %f' % downside_variance)
Sortino_Ratio = np.inf if downside_variance == 0.0 else float((annual_geometric_return_R - risk_free_R) / math.sqrt(downside_variance))
print('Sortino Ratio = %f' % Sortino_Ratio)

#Maximum drawdown calculation
X = pandas.Series(capital_per_month) #pandas series for max drawdown calculation
#print(X)
def create_drawdowns(equity_curve):
    """
    Calculate the largest peak-to-trough drawdown of the equity curve
    as well as the duration of the drawdown. Requires that the
    equity returns is a pandas Series.

    Parameters:
    equity_curve - A pandas Series representing period net returns.

    Returns:
    drawdown, duration - Highest peak-to-trough drawdown and duration.
    """

    # Calculate the cumulative returns curve
    # and set up the High Water Mark
    # Then create the drawdown and duration series
    hwm = [0]
    eq_idx = equity_curve.index
    #print(eq_idx)
    drawdown = pandas.Series(index = eq_idx)
    duration = pandas.Series(index = eq_idx)

    # Loop over the index range
    for t in range(1, len(eq_idx)):
        cur_hwm = max(hwm[t-1], equity_curve[t])
        hwm.append(cur_hwm)
        drawdown[t]= hwm[t] - equity_curve[t]
        duration[t]= 0 if drawdown[t] == 0 else duration[t-1] + 1
    '''print(drawdown)
    print(duration)'''
    return drawdown.max(), duration.max()
ans = create_drawdowns(X)
#print(ans)
MDD = ans[0]
print('Maximum DrawDown = %f' % MDD)

#VaR at h = 0.01 (1%)
h = 0.01
VaR_point = int(h * len(capital_per_month))
'''print(capital_per_month)
print(VaR_point)'''
capital_per_month_sorted = np.sort(capital_per_month)
VaR = capital_per_month_sorted[VaR_point]
print('VaR = %f' % VaR)
#CVaR at h = 0.01 (1%)
CVaR = 0
for i in range(VaR_point):
    CVaR = float(CVaR + capital_per_month_sorted[i])
CVaR = float(CVaR / VaR_point)
print('CVaR = %f' % CVaR)