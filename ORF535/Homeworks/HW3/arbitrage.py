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

#some basic currency conversions among the 4 currencies and other assets converted into USD
eu_to_usd = 1.1844
usd_to_eu = 1 / eu_to_usd
#print('USD to EU = %f' % usd_to_eu)
usd_to_rmb = 6.58
rmb_to_usd = 1 / usd_to_rmb
#print('USD to RMB = %f' % usd_to_rmb)
gbp_to_usd = 1.3388
usd_to_gbp = 1 / gbp_to_usd
#print('USD to GBP = %f' % usd_to_gbp)
gold_to_usd = 1263
usd_to_gold = 1 / gold_to_usd
#print('USD to Gold = %f' % usd_to_gold)
usbill_to_usd = 1.0
usd_to_usbill = 1 / usbill_to_usd
#print('USD to US Bill = %f' % usd_to_usbill)
ukbill_to_gbp = 1.0
ukbill_to_usd = ukbill_to_gbp * gbp_to_usd
usd_to_ukbill = 1 / ukbill_to_usd
#print('USD to UK Bill = %f' % usd_to_ukbill)
germanbill_to_eu = 1.0
germanbill_to_usd = germanbill_to_eu * eu_to_usd
usd_to_germanbill = 1 / germanbill_to_usd
#print('USD to German Bill = %f' % usd_to_germanbill)
chinesebill_to_rmb = 1.0
chinesebill_to_usd = chinesebill_to_rmb * rmb_to_usd
usd_to_chinesebill = 1 / chinesebill_to_usd
#print('USD to Chinese Bill = %f' % usd_to_chinesebill)
sp500_to_usd = 267.17
usd_to_sp500 = 1 / sp500_to_usd
#print('USD to SP500 = %f' % usd_to_sp500)


#let me make a 2D array first
#index0 - USD, 1 - GBP, 2 - EU, 3 - CNY, 4 - gold, 5 - US Bill
#6 - German Bill, 7 - UK Bill (EU), 8 - Chinese Bill, 9 - SP500
rows, cols = 10, 10 #since there are 10 rows and 10 columns
Spots = [[0 for x in range(cols)] for y in range(rows)] #the matrix to store the spot price info
#since the (i, i) element will have conversion factor of 1 we can just hardcode it
for i in range(rows):
    Spots[i][i] = 1.0
#print(np.array(Spots))
#now, for the first row (USD) conversions, let us put in all the values pertaining to usd conversions
Spots[0][1] = usd_to_gbp #USD to GBP
Spots[0][2] = usd_to_eu #USD to EU
Spots[0][3] = usd_to_rmb #USD to CNY
Spots[0][4] = usd_to_gold #USD to gold
Spots[0][5] = usd_to_usbill #USD to USBill
Spots[0][6] = usd_to_germanbill #USD to GermanBill
Spots[0][7] = usd_to_ukbill #USD to UKBill
Spots[0][8] = usd_to_chinesebill #USD to GermanBill
Spots[0][9] = usd_to_sp500 #USD to GermanBill
#print(np.array(Spots))
#now, we will fill up all the other securities in terms of usd - all rows, col = 0 - reciprocal
#of usd to the security
for i in range(1, rows):
    Spots[i][0] = 1 / Spots[0][i] #converting in terms of usd to the security from above
#print(np.array(Spots))
#finally we can fill up the rest from a transitive relationship - assetA_to_assetB =
#assetA_to_usd * usd_to_assetB
for i in range(1, rows): #all rows beginning from the second row - GBP
    for j in range(1, cols): #all columns next to usd
        Spots[i][j] = Spots[i][0] * Spots[0][j] #the transitive relationship
print(np.array(Spots))

