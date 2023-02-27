from time import sleep
from FinMind.data import DataLoader
import numpy as np
from datetime import datetime,timedelta
import pandas as pd
import requests
from scipy import stats
from bs4 import BeautifulSoup



def Dupont_analysis(candidate,quarter='',year=''):
    # connect to api
    api = DataLoader()
    
    quarters = {'Q1':'03-31','Q2':'06-30','Q3':'09-30','Q4':'12-31'}
    
    # embedded functions: profit margin, total asset turnover, financial leverage
    def profit_margin(IncomeAfterTaxes, Revenue):
        return IncomeAfterTaxes/Revenue
    
    def total_asset_turnover(Revenue,TotalAssets):
        return Revenue/TotalAssets
    
    def financial_leverage(TotalAssets,Equity):
        return TotalAssets/Equity

    # data access via API
    balance_sheet = api.taiwan_stock_balance_sheet(
        stock_id=candidate,
        start_date='2000-01-31',
    )

    financial_statements = api.taiwan_stock_financial_statement(
        stock_id=candidate,
        start_date='2000-01-31',
    )

    
    df = pd.concat([balance_sheet,financial_statements],axis=0)

    # date intersect: find the dates where all variables are available 
    IAF_date = set(df[df['type'].str.contains('IncomeAfterTax')]['date'].values)
    R_date = set(df[df['type'].str.contains('Revenue')]['date'].values)
    TA_date = set(df[df['type'].str.contains('TotalAssets')]['date'].values)
    date_range = sorted(IAF_date & R_date & TA_date)

    # assemble and calculate time-series Dupont analysis
    Dupont = pd.DataFrame({'quarter':[],'profit_margin':[],'total_asset_turnover':[],'financial_leverage':[],'ROE':[]})
    for d in date_range:
        IncomeAfterTaxes = df[(df['type']=='IncomeAfterTaxes') & (df['date'] == d)]['value'].values[0]
        Revenue = df[(df['type']=='Revenue') & (df['date'] == d)]['value'].values[0]
        TotalAssets = df[(df['type']=='TotalAssets') & (df['date'] == d)]['value'].values[0]
        Equity = df[(df['type']=='Equity') & (df['date'] == d)]['value'].values[0]
        pm = profit_margin(IncomeAfterTaxes,Revenue)
        tat = total_asset_turnover(Revenue,TotalAssets)
        fl = financial_leverage(TotalAssets,Equity)
        df_single = pd.DataFrame({'quarter':[d],'profit_margin':[pm],'total_asset_turnover':[tat],'financial_leverage':[fl],'ROE':[pm*tat*fl* 100]})
        Dupont = pd.concat([Dupont,df_single])
    Dupont.insert(loc=0, column='stock id', value=candidate)
    return Dupont




def Capm_model(candidate):


    etf0050price, div_dict = load_ERm()
    us_bond = load_Rf()
    
    
    stdict = {'Stock ID': [],
              'beta': [], 'alpha': [],
              'r_value': [],'p_value':[],
              'std_err':[],
              'expected_return_mean':[],
              'risk_free_asset_mean':[]
              }
    # CAPM model
    stockprice, div_dict_can = load_ERi(candidate)
    div_dict.update(div_dict_can)
    
    # Overlapped date
    day_span = sorted(set(stockprice['date'].values) & set(etf0050price['date'].values) & set(us_bond['date'].values))
    df_day_span = pd.DataFrame(day_span,columns=['specific date'])
    stockprice = pd.merge(stockprice,df_day_span,left_on='date',right_on='specific date')
    etf0050price = pd.merge(etf0050price,df_day_span,left_on='date',right_on='specific date')
    us_bond = pd.merge(us_bond,df_day_span,left_on='date',right_on='specific date')

    # Calculate the Daily Return
    candidate = stockprice.stock_id[0]
    stockprice[f'daily return {candidate}'] = stockprice['close'].pct_change(periods=1)
    etf0050price['daily return'] = etf0050price['close'].pct_change(periods=1)
    
    
    price0050 = etf0050price['close'].iloc[1:]
    pricestock = stockprice['close'].iloc[1:]
    ERi = stockprice[f'daily return {candidate}'].iloc[1:]
    ERm = etf0050price['daily return'].iloc[1:]
    Rf  = us_bond['daily return'][1:]/100

    # Calculate the beta and alpha for CAPM
    beta,alpha,r_value,p_value,std_err = stats.linregress((ERm-Rf),(ERi-Rf))
        
    expected_return_mean = np.mean(ERm-Rf)
    risk_free_asset_mean = np.mean(Rf)
    stdict['Stock ID'].append(candidate)
    stdict['beta'].append(beta)
    stdict['alpha'].append(alpha)
    stdict['r_value'].append(r_value)
    stdict['p_value'].append(p_value)
    stdict['std_err'].append(std_err)
    stdict['expected_return_mean'].append(expected_return_mean)
    stdict['risk_free_asset_mean'].append(risk_free_asset_mean)
        
    # COV martix
    # combine columns (daily return from different stocks) 
    df_all = pd.DataFrame()
    df_all['date'] = df_day_span.iloc[1:]
    df_all['eft0050'] = ERm
    df_all['us_bond'] = Rf
    df_all[f'price_etf0050'] = price0050
    df_all[f'price_{candidate}'] = pricestock
    df_all[f'daily return {candidate}'] = ERi

    capm_res = pd.DataFrame(stdict)
    div_df = pd.DataFrame(div_dict)
    return capm_res,df_all,div_df

# data access - Rf via GET request
def load_Rf():
    # define time-series length
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*7)
    
    # connect to api
    api = DataLoader()
    print('Rf loading...')
    url = "https://api.finmindtrade.com/api/v3/data"
    parameter = {
         "dataset": "GovernmentBondsYield",
         "data_id": "United States 10-Year",
         "date": start_date.strftime('%Y-%m-%d')
    }
    us_bond = requests.get(url, params=parameter)
    us_bond = us_bond.json()
    us_bond = pd.DataFrame(us_bond['data'])
    us_bond['daily return'] = us_bond['value']/250
    us_bond.date = us_bond.date.apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
    return us_bond

def load_ERm():
    # define time-series length
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*7)
    
    # connect to api
    api = DataLoader()
    # data access (ERm) via API
    print('ERm loading...')
    etf0050price = api.taiwan_stock_daily(
    stock_id='0050',
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d')
    )

    # web scraping for 0050's ex-dividend date
    URL = f"https://histock.tw/stock/0050/除權除息"
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find_all(class_ = "tb-stock text-center tbBasic")
    df = pd.read_html(str(table))[0]
    df = df[1:]

    info = zip(df['發放年度'].values.astype(int),df['除息日'].values)
    divid_df = pd.concat([df['發放年度'].apply(lambda x: str(int(x))),df['現金殖利率'].apply(lambda x: float(x[:-1]))],axis=1)
    divident = divid_df.groupby('發放年度').sum()
    div_dict = divident.to_dict()
    div_dict['ERm'] = div_dict.pop('現金殖利率')
    # removing stock exchange data at ex-dividend date
    for y,md in info:
        ex_divident_date = datetime(y,int(md.split('/')[0]),int(md.split('/')[1]))
        etf0050price = etf0050price[etf0050price['date'] !=  ex_divident_date]
    etf0050price.date = etf0050price.date.apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
    return etf0050price, div_dict

def load_ERi(candidate):
    # define time-series length
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*7)
    
    # connect to api
    api = DataLoader()
    
    # data access (ERm) via API
    print(f'ERi - {candidate} loading...')
    stockprice = api.taiwan_stock_daily(
    stock_id=candidate,
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d')
    )
    # web scraping for candidate's ex-dividend date
    URL = f"https://histock.tw/stock/{candidate}/除權除息"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find_all(class_ = "tb-stock text-center tbBasic")
    df = pd.read_html(str(table))[0]
    df = df[1:]
    # removing stock exchange data at ex-dividend date
    info = zip(df['發放年度'].values.astype(int),df['除息日'].values)
    divid_df = pd.concat([df['發放年度'].apply(lambda x: str(int(x))),df['現金殖利率'].apply(lambda x: float(x[:-1]))],axis=1)
    divident = divid_df.groupby('發放年度').sum()
    div_dict_can = divident.to_dict()
    div_dict_can[candidate] =  div_dict_can.pop('現金殖利率')
    for y,md in info:
        ex_divident_date = datetime(y,int(md.split('/')[0]),int(md.split('/')[1]))
        stockprice = stockprice[stockprice['date'] !=  ex_divident_date]
    stockprice.date = stockprice.date.apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
    return stockprice, div_dict_can

#def return_aligning(stockprice,etf0050price,Rf):
#    # Overlapped date
#    day_span = sorted(set(stockprice['date'].values) & set(etf0050price['date'].values) & set(us_bond['date'].values))
#    df_day_span = pd.DataFrame(day_span,columns=['specific date'])
#    stockprice = pd.merge(stockprice,df_day_span,left_on='date',right_on='specific date')
#    etf0050price = pd.merge(etf0050price,df_day_span,left_on='date',right_on='specific date')
#    Rf = pd.merge(Rf,df_day_span,left_on='date',right_on='specific date')
#
#    # Calculate the Daily Return
#    candidate = stockprice.stock_id[0]
#    stockprice[f'daily return {candidate}'] = stockprice['close'].pct_change(periods=1)
#    etf0050price['daily return'] = etf0050price['close'].pct_change(periods=1)
#    ERi = stockprice[f'daily return {candidate}'].iloc[1:]
#    ERm = etf0050price['daily return'].iloc[1:]
#    Rf  = Rf.iloc[1:,:-1]
#    return ERi,ERm,Rf


#def PER_method(candidate):
#    end_date = datetime.today()
#    start_date = end_date - timedelta(days=365*7)
#    api = DataLoader()
#
#    stockper = api.taiwan_stock_per_pbr(
#        stock_id=candidate,
#        start_date=start_date.strftime('%Y-%m-%d'),
#        end_date=end_date.strftime('%Y-%m-%d')
#    )
#
#    ID = api.taiwan_stock_info()[['stock_id','stock_name']]
#
#    stockprice = api.taiwan_stock_daily(
#        stock_id=candidate,
#        start_date=start_date.strftime('%Y-%m-%d'),
#        end_date=end_date.strftime('%Y-%m-%d')
#    )
#
#    price_now = stockprice['close'].values[-1]
#    df = pd.merge(stockper, stockprice, on=["date",'stock_id'])
#    df = pd.merge(df,ID, on='stock_id')
#
#    zh_name = df['stock_name'][0]
#    df_select = df[['date','stock_id','PBR','close']]
#    df_select.loc[:,'EPS'] = df_select.loc[:,'close'] / df_select.loc[:,'PBR']
#
#    df_select['date']  = df_select['date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d').strftime('%Y'))
#    PBR_max = df_select.groupby(['date']).max()['PBR'].values
#    PBR_min = df_select.groupby(['date']).min()['PBR'].values
#    EPS_min = df_select.groupby(['date']).mean()['EPS'].values
#
#    tar_eps = np.mean(EPS_min[-12:])
#    sell = np.mean(PBR_max)
#    buy = np.mean(PBR_min)
#    neural = (sell+buy)/2
#
#    stdict = {'Stock ID': candidate, 'Stock Name': zh_name, 
#              'Cheap price': tar_eps*buy, 'Middle price': tar_eps*neural,
#              'Expensive price':tar_eps*sell,'Today price':price_now}
#    return stdict