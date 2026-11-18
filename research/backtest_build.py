import pandas as pd, numpy as np
x=pd.read_excel('dl/aaii.xls',sheet_name='SENTIMENT',header=None,skiprows=5)
x=x[[0,1,3,10,11,12]]; x.columns=['date','bull','bear','hi','lo','close']
x['date']=pd.to_datetime(x.date,errors='coerce'); x=x.dropna(subset=['date','close']).set_index('date').astype(float)
x=x[~x.index.duplicated()].sort_index()
x['aaii']=(x.bull-x.bear)*100
c,h,l=x.close,x.hi,x.lo
def rsi(s,n):
    d=s.diff(); u=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean(); return 100-100/(1+u/dn)
st=(c-l.rolling(13).min())/(h.rolling(13).max()-l.rolling(13).min())*100
z=((c-c.rolling(20).mean())/c.rolling(20).std()*20+50).clip(0,100)
x['osc']=(rsi(c,9)+st+z)/3
# BofA monthly, available from ~20th of the month
def monthly(f,col):
    s=pd.read_csv(f,index_col=0,parse_dates=True)[col]; s.index=s.index+pd.Timedelta(days=19); return s
for f,col in (('bofa_sent.csv','bofa_sent'),('bofa_cash.csv','bofa_cash')):
    s=monthly(f,col); x[col]=s.reindex(x.index.union(s.index)).ffill().reindex(x.index)
cnn=pd.read_csv('cnn.csv',index_col=0,parse_dates=True)['fear_and_greed_historical']
x['cnn']=cnn.reindex(x.index.union(cnn.index)).ffill().reindex(x.index); x.loc[x.index<'2020-07-20','cnn']=np.nan
for H in (4,8,13,26): x[f'f{H}']=(c.shift(-H)/c-1)*100
x['aaii4']=x.aaii.rolling(4).mean()
x['aaii_z']=(x.aaii-x.aaii.rolling(52).mean())/x.aaii.rolling(52).std()
x.to_pickle('bt.pkl')
