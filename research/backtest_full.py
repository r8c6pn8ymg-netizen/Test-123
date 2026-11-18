import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
# ---------- sentiment (weekly, aligned to week-start Monday like TradingView weekly bars) ----------
a=pd.read_csv('aaii.csv',parse_dates=['date']); a['date']=a.date.astype('datetime64[ns]')
a=a.set_index('date').spread*100
cnn=pd.read_csv('cnn.csv',index_col=0,parse_dates=True)['fear_and_greed_historical']; cnn.index=cnn.index.astype('datetime64[ns]')
def mon(f,col):
    s=pd.read_csv(f,index_col=0,parse_dates=True)[col]; s.index=(s.index+pd.Timedelta(days=19)).astype('datetime64[ns]'); return s
bs=mon('bofa_sent.csv','bofa_sent'); bc=mon('bofa_cash.csv','bofa_cash')
def wk(s):  # value known by end of week (Mon-start week)
    s=s.copy(); s.index=s.index.to_period('W-SUN').start_time; return s.groupby(level=0).last()
SENT=pd.DataFrame({'aaii':wk(a),'cnn':wk(cnn),'bofa':wk(bs),'cash':wk(bc)})
SENT=SENT.sort_index().ffill()
SENT.loc[SENT.index<'2020-07-13','cnn']=np.nan
SENT['aaii4']=SENT.aaii.rolling(4).mean()
def rsi(s,n):
    d=s.diff(); u=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean(); return 100-100/(1+u/dn)
def osc(df):
    c,h,l=df.close,df.high,df.low
    st=(c-l.rolling(13).min())/(h.rolling(13).max()-l.rolling(13).min())*100
    z=((c-c.rolling(20).mean())/c.rolling(20).std()*20+50).clip(0,100)
    return (rsi(c,9)+st+z)/3
def load(name):
    if name=='SPX87':
        x=pd.read_excel('dl/aaii.xls',sheet_name='SENTIMENT',header=None,skiprows=5)[[0,10,11,12]]
        x.columns=['date','high','low','close']; x['date']=pd.to_datetime(x.date,errors='coerce'); x=x.dropna()
        x=x.set_index('date').astype(float); x.index=x.index.astype('datetime64[ns]').to_period('W-SUN').start_time
        return x[~x.index.duplicated(keep='last')]
    d=pd.read_csv(f'px/{name}.csv'); d.index=pd.to_datetime(d.time,unit='s').dt.tz_localize(None).dt.to_period('W-SUN').dt.start_time
    return d[['high','low','close']][~d.index.duplicated(keep='last')]
def build(name):
    d=load(name); d['osc']=osc(d)
    d=d.join(SENT,how='left'); d[SENT.columns]=d[SENT.columns].ffill()
    c=d.close
    for H in (4,8,13,26): d[f'f{H}']=(c.shift(-H)/c-1)*100
    # scores 0..100 (100 = greed)
    d['S_osc']=d.osc
    d['S_aaii']=((d.aaii4.clip(-30,30)+30)/60*100)
    d['S_cnn']=d.cnn
    d['S_bofa']=d.bofa*10
    d['S_cash']=((6.5-d.cash).clip(0,3)/3*100)
    return d
ASSETS=['SPX87','SPX','NDX','DOW','GOLD','SILVER','DXY']
D={k:build(k) for k in ASSETS}
pd.to_pickle(D,'D.pkl')
