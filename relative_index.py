import pandas as pd
import matplotlib.pyplot as plt 

tickers = ['AMC','GME','SPRT','KOSS','BB','SIGL']

AMC = pd.read_csv('data/nasdaq/csv/AMC.CSV')
GME = pd.read_csv('data/nasdaq/csv/GME.CSV')
SPRT = pd.read_csv('data/nasdaq/csv/SPRT.CSV')
KOSS = pd.read_csv('data/nasdaq/csv/KOSS.CSV')
SIGL = pd.read_csv('data/nasdaq/csv/SIGL_1.csv')
BB = pd.read_csv('data/nasdaq/csv/BB_1.csv')
AMC_Price = AMC[['Date', 'Close']].rename(columns={'Close':'AMC'})
GME_Price = GME[['Date','Close']].rename(columns={'Close':'GME'})
SPRT_Price = SPRT[['Date','Close']].rename(columns={'Close':'SPRT'})
KOSS_Price = KOSS[['Date','Close']].rename(columns={'Close':'KOSS'})
SIGL_Price = SIGL[['Date','Close']].rename(columns={'Close':'SIGL'})
BB_Price = BB[['Date','Close']].rename(columns={'Close':'BB'})
#print(SNDL_Price)
#data = pd.concat(['AMC_Price','GME_Price','BBBY_Price'], axis=1)
data = AMC_Price.merge(GME_Price, on='Date').merge(SPRT_Price, on='Date').merge(KOSS_Price, on='Date').merge(BB_Price, on='Date', how='left').merge(SIGL_Price, on='Date', how='left')
data.Date = pd.to_datetime(data.Date, format='%d-%m-%Y')
data.set_index('Date', inplace=True)
data = data['2021-01-01':]
data.bfill(inplace=True)
print(data.head(20))
normalized=data.div(data.iloc[0]).mul(100)
print(normalized.head(20))
normalized[tickers].plot()
plt.title('Normalised Growth Rates up to 08/11', fontsize=12)
plt.rc('ytick', labelsize=15) 
plt.legend(loc='best')
plt.savefig('destination_path.eps', format='eps')
plt.show()
