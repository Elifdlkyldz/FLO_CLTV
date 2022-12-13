#CLTV VALUE PREDİCTİON WİTH BG/NBD AND GAMMA GAMMA

from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import datetime as dt
import pandas as pd
pd.set_option('display.max_columns',None) #-değişkenler
pd.set_option('display.max_rows', None) #-satırlar
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df_ = pd.read_csv(r'C:\Users\elifd\PycharmProjects\pythonProject1\flo_data_20k.csv')

df= df_.copy()


# Missing Value Analysis;

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range.round()
    low_limit = quartile1 - 1.5 * interquantile_range.round()
    return low_limit, up_limit


# Outlier Analysis;

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(df, 'order_num_total_ever_online')

replace_with_thresholds(df, 'order_num_total_ever_offline')

replace_with_thresholds(df,'customer_value_total_ever_offline')

replace_with_thresholds(df,'customer_value_total_ever_online')

df.describe().T
df.head()
df.isnull().sum()

# Omnichannel refers to the total purchase made over both online and offline platforms.
df['Omnichannel'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']

df.head()

df.info()

df['total_order'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']

df['total_value'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']
df.head(3)

# Converting the above mentioned column types from object to datetime format

df.columns
date_columns = df.columns [df.columns.str.contains('date')]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df.shape

####   Creating the CLTV Data Structure 

df["last_order_date"].max()
df['last_order_date_online'].max()
df['last_order_date_offline'].max()

today_date = dt.datetime(2021, 6, 1)
#type(today_date)


cltv_df = pd.DataFrame()
type(cltv_df)

cltv_df['customer_id'] = df['master_id']

cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).astype('timedelta64[D]'))/7
cltv_df["frequency"] = df["total_order"]
cltv_df["monetary_cltv_avg"] = df["total_value"] / df["total_order"]

cltv_df.head()

cltv_df.columns = ['customer_id', 'recency', 'T', 'frequency', 'monetary']

#### Establishment of BG/NBD, Gamma-Gamma Models and calculation of CLTV 




bgf = BetaGeoFitter(penalizer_coef= 0.001) #-katsayılara uygulanacak olan ceza katsayısıdır.

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])



cltv_df['exp_sales_3_month']= bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                       cltv_df['T']).sort_values(ascending=False)


cltv_df['exp_sales_6_month']= bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False)





### # Calculation of CLTV with BG-NBD and GG model 

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)
cltv_df['exp_average_value'] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary'])



cltv_df['cltv'] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.HAFTALIK
                                   discount_rate=0.01) #indirim katsayısı

cltv_df.head()

cltv_df.sort_values('cltv', ascending=False).head(20)

###Creating Segments Based on CLTV Values


cltv_df['segment'] = pd.qcut(cltv_df['cltv'], 4, labels=['D', 'C', 'B', 'A'])

cltv_df.head()

# RESULT 

cltv_df.groupby("SEGMENT").agg({"count","mean","sum"})
