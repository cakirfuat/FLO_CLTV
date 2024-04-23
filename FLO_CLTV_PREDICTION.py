import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

#### Veri Setinin Okunması
df_ = pd.read_csv("/Users/fuatcakir/PycharmProjects/pythonProject2/.venv/flo_data_20k.csv")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()


## Aykırı değer ayıklama
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df['customer_value_total'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']
df['order_num_total'] = df['order_num_total_ever_offline'] + df['order_num_total_ever_online']
df.describe().T
replace_with_thresholds(df, "order_num_total")
replace_with_thresholds(df, "customer_value_total")
replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")
replace_with_thresholds(df, "customer_value_total_ever_offline")
df.describe().T


### Metriklerin hesaplanması
df['last_order_date'].sort_values(ascending=False).head()

today_date = dt.datetime(2021, 6, 2)
type(today_date)

df['last_order_date'] = pd.to_datetime(df['last_order_date'])
df['first_order_date'] = pd.to_datetime(df['first_order_date'])
df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])
df['last_order_date_offline'] = pd.to_datetime(df['last_order_date_offline'])
df.dtypes


df['recency'] = (df['last_order_date'] - df['first_order_date'])
df['T'] = ((today_date - df['first_order_date']))
df['frequency'] = df['order_num_total']
df['monetary'] =  df['customer_value_total']

cltv_df = df[['master_id','recency','T','frequency','monetary']].copy()
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]


cltv_df.describe().T
cltv_df["recency"] = (cltv_df["recency"] / pd.Timedelta(days=1)) / 7
cltv_df["T"] = (cltv_df["T"] / pd.Timedelta(days=1)) / 7


##############################################################
# 2. BG-NBD Modelinin Kurulması
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
                cltv_df['recency'],
                cltv_df['T'])

################################################################
# 1 hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)

bgf.predict(1,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

cltv_df.sort_values(by=['expected_purc_1_week'],ascending=False)

df.drop('cltv',axis=1,inplace=True)


################################################################
# 1 ay içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()
################################################################
# 3 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?
################################################################

bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

cltv_df["expected_purc_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
##############################################################
# 3. GAMMA-GAMMA Modelinin Kurulması
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
cltv_df.sort_values("expected_average_profit", ascending=False).head(10)
##############################################################
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################

cltv_df['clv'] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

##############################################################
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
##############################################################
cltv_final = cltv_df.copy()

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(50)

cltv_final.groupby("segment")[["clv","recency",'frequency','monetary']].agg({"count", "mean", "sum"})
