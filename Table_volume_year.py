
#%%
import numpy as np
import pandas as pd
import polars as pl
import os


#%%
# Read parquet
df=pl.read_parquet("/Users/ednaloav/Dropbox/JMP/Data/NEW/year/open_volume_*.parquet")       
df=df.with_columns(pl.col('date').dt.strftime('%Y').alias('Year')) 

# %%
# Table of average volume
vol_cols=["customer_trade_volume",
        "customer_dollar_trade_volume",
        "customer_100_trade_volume",
        "customer_100_dollar_trade_volume",
        "customer_100_199_trade_volume",
        "customer_100_199_dollar_trade_volume",
        "customer_199_trade_volume",
        "customer_199_dollar_trade_volume",
        "professional_trade_volume",
        "professional_dollar_trade_volume",
        "firm_trade_volume",
        "firm_dollar_trade_volume",]

df = df.with_columns([
    (pl.col(c) * 100).alias(c) for c in vol_cols
])

#%%
daily_totals = (
    df
    .group_by(["date", "ticker", "f_k_moneyness", "cp_flag"])
    .agg([pl.col(c).sum() for c in vol_cols])
)

def summary_table(df_,variable,tipo):
    df_=df_.filter(pl.col(variable)>0)
    df_=df_.filter(pl.col('cp_flag')==tipo).group_by('f_k_moneyness').agg(
        pl.mean(variable).alias('Mean'),
        pl.col(variable).quantile(.05).alias('5th'),
        pl.col(variable).quantile(.25).alias('25th'),
        pl.col(variable).quantile(.5).alias('Median'),
        pl.col(variable).quantile(.75).alias('75th'),
        pl.col(variable).quantile(.95).alias('95th'),
        pl.sum(variable).alias('Total'),
        ).to_pandas().set_index('f_k_moneyness')
    total=df_['Total'].sum()
    df_['Percentage']=df_['Total']/total*100
    df_['Total Percentage']=df_['Percentage'].apply(lambda x: '(' + str("{:,.0f}".format(x)) +'\%)')
    df_=df_[['Mean','5th','25th','Median','75th','95th','Total Percentage']].T[['itm','otm','atm']]
    return df_

df_customer_100= summary_table(daily_totals,'customer_100_dollar_trade_volume','C')
df_customer_100_199= summary_table(daily_totals,'customer_100_199_dollar_trade_volume','C')
df_customer_199= summary_table(daily_totals,'customer_199_dollar_trade_volume','C')
df_prof= summary_table(daily_totals,'professional_dollar_trade_volume','C')
df_firm= summary_table(daily_totals,'firm_dollar_trade_volume','C')

df_all_call=pd.concat([df_customer_100,df_customer_100_199,df_customer_199,df_prof,df_firm],axis=1)




#%%

# Step 2: average across days by ticker
ticker_averages = (
    daily_totals
    .group_by(["ticker", "f_k_moneyness", "cp_flag"])
    .agg([pl.col(c + "_sum").mean().alias(c + "_avg") for c in vol_cols])
)

# Step 3: average across tickers by moneyness
table_avg_volume = (
    ticker_averages
    .group_by(["f_k_moneyness", "cp_flag"])
    .agg([pl.col(c + "_avg").mean().alias(c + "_avg") for c in vol_cols])
)

table_avg_volume.write_csv("/Users/ednaloav/Dropbox/JMP/Data/NEW/Tables/table_avg_volume.csv")

# %%

# 1) Totals by day, cp_flag
daily_total_pct = (
    df
    .group_by(["date","Year", "cp_flag"])
    .agg([pl.col(c).sum().alias(f"{c}_total") for c in vol_cols])
)

# 2) Totals by day, moneyness, cp_flag
daily_total_moneyness_pct = (
    df
    .group_by(["date","Year", "f_k_moneyness", "cp_flag"])
    .agg([pl.col(c).sum().alias(f"{c}_f_k_total") for c in vol_cols])
)

# 3) Join and compute per-metric moneyness percentages
table_avg_volume_pct = (
    daily_total_moneyness_pct
    .join(
        daily_total_pct,
        on=["date", "cp_flag"],
        how="left"
    )
    .with_columns([
        # For every metric c, compute c_f_k_total / c_total
        (pl.col(f"{c}_f_k_total") / pl.col(f"{c}_total")).alias(f"{c}_f_k_pct")
        for c in vol_cols
    ]))

def summary_table_year(df_: pl.DataFrame, variable: str, tipo: str) -> pl.DataFrame:
    """
    Summarizes the mean of a variable by Year and f_k_moneyness for a given cp_flag,
    and pivots the result so ITM/OTM/ATM appear as columns.
    """
    result = (
        df_
        .filter((pl.col(variable) > 0) & (pl.col("cp_flag") == tipo) & (pl.col("f_k_moneyness") != "nan"))
        .group_by(["Year", "f_k_moneyness"])
        .agg(pl.col(variable).mean().alias("Mean"))
        .pivot(
            values="Mean",
            index="Year",
            columns="f_k_moneyness"
        )
        .sort("Year")
    )
    return result[['itm','otm','atm']].to_pandas()


df_maturity=pl.read_parquet("/Users/ednaloav/Dropbox/JMP/Data/NEW/year/open_volume_*.parquet")       

df_maturity = (
    df_maturity
    .group_by(["date", "maturity", "f_k_moneyness", "cp_flag"])
    .agg([pl.col(c).sum() for c in vol_cols])
)

def summary_table_maturity(df_: pl.DataFrame, variable: str, tipo: str) -> pl.DataFrame:
    """
    Summarizes the mean of a variable by Year and f_k_moneyness for a given cp_flag,
    and pivots the result so ITM/OTM/ATM appear as columns.
    """
    result = (
        df_
        .filter((pl.col(variable) > 0) & (pl.col("cp_flag") == tipo) & (pl.col("f_k_moneyness") != "nan"))
        .group_by(["maturity", "f_k_moneyness"])
        .agg(pl.col(variable).mean().alias("Mean"))
        .pivot(
            values="Mean",
            index="maturity",
            columns="f_k_moneyness"
        )
    )
    maturity_order = ["0_days","1_7_days","7_30_days", "30_90_days","90_120_days","120_days",]
    result=result.to_pandas().set_index('maturity').reindex(maturity_order)
    result=result[['itm','otm','atm']]
    return result

df_customer_100_maturity= summary_table_maturity(df_maturity,'customer_100_trade_volume','C')
df_customer_100_199_maturity= summary_table_maturity(df_maturity,'customer_100_199_trade_volume','C')
df_customer_199_maturity= summary_table_maturity(df_maturity,'customer_199_trade_volume','C')
df_prof_maturity= summary_table_maturity(df_maturity,'professional_trade_volume','C')
df_firm_maturity= summary_table_maturity(df_maturity,'firm_trade_volume','C')

df_all_call_maturity=pd.concat([df_customer_100_maturity,df_customer_100_199_maturity,df_customer_199_maturity,df_prof_maturity,df_firm_maturity],axis=1)


df_customer_100_maturity= summary_table_maturity(df_maturity,'customer_100_dollar_trade_volume','C')
df_customer_100_199_maturity= summary_table_maturity(df_maturity,'customer_100_199_dollar_trade_volume','C')
df_customer_199_maturity= summary_table_maturity(df_maturity,'customer_199_dollar_trade_volume','C')
df_prof_maturity= summary_table_maturity(df_maturity,'professional_dollar_trade_volume','C')
df_firm_maturity= summary_table_maturity(df_maturity,'firm_dollar_trade_volume','C')

df_all_call_maturity_dollar=pd.concat([df_customer_100_maturity,df_customer_100_199_maturity,df_customer_199_maturity,df_prof_maturity,df_firm_maturity],axis=1)

table_avg_volume_pct_2.write_csv("/Users/ednaloav/Dropbox/JMP/Data/NEW/Tables/table_avg_volume_pct_2.csv")





# %%
