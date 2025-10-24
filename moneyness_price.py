
#%%
import numpy as np
import pandas as pd
import polars as pl
import os

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"   
pio.templates.default = "simple_white"
from statsmodels.iolib.summary2 import summary_col

import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

os.chdir("/Users/ednaloav/Dropbox/JMP/JMP_Github/")
from functions_jmp import *
from table_format import *

#%%
# Read parquet
df=pl.read_parquet("/Users/ednaloav/Dropbox/JMP/Data/NEW/year/open_volume_*.parquet")       

df=df.with_columns([
pl.when((pl.col("delta") >45) & (pl.col("delta") < 55))
.then(pl.lit('atm'))
.when((pl.col("delta") >= 55) & (pl.col("cp_flag")=='C'))
.then(pl.lit("itm"))
.when((pl.col("delta") <= 45) & (pl.col("cp_flag")=='C'))
.then(pl.lit("otm"))
.when((pl.col("delta") >-55) & (pl.col("delta") < -45))
.then(pl.lit("atm"))
.when((pl.col("delta") <= -55) & (pl.col("cp_flag")=='P'))
.then(pl.lit("itm"))
.when((pl.col("delta") >= -45) & (pl.col("cp_flag")=='P'))
.then(pl.lit("otm"))
.otherwise(pl.lit('nan')).alias("delta_moneyness")]) 


# %%
path="/Users/ednaloav/Dropbox/AvilaLopez_Martineaua_WSB/wisdom_stocktwits/data/restricted/crsp/"
crsp = pl.read_parquet(path + "crsp_dsf.parquet")
crsp = crsp.with_columns(((pl.col("shrout") * 1000) * pl.col("prc")).alias("mcap"))
crsp = crsp.select(["date", "ticker", "comnam", "mcap", "prc"])

# Join on both date and ticker
df = df.join(
    crsp,
    on=["date", "ticker"],
    how="left"  
)
del crsp

# %%
## Filter the data 

def database_filter(df_, cp_flag):
    df0 = (
        df_.filter(pl.col("maturity").is_in(["0_days" ,"1_7_days"]))
        #df_.filter(pl.col("maturity").is_in(["90_120_days","120_days" ]))
        .filter(pl.col("cp_flag") == cp_flag)
        .filter(pl.col("prc") < 100000)
        .filter(pl.col("f_k_moneyness").is_in(["itm", "otm"]))
    )

    df0 = df0.with_columns((pl.col("prc").round()).alias("prc"))    
    df0 = df0.with_columns(((pl.col("prc") / 5).round() * 5).alias("prc"))   
    df0 = df0.with_columns(((pl.col("mcap") / 1000000000).round() ).alias("mcap"))  

    df0=df0.with_columns(pl.col("prc").log().alias("ln_prc"))
    df0=df0.with_columns(pl.col("mcap").log().alias("ln_mcap"))
    df0=df0.with_columns(pl.col("impl_volatility").log().alias("ln_impl_volatility"))
    return df0

def rdd(df_, x, y, cutoff, var_name):
    # Define grid of candidate cutoffs for MSE optimization
    grid = np.linspace(3, 7, 100)
    # Sharp RDD global linear fit at the given cutoff
    model, df_used, fig_rdd = sharp_rdd_global_linear(df_, y, x, cutoff, var_name=var_name)
    fig_cutoff , best_cutoff = optimal_cutoff_mse(df_used, y=y, x=x, cutoff_grid=grid)  
    return model, best_cutoff, fig_rdd, fig_cutoff


#%%
df_short_call=database_filter(df, "C")
df_short_put=database_filter(df, "P")

#%%
# Price Analysis

## CALL OPTIONS
# Customers <100
df_volume_vs_prc_customer=df_itm_m_otm_vs(df_short_call, 'ln_prc', 'customer_100_dollar_trade_volume','f_k_moneyness')
df0_customer=df_volume_vs_prc_customer.to_pandas().replace([np.inf, -np.inf], np.nan).dropna()
model_cutomer_call, best_cutoff, fig_rdd, fig_cutoff= rdd(df0_customer,'ln_prc','itm_otm',6.3, 'Log Price')
#print(model_cutomer_call.summary())
print(f"Optimal cutoff = {best_cutoff:.3f}")
fig_rdd.show()
fig_cutoff.show()


# For Professionals
df_volume_vs_prc_prof=df_itm_m_otm_vs(df_short_call, 'ln_prc', 'professional_dollar_trade_volume','f_k_moneyness')
df0_prof_call=df_volume_vs_prc_prof.to_pandas().replace([np.inf, -np.inf], np.nan).dropna()
model_prof_call, best_cutoff, fig_rdd, fig_cutoff= rdd(df0_prof_call,'ln_prc','itm_otm',6.3, 'Log Price')
#print(model_prof.summary())
print(f"Optimal cutoff = {best_cutoff:.3f}")
fig_rdd.show()
fig_cutoff.show()


# For Firms
df_volume_vs_prc_firm=df_itm_m_otm_vs(df_short_call, 'ln_prc', 'firm_dollar_trade_volume','f_k_moneyness')
df0_firm_call=df_volume_vs_prc_firm.to_pandas().replace([np.inf, -np.inf], np.nan).dropna()
model_firm_call, best_cutoff, fig_rdd, fig_cutoff= rdd(df0_firm_call,'ln_prc','itm_otm',6.7, 'Log Price')
#print(model_firm.summary())
print(f"Optimal cutoff = {best_cutoff:.3f}")
fig_rdd.show()
fig_cutoff.show()


#%%
# Price Analysis

## PUT OPTIONS
# Customers <100
df_volume_vs_prc_customer=df_itm_m_otm_vs(df_short_put, 'ln_prc', 'customer_100_dollar_trade_volume','f_k_moneyness')
df0_customer=df_volume_vs_prc_customer.to_pandas().replace([np.inf, -np.inf], np.nan).dropna()
model_cutomer_put, best_cutoff, fig_rdd, fig_cutoff= rdd(df0_customer,'ln_prc','itm_otm',6.3, 'Log Price')
print(f"Optimal cutoff = {best_cutoff:.3f}")
fig_rdd.show()
fig_cutoff.show()


# For Professionals
df_volume_vs_prc_prof=df_itm_m_otm_vs(df_short_put, 'ln_prc', 'professional_dollar_trade_volume','f_k_moneyness')
df0_prof_put=df_volume_vs_prc_prof.to_pandas().replace([np.inf, -np.inf], np.nan).dropna()
model_prof_put, best_cutoff, fig_rdd, fig_cutoff= rdd(df0_prof_put,'ln_prc','itm_otm',6.3, 'Log Price')
print(f"Optimal cutoff = {best_cutoff:.3f}")
fig_rdd.show()
fig_cutoff.show()


# For Firms
df_volume_vs_prc_firm=df_itm_m_otm_vs(df_short_put, 'ln_prc', 'firm_dollar_trade_volume','f_k_moneyness')
df0_firm_put=df_volume_vs_prc_firm.to_pandas().replace([np.inf, -np.inf], np.nan).dropna()
model_firm_put, best_cutoff, fig_rdd, fig_cutoff= rdd(df0_firm_put,'ln_prc','itm_otm',6.7, 'Log Price')
print(f"Optimal cutoff = {best_cutoff:.3f}")
fig_rdd.show()
fig_cutoff.show()



# %%
import re
table = summary_col(
    results=[model_cutomer_call, model_prof_call, model_firm_call,model_cutomer_put,model_prof_put,model_firm_put],
    model_names=['Customers <100', 'Professionals', 'Firms','Customers <100', 'Professionals', 'Firms'],
    stars=True,
    float_format='%0.2f',
    regressor_order=['Z', 'Xc', 'Z:Xc', 'Intercept'],
    drop_omitted=True,                  # prevents duplicated rows
)
latex_table = table.as_latex()
latex_table = re.sub(
    r'(\\begin\{tabular\}\{l.*?\}\n)(.*\n)',
    r'\1'
    r'& \\multicolumn{3}{c}{Call} & \\multicolumn{3}{c}{Put} \\\\\n'
    r'\\cline{2-4} \\cline{5-7}\n'
    r'\2',
    latex_table
)


# Step 4. Optional small adjustments for paper formatting
latex_table = latex_table.replace('\\\\\nNotes:', '\\\\[-1.2ex]\n\\midrule\nNotes:')

# Step 5. Output final LaTeX code
print(latex_table)


#%%













#%%

X='ln_prc'
var='itm_otm'
cutoff=6.3
var_name='Log Price'


grid = np.linspace(3, 7, 61)

model1, df_used, df_pred, yhat = sharp_rdd_global_linear(df0_customer, 'itm_otm', X, cutoff, var_name=var_name)
print(model1.summary())
print("Estimated jump (RDD effect) at cutoff:", model1.params['Z'])
c_star, results_df, model_star = optimal_cutoff_mse(df0_customer, y="itm_otm", x=X, cutoff_grid=grid)
print(f"Optimal cutoff = {c_star:.3f}")
print(f"Estimated jump (effect_Z) = {model_star.params['Z']:.3f}")


model2, df_used, df_pred, yhat = sharp_rdd_global_linear(df0_prof, 'itm_otm', X, cutoff, var_name=var_name)
print(model2.summary())
print("Estimated jump (RDD effect) at cutoff:", model2.params['Z'])
c_star, results_df, model_star = optimal_cutoff_mse(df0_prof, y="itm_otm", x=X, cutoff_grid=grid)
print(f"Optimal cutoff = {c_star:.3f}")
print(f"Estimated jump (effect_Z) = {model_star.params['Z']:.3f}")


model3, df_used, df_pred, yhat = sharp_rdd_global_linear(df0_firm, 'itm_otm', X, cutoff, var_name=var_name)
print(model3.summary())
print("Estimated jump (RDD effect) at cutoff:", model3.params['Z'])
c_star, results_df, model_star = optimal_cutoff_mse(df0_firm, y="itm_otm", x=X, cutoff_grid=grid)
print(f"Optimal cutoff = {c_star:.3f}")
print(f"Estimated jump (effect_Z) = {model_star.params['Z']:.3f}")


#%%
modelos=[model1,model2,model3]
results_table_customer = summary_col(
       model1,stars=True,float_format='%0.2f',
        model_names=['Customer <100'],
        info_dict={'N': lambda x: f"{int(x.nobs)}"})
results_table_prof = summary_col(
       model2,stars=True,float_format='%0.2f',
        model_names=['Customer <100'],
        info_dict={'N': lambda x: f"{int(x.nobs)}"})
results_table_firm = summary_col(
       model3,stars=True,float_format='%0.2f',
        model_names=['Customer <100'],
        info_dict={'N': lambda x: f"{int(x.nobs)}"})

   


#%%
## Market Cap
df_volume_vs_mcap=df_itm_m_otm_vs(df_short_call, 'mcap', 'customer_100_dollar_trade_volume','f_k_moneyness')
#graph_itm_m_otm(df_volume_vs_mcap,'mcap', 'Market Cap')

df_volume_vs_mcap_customer=df_itm_m_otm_vs(df_short_call, 'ln_mcap', 'customer_100_dollar_trade_volume','f_k_moneyness')
#graph_itm_m_otm(df_volume_vs_mcap_customer,'ln_mcap', 'Market Cap')

df_volume_vs_mcap_prof=df_itm_m_otm_vs(df_short_call, 'ln_mcap', 'professional_dollar_trade_volume','f_k_moneyness')
#graph_itm_m_otm(df_volume_vs_mcap_prof,'ln_mcap', 'Market Cap')

df_volume_vs_mcap_firm=df_itm_m_otm_vs(df_short_call, 'ln_mcap', 'firm_dollar_trade_volume','f_k_moneyness')
#graph_itm_m_otm(df_volume_vs_mcap_firm,'ln_mcap', 'Market Cap')

#%%
## Volatility
df_volume_vs_vol_customer=df_itm_m_otm_vs(df_short_call, 'impl_volatility', 'customer_100_dollar_trade_volume','f_k_moneyness')
graph_itm_m_otm(df_volume_vs_prc,'impl_volatility', 'Implied Volatility')

df_volume_vs_vol_prof=df_itm_m_otm_vs(df_short_call, 'impl_volatility', 'professional_dollar_trade_volume','f_k_moneyness')
graph_itm_m_otm(df_volume_vs_prc,'impl_volatility', 'Implied Volatility')

df_volume_vs_vol_firm=df_itm_m_otm_vs(df_short_call, 'impl_volatility', 'firm_dollar_trade_volume','f_k_moneyness')
graph_itm_m_otm(df_volume_vs_prc,'impl_volatility', 'Implied Volatility')





#%%
X='ln_mcap'
var='itm_otm'
cutoff=6.3
df0_customer=df_volume_vs_mcap_customer.to_pandas().replace([np.inf, -np.inf], np.nan).dropna()
df0_prof=df_volume_vs_mcap_prof.to_pandas().replace([np.inf, -np.inf], np.nan).dropna()
df0_firm=df_volume_vs_mcap_firm.to_pandas().replace([np.inf, -np.inf], np.nan).dropna()

#%%
grid = np.linspace(3, 7, 61)

model1, df_used, df_pred, yhat = sharp_rdd_global_linear(df0_customer, 'itm_otm', X, cutoff, var_name=var_name)
print(model1.summary())
print("Estimated jump (RDD effect) at cutoff:", model1.params['Z'])
c_star, results_df, model_star = optimal_cutoff_mse(df0_customer, y="itm_otm", x=X, cutoff_grid=grid)
print(f"Optimal cutoff = {c_star:.3f}")
print(f"Estimated jump (effect_Z) = {model_star.params['Z']:.3f}")


model2, df_used, df_pred, yhat = sharp_rdd_global_linear(df0_prof, 'itm_otm', X, cutoff, var_name=var_name)
print(model2.summary())
print("Estimated jump (RDD effect) at cutoff:", model2.params['Z'])
c_star, results_df, model_star = optimal_cutoff_mse(df0_prof, y="itm_otm", x=X, cutoff_grid=grid)
print(f"Optimal cutoff = {c_star:.3f}")
print(f"Estimated jump (effect_Z) = {model_star.params['Z']:.3f}")


model3, df_used, df_pred, yhat = sharp_rdd_global_linear(df0_firm, 'itm_otm', X, cutoff, var_name=var_name)
print(model3.summary())
print("Estimated jump (RDD effect) at cutoff:", model3.params['Z'])
c_star, results_df, model_star = optimal_cutoff_mse(df0_firm, y="itm_otm", x=X, cutoff_grid=grid)
print(f"Optimal cutoff = {c_star:.3f}")
print(f"Estimated jump (effect_Z) = {model_star.params['Z']:.3f}")


#%%
modelos=[model1,model2,model3]
results_table_customer = summary_col(
       model1,stars=True,float_format='%0.2f',
        model_names=['Customer <100'],
        info_dict={'N': lambda x: f"{int(x.nobs)}"})
results_table_prof = summary_col(
       model2,stars=True,float_format='%0.2f',
        model_names=['Customer <100'],
        info_dict={'N': lambda x: f"{int(x.nobs)}"})
results_table_firm = summary_col(
       model3,stars=True,float_format='%0.2f',
        model_names=['Customer <100'],
        info_dict={'N': lambda x: f"{int(x.nobs)}"})

        



# %%
