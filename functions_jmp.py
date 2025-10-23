
#%%
import numpy as np
import pandas as pd
import polars as pl
import os

import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

#%%

def summary_stats_latex_tab(
    tab,
    panel_title,
    begin_tab=True,
    end_tab=True,
):
    rows = list(tab.index)
    n_cols = tab.shape[1]

    tot_cols_for_cmidrule = n_cols + 1

    header = "\\begin{tabular}{r" + ("c" * n_cols) + "}\n" if begin_tab else ""
    header += "\\\\\n"
    header += (
        "\\multicolumn{"
        + str(tot_cols_for_cmidrule)
        + "}{l}{"
        + panel_title
        + "} \\\\\n"
    )
    header += "\\hline\\hline\n"
    header += "{} &  \multicolumn{3}{c}{Small Customers} & \multicolumn{3}{c}{Professionals} & \multicolumn{3}{c}{Firms}   \\\\\n"   
    header += "{} &  ITM &  OTM &  ATM &  ITM &  OTM &  ATM&  ITM &  OTM &  ATM  \\\\\n"
    header += "\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\n"

    col_names_cnt = []
    for col in tab.columns:
        if col_names_cnt and (col == col_names_cnt[-1][0]):
            col_names_cnt[-1][1] += 1
        else:
            col_names_cnt.append([col, 1])
    format_multicolumn = lambda col, cnt: (
        col if cnt == 1 else "\multicolumn{" + str(cnt) + "}{c}{" + col + "}"
    )
    midrule_cnt = 1
    for _, cnt in col_names_cnt:
        header += (
            "\cmidrule(lr){" + str(midrule_cnt + 1) + "-" + str(midrule_cnt + cnt) + "}"
        )
        midrule_cnt += cnt
    header += "\n"

    body = ""
    for row in rows:
        body += f"{str(row)}&"
        body += (
            " & ".join(
                [
                    (
                        tab.iloc[:, i][row]
                        if isinstance(tab.iloc[:, i][row], str)
                        else "{:,.0f}".format(tab.iloc[:, i][row])
                    )
                    for i, _ in enumerate(tab.columns)
                ]
            )
            + "\\\\\n "
        )

    footer = "\\hline\\hline\n" + "\\end{tabular}\n" if end_tab else "\\hline \n"
    return header + body + footer



def df_itm_m_otm_vs(df_, vs_var, investor,moneyness):
    df_ = (df_.group_by(["date", vs_var, "ticker", moneyness])
    .agg(pl.col(investor).sum().alias("avg_volume"))
    .pivot(values="avg_volume",
        index=["date", vs_var, "ticker"],
        columns=moneyness)
    .select(["date", "ticker", vs_var, "itm", "otm"])
    .sort(["date", "ticker", vs_var]).with_columns([
        pl.col("itm").fill_null(0),
        pl.col("otm").fill_null(0) ]))
    #
    df_ = df_.with_columns(
    ((pl.col("itm") - pl.col("otm")) / (pl.col("itm") + pl.col("otm"))).alias("itm_otm"))

    df_=df_.filter((pl.col("itm") + pl.col("otm")) > 0) 
    #
    df_avg = (df_.group_by(["ticker", vs_var])
        .agg(pl.col("itm_otm").mean().alias("itm_otm"))
        .sort(["ticker", vs_var]))
    #
    df_avg = (df_avg.group_by([vs_var])
        .agg(pl.col("itm_otm").mean().alias("itm_otm"))
        .sort(vs_var))
    
    return df_avg

def graph_itm_m_otm(df_, vs_var, name_vs_var):
    fig = px.scatter(
        df_,
        x=vs_var,
        y="itm_otm",
        #color=color_var, 
        color_continuous_scale="viridis",
        opacity=0.7,
        title="Option Trading Volume vs" + str(name_vs_var)
    )

    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.update_layout(
        width=800,
        height=600,
        xaxis_title=name_vs_var,
        yaxis_title="ITM / OTM",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    fig.show()



def sharp_rdd_global_linear(
    df, y, x, cutoff, cluster=None, plot=True, var_name=None):
    """
    Sharp RDD with global linear specification using the full range of X:
        Y ~ Z + Xc + Z:Xc
    where Z = 1{X >= cutoff} and Xc = X - cutoff.

    Returns:
      model, df_used, df_pred, yhat
    """
    dat = df.copy()
    dat['Xc'] = dat[x] - cutoff
    dat['Z']  = (dat[x] >= cutoff).astype(int)

    # estimate model
    formula = f"{y} ~ Z + Xc + Z:Xc"
    if cluster is not None:
        model = smf.ols(formula, data=dat).fit(
            cov_type="cluster", cov_kwds={"groups": dat[cluster]}
        )
    else:
        model = smf.ols(formula, data=dat).fit(cov_type="HC1")

    # predictions
    x_vals = np.linspace(dat[x].min(), dat[x].max(), 400)
    df_pred = pd.DataFrame({x: x_vals})
    df_pred['Xc'] = df_pred[x] - cutoff
    df_pred['Z']  = (df_pred[x] >= cutoff).astype(int)
    yhat = model.predict(df_pred)

    #if plot:
    #    plt.figure(figsize=(8,5))
    #    plt.scatter(dat[x], dat[y], alpha=0.25, s=8, label='Observed data')
    #    plt.plot(df_pred[x], yhat, color='red', linewidth=2, label='Fitted values')
    #    plt.axvline(cutoff, color='gray', linestyle='--', label='Cutoff')
    #    plt.axhline(0, color='black', linestyle='-', lw=0.8)
    #    #plt.title('Sharp RDD: Global Linear Fit')
    #    plt.xlabel('Running variable')
    #    plt.ylabel('ITM/OTM Ratio')
    #    plt.legend()
    #    plt.show()

    if plot:
        fig = go.Figure()

        # Scatter plot of observed data
        fig.add_trace(go.Scatter(x=dat[x],y=dat[y],mode='markers',name='Observed data',
            opacity=0.5,marker=dict(size=5, color='darkblue')))

        # Fitted line
        fig.add_trace(go.Scatter(x=df_pred[x],y=yhat,mode='lines',name='Fitted values',
            line=dict(color='red', width=2)))

        # Vertical cutoff line
        fig.add_vline(x=cutoff,line=dict(color='black', width=1.5, dash='dash'),
            annotation_text='Cutoff',annotation_position='top')

        # Horizontal zero line (thin)
        fig.add_hline(y=0,line=dict(color='black', width=1.5))

        # Layout settings
        fig.update_layout(width=800, height=600,
        xaxis_title=var_name, yaxis_title='ITM/OTM Ratio',showlegend=True,template='simple_white',
        legend=dict(x=0.02, y=0.08, bgcolor='rgba(255,255,255,0.7)', 
        bordercolor='gray',borderwidth=0.5
    ))

    fig.show()

    return model, dat, df_pred, yhat



# ---------- Sharp RDD function ----------
def sharp_rdd_global_linear_mse(df, y, x, cutoff):
    """
    Sharp RDD using full range of X with global piecewise linear specification:
        Y ~ Z + Xc + Z:Xc
    where Z = 1{X >= cutoff}, Xc = X - cutoff.
    """
    dat = df.copy()
    dat['Xc'] = dat[x] - cutoff
    dat['Z']  = (dat[x] >= cutoff).astype(int)
    model = smf.ols(f"{y} ~ Z + Xc + Z:Xc", data=dat).fit(cov_type="HC1")
    return model

# ---------- Optimal cutoff search ----------
def optimal_cutoff_mse(df, y, x, cutoff_grid=None, plot=True):
    """
    Search over candidate cutoffs and choose the one minimizing in-sample MSE.
    Returns:
      optimal_cutoff, results_df, model_at_opt
    """
    if cutoff_grid is None:
        xmin, xmax = df[x].min(), df[x].max()
        cutoff_grid = np.linspace(xmin, xmax, 101)

    results = []
    best_cutoff, best_mse, best_model = None, np.inf, None

    for c in cutoff_grid:
        model = sharp_rdd_global_linear_mse(df, y, x, cutoff=c)
        mse = mean_squared_error(df[y], model.fittedvalues)
        effect = model.params.get("Z", np.nan)
        results.append({"cutoff": c, "mse": mse, "effect_Z": effect})
        if mse < best_mse:
            best_cutoff, best_mse, best_model = c, mse, model

    results_df = pd.DataFrame(results)

    # ---------- Plot ----------
    #if plot:
    #    plt.figure(figsize=(8,5))
    #    plt.plot(results_df["cutoff"], results_df["mse"], marker="o", color="orange", label="MSE by cutoff")
    #    plt.axvline(best_cutoff, color="red", linestyle="--", label=f"Optimal cutoff = {best_cutoff:.3f}")
    #    plt.title("Sharp RDD: Cutoff Level Minimizing MSE")
    #    plt.xlabel("Candidate cutoff")
    #    plt.ylabel("Mean Squared Error (MSE)")
    #    plt.legend()
    #    plt.show()
    #    print(f"Optimal cutoff (min MSE): {best_cutoff:.3f}")

    return best_cutoff, results_df, best_model


