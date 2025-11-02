# ================================
#  TAREA 3
#  Autor: Camille.A - Adrien.B - Julien.D
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import os

# Importación de datos
data = pd.read_excel("/Users/camilleauvity/Desktop/tarea3-inversiones/MSCI2019Update.xlsx")

print("Aperçu des données :")
print(data.head())

# Calcul des rendimientos aritméticos
returns = data.set_index(data.columns[0]).pct_change().dropna()

print("\nRendimientos mensuales (aperçu) :")
print(returns.head())

# Construcción de los factores

rfree = returns['RFREE']

factors = pd.DataFrame(index=returns.index)
factors['MKT'] = returns['USA MKT'] - rfree
factors['VAL'] = returns['USA_VALUE'] - returns['USA_GROWTH']
factors['CAP'] = returns['USA_SMALL'] - returns['USA_LARGE']
factors['MOM'] = returns['USA_MOMENTUM'] - returns['USA MKT']
factors['QUA'] = returns['USA_QUALITY'] - returns['USA MKT']
factors['VOL'] = returns['USA_VOLATILITY'] - returns['USA MKT']

print("\nFactores creados (aperçu) :")
print(factors.head())

# Estadísticas descriptivas

desc_stats = pd.DataFrame({
    'Mean': factors.mean(),
    'Std Dev': factors.std(),
    'Skewness': factors.apply(skew),
    'Kurtosis': factors.apply(kurtosis)
})

print("\n=== Estadísticas descriptivas ===")
print(desc_stats.round(4))

# Matriz de correlaciones de Pearson

corr_matrix = factors.corr()
print("\n=== Matriz de correlaciones de Pearson ===")
print(corr_matrix.round(3))

# Gráfico de rentabilidad acumulada

cumulative = (1 + factors).cumprod()

plt.figure(figsize=(10,6))
for col in cumulative.columns:
    plt.plot(cumulative.index, cumulative[col], label=col)
plt.title("Rentabilidad Acumulada de Factores (Base = 1)")
plt.xlabel("Fecha")
plt.ylabel("Valor Acumulado")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Enregistrement du graphique
output_folder = "/Users/camilleauvity/Desktop/tarea3-inversiones/"
os.makedirs(output_folder, exist_ok=True)
fig_path = os.path.join(output_folder, "rentabilidad_acumulada_factores.png")
plt.savefig(fig_path, dpi=300)

plt.show()

# Export des résultats vers Excel

excel_path = os.path.join(output_folder, "Resultados_Parte1.xlsx")

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    returns.to_excel(writer, sheet_name="Returns")
    factors.to_excel(writer, sheet_name="Factors")
    desc_stats.to_excel(writer, sheet_name="Descriptive_Stats")
    corr_matrix.to_excel(writer, sheet_name="Correlations")


import pandas as pd
import numpy as np

# Paramètres
window = 60          # fenêtre de 60 mois
gamma = 1            # aversión al riesgo
tau = 1/60           # pour Black–Litterman
N = factors.shape[1] # nombre de factores

dates = factors.index
out_sample_returns = pd.DataFrame(index=dates[window:], columns=['EW','MinVar','BL','VT','RRT','MV'])
weights_dict = {k: [] for k in ['EW','MinVar','BL','VT','RRT','MV']}

# Boucle rolling
for t in range(window, len(factors)):
    sample = factors.iloc[t-window:t]   # fenêtre historique de 60 mois
    mu = sample.mean().values           # moyenne vectorielle
    Sigma = sample.cov().values         # matrice de covariance
    sigma_diag = np.diag(np.diag(Sigma))# matrice diagonale pour VT/RRT
    
    # Equal Weight
    w_EW = np.repeat(1/N, N)
    
    # Minimum Variance
    inv_S = np.linalg.pinv(Sigma)
    ones = np.ones(N)
    w_MinVar = inv_S @ ones / (ones.T @ inv_S @ ones)
    
    # Black–Litterman (simplifié, croyances = moyennes historiques)
    P = np.eye(N)
    Q = mu
    Omega = P @ Sigma @ P.T
    term1 = np.linalg.inv((tau*Sigma)**(-1) + P.T @ np.linalg.inv(Omega) @ P)
    term2 = (tau*Sigma)**(-1) @ (Sigma @ w_EW) + P.T @ np.linalg.inv(Omega) @ Q
    mu_BL = term1 @ term2
    Sigma_BL = Sigma + term1
    inv_SBL = np.linalg.pinv(Sigma_BL)
    w_BL = inv_SBL @ mu_BL / (ones.T @ inv_SBL @ mu_BL)
    
    # Volatility Timing
    sigmas = np.sqrt(np.diag(Sigma))
    inv_var = 1 / sigmas**2
    w_VT = inv_var / np.sum(inv_var)
    
    # Reward-to-Risk Timing
    mu_pos = np.maximum(mu, 0)
    rr_ratio = mu_pos / (sigmas**2)
    if rr_ratio.sum() == 0:
        w_RRT = np.repeat(1/N, N)
    else:
        w_RRT = rr_ratio / rr_ratio.sum()
    
    # Mean–Variance (Markowitz)
    w_MV = inv_S @ mu / (ones.T @ inv_S @ mu)
    
    # Empêche les ventes à découvert
    def normalize_nonneg(w):
        w = np.clip(w, 0, None)
        return w / w.sum() if w.sum() != 0 else np.repeat(1/N, N)
    
    w_EW = normalize_nonneg(w_EW)
    w_MinVar = normalize_nonneg(w_MinVar)
    w_BL = normalize_nonneg(w_BL)
    w_VT = normalize_nonneg(w_VT)
    w_RRT = normalize_nonneg(w_RRT)
    w_MV = normalize_nonneg(w_MV)
    
    r_next = factors.iloc[t].values
    out_sample_returns.loc[dates[t]] = {
        'EW': w_EW @ r_next,
        'MinVar': w_MinVar @ r_next,
        'BL': w_BL @ r_next,
        'VT': w_VT @ r_next,
        'RRT': w_RRT @ r_next,
        'MV': w_MV @ r_next
    }
    
    for name, w in zip(['EW','MinVar','BL','VT','RRT','MV'],
                       [w_EW,w_MinVar,w_BL,w_VT,w_RRT,w_MV]):
        weights_dict[name].append(w)

# Conversion des poids en DataFrames
weights_df = {name: pd.DataFrame(weights_dict[name],
                                 index=dates[window:],
                                 columns=factors.columns)
              for name in weights_dict.keys()}

# Enregistrement des résultats 
output_folder = "/Users/camilleauvity/Desktop/tarea3-inversiones/"
out_sample_returns.to_excel(output_folder + "Retornos_Portafolios_Parte2.xlsx")
for name, df in weights_df.items():
    df.to_excel(output_folder + f"Pesos_{name}_Parte2.xlsx")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

file_path = "/Users/camilleauvity/Desktop/tarea3-inversiones/Retornos_Portafolios_Parte2.xlsx"
assert os.path.exists(file_path), f"File not found: {file_path}"

# Read Excel and list sheets
xls = pd.ExcelFile(file_path)
sheets = xls.sheet_names
sheets_preview = {}
for sh in sheets:
    df = pd.read_excel(xls, sheet_name=sh)
    if df.shape[1] > 0:
        try:
            df.iloc[:,0] = pd.to_datetime(df.iloc[:,0])
            df = df.set_index(df.columns[0])
        except Exception:
            pass
    sheets_preview[sh] = df.head(5)

# Display sheet list
sheet_list_df = pd.DataFrame({"sheet_name": sheets})
print("Sheets in Excel file", sheet_list_df)

# Display previews
for sh, df in sheets_preview.items():
    print(f"Preview - {sh}", df.reset_index().head(10))

# Try to identify portfolio returns and weights by column names
# Load all sheets into dict
data = {sh: pd.read_excel(xls, sheet_name=sh) for sh in sheets}

# Flatten columns list with sheet info to inspect
col_info = []
for sh, df in data.items():
    for col in df.columns:
        col_info.append({"sheet": sh, "column": col})
col_info_df = pd.DataFrame(col_info)
print("Columns found in file", col_info_df)

# Heuristic: find returns columns that contain keywords
keywords_returns = ["return", "ret", "Rp", "portfolio", "R_p", "Rp_t", "Rp"]
keywords_weights = ["weight", "w_", "wt", "ponder", "peso", "w."]
strategy_keywords = ["EW", "MinVar", "Minvar", "MV", "BL", "Black", "VT", "RRT", "RRT-mean", "Mean-Var", "MeanVar", "Naive", "EW", "Naïve"]

# Create unified DataFrame if a sheet looks like time series (date index)
# We'll pick the sheet that has a Date column or first column datetime
time_sheet = None
for sh, df in data.items():
    first_col = df.columns[0]
    try:
        tmp = pd.to_datetime(df[first_col])
        time_sheet = sh
        break
    except Exception:
        continue

if time_sheet is None:
    # fallback: use the first sheet
    time_sheet = sheets[0]

ts_df = pd.read_excel(xls, sheet_name=time_sheet)
first_col = ts_df.columns[0]
try:
    ts_df[first_col] = pd.to_datetime(ts_df[first_col])
    ts_df = ts_df.set_index(first_col)
except Exception:
    pass

# Identify candidate return columns and weight columns across all sheets
ret_cols = []
weight_cols = []
rf_cols = []
for sh, df in data.items():
    local = df.copy()
    try:
        local.iloc[:,0] = pd.to_datetime(local.iloc[:,0])
        local = local.set_index(local.columns[0])
    except Exception:
        pass
    for col in local.columns:
        col_lower = str(col).lower()
        if any(k.lower() in col_lower for k in keywords_returns) or any(s.lower() in col_lower for s in strategy_keywords):
            if pd.api.types.is_numeric_dtype(local[col]):
                ret_cols.append({"sheet": sh, "column": col})
        if any(k.lower() in col_lower for k in keywords_weights):
            if pd.api.types.is_numeric_dtype(local[col]):
                weight_cols.append({"sheet": sh, "column": col})
        if "rf" in col_lower or "riskfree" in col_lower or "r_free" in col_lower:
            rf_cols.append({"sheet": sh, "column": col})

ret_cols_df = pd.DataFrame(ret_cols).drop_duplicates().reset_index(drop=True)
weight_cols_df = pd.DataFrame(weight_cols).drop_duplicates().reset_index(drop=True)
rf_cols_df = pd.DataFrame(rf_cols).drop_duplicates().reset_index(drop=True)

print("Candidate return columns (heuristic)", ret_cols_df)
print("Candidate weight columns (heuristic)", weight_cols_df)
print("Candidate Rf columns (heuristic)", rf_cols_df)

# For the main analysis, try to find a sheet that contains portfolio returns per strategy in columns.
# Heuristic: find sheet with columns that include many strategy names (EW, MinVar, MV, BL, VT, RRT)
candidate_sheet = None
for sh, df in data.items():
    cols = " ".join([str(c).lower() for c in df.columns])
    
    strategy_keywords = ["ew", "minvar", "mv", "black", "bl", "vt", "rrt", "rrt-mean"]
    
    main_strategies = ["ew", "minvar", "mv"]
    count = sum(1 for s in main_strategies if any(s in c for c in cols.split()))
    
    if count >= 2:
        print(f"{sh}: au moins deux stratégies principales détectées dans les colonnes.")
        pass


# Simpler: pick the sheet that has the most numeric columns (likely the results sheet)
max_num = -1
for sh, df in data.items():
    num = sum(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
    if num > max_num:
        max_num = num
        candidate_sheet = sh

candidate_sheet, max_num
main_df = pd.read_excel(xls, sheet_name=candidate_sheet)
# Try parse date index
try:
    main_df.iloc[:,0] = pd.to_datetime(main_df.iloc[:,0])
    main_df = main_df.set_index(main_df.columns[0])
except Exception:
    pass

print(f"Main sheet selected: {candidate_sheet} (preview)", main_df.reset_index().head(10))

# Now attempt to automatically detect portfolio return series inside main_df
cols = list(main_df.columns)
# Identify possible portfolio return columns by looking for typical names or values range (-1..1)
possible_returns = []
for col in cols:
    series = main_df[col]
    if pd.api.types.is_numeric_dtype(series):
        # check if values look like returns (small floats)
        if series.dropna().between(-1, 1).all() and series.dropna().abs().mean() < 0.5:
            possible_returns.append(col)

# Also find groups of weight columns by pattern: columns summing approx 1 over rows
possible_weight_groups = {}
for col in cols:
    if "weight" in str(col).lower() or "w_" in str(col).lower() or "peso" in str(col).lower() or "ponder" in str(col).lower():
        possible_weight_groups[col] = main_df[col]

# Another approach: check for columns that sum to 1 across a row (weights for strategies grouped by prefix)
# We'll search for prefixes before an underscore
prefixes = {}
for col in cols:
    if "_" in str(col):
        pref = str(col).split("_")[0]
        prefixes.setdefault(pref, []).append(col)
# compute average row-sum for groups with >=2 columns
for pref, group in prefixes.items():
    grp_sum = main_df[group].sum(axis=1)
    mean_sum = grp_sum.mean()
    if len(group) >= 2 and np.isfinite(mean_sum):
        possible_weight_groups[pref] = {"cols": group, "mean_sum": mean_sum}

print("Detected possible return columns (simple heuristic)", pd.DataFrame({"possible_returns": possible_returns}))
print("Detected weight groups (by prefix)", pd.DataFrame([{**{"prefix":k}, **({"cols": v} if isinstance(v, list) else v)} for k,v in possible_weight_groups.items()]))

# Compute performance metrics for each candidate return series discovered
# Find a risk-free series if possible; else assume zero
rf_series = None
rf_name = None
if not rf_cols_df.empty:
    rf_entry = rf_cols_df.iloc[0]
    rf_df = pd.read_excel(xls, sheet_name=rf_entry['sheet'])
    try:
        rf_df.iloc[:,0] = pd.to_datetime(rf_df.iloc[:,0])
        rf_df = rf_df.set_index(rf_df.columns[0])
        rf_series = rf_df[rf_entry['column']]
        rf_name = f"{rf_entry['sheet']}::{rf_entry['column']}"
    except Exception:
        rf_series = None

# Define functions for metrics (assuming monthly data unless detected)
def infer_periods_per_year(index):
    # infer average gap in days
    if hasattr(index, "to_series"):
        idx = pd.to_datetime(index)
        diffs = idx.to_series().diff().dropna().dt.days
        median = diffs.median()
        if median >= 27 and median <= 31:
            return 12
        if median >= 6 and median <= 9:
            return 52
        if median >= 0 and median <= 2:
            return 252
    return 12

periods_per_year = infer_periods_per_year(main_df.index)

def annualize_return(r_mean, periods=12):
    return (1 + r_mean) ** periods - 1

def annualize_std(s, periods=12):
    return s * (periods ** 0.5)

def sharpe_ratio(series, rf=0.0, periods=12):
    excess = series - rf
    mean_excess = excess.mean()
    std_excess = excess.std(ddof=1)
    # annualize
    ann_mean = annualize_return(mean_excess, periods)
    ann_std = annualize_std(std_excess, periods)
    if ann_std == 0:
        return np.nan
    return ann_mean / ann_std

def downside_risk(series, periods=12):
    # DR = sqrt(1/T sum(min(R, 0)^2))
    downs = series.copy()
    downs = downs[downs < 0].fillna(0)
    dr = np.sqrt((downs**2).mean())
    return annualize_std(dr, periods)  # approximate

def sortino_ratio(series, rf=0.0, periods=12):
    er = series.mean()
    dr = downside_risk(series, periods)
    if dr == 0:
        return np.nan
    ann_return = annualize_return(er, periods)
    return ann_return / dr

def max_drawdown(series):
    # series are periodic returns; compute cumulative wealth
    wealth = (1 + series).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth / running_max) - 1
    mdd = drawdown.min()
    return -mdd

def mppm(series, rf_series=None, rho=4, periods=12):
    # MPPM = (1/(1+rho)) * ln( (1/T) sum [ ((1+Rp)/(1+Rf))^(1-rho) ] )
    if rf_series is None:
        rf = 0.0
        rr = ((1 + series) / (1 + rf)) ** (1 - rho)
    else:
        # align
        rf_aligned = rf_series.reindex(series.index).fillna(method='ffill').fillna(0)
        rr = ((1 + series) / (1 + rf_aligned)) ** (1 - rho)
    val = rr.mean()
    if val <= 0:
        return np.nan
    return (1 / (1 + rho)) * np.log(val) * periods  # annualized approx

# Build metrics table
metrics = []
returns_table = pd.DataFrame()
for col in possible_returns:
    series = main_df[col].astype(float).dropna()
    # align rf
    if rf_series is not None:
        rf_al = rf_series.reindex(series.index)
        rf_monthly = rf_al
        rf_val = rf_monthly.mean() if len(rf_monthly.dropna())>0 else 0.0
    else:
        rf_val = 0.0
    sr = sharpe_ratio(series, rf=rf_val, periods=periods_per_year)
    dr = downside_risk(series, periods=periods_per_year)
    sortino = sortino_ratio(series, rf=rf_val, periods=periods_per_year)
    mdd = max_drawdown(series)
    mppm_val = mppm(series, rf_series=None, rho=4, periods=periods_per_year)
    metrics.append({"series": col, "Sharpe": sr, "DownsideRisk": dr, "Sortino": sortino, "MDD": mdd, "MPPM": mppm_val})
    returns_table[col] = series

metrics_df = pd.DataFrame(metrics).set_index("series")
print("Performance metrics (heuristic detection)", metrics_df)

# Plot cumulative returns for the detected return series
plt.figure(figsize=(10,6))
for col in returns_table.columns:
    series = returns_table[col].fillna(0)
    cum = (1 + series).cumprod()
    plt.plot(cum.index, cum.values, label=col)
plt.title("Cumulative returns (detected series)")
plt.xlabel("Date")
plt.ylabel("Cumulative wealth")
plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/camilleauvity/Desktop/tarea3-inversiones/cumulative_returns.png")
plt.close()

# Save weight stacked plots for prefixes detected earlier
weight_plot_paths = []
for pref, info in possible_weight_groups.items():
    if isinstance(info, dict) and "cols" in info:
        cols = info["cols"]
        dfw = main_df[cols].fillna(0)
        row_sums = dfw.sum(axis=1)
        dfw_norm = dfw.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
        plt.figure(figsize=(10,5))
        dfw_norm.plot.area()
        plt.title(f"Stacked weights over time - group: {pref}")
        plt.xlabel("Date")
        plt.ylabel("Weight (stacked)")
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
        path = f"/mnt/data/weights_{pref}.png"
        plt.savefig(path)
        plt.close()
        weight_plot_paths.append(path)

metrics_df.to_csv("/Users/camilleauvity/Desktop/tarea3-inversiones/metrics_detected.csv")
returns_table.to_csv("/Users/camilleauvity/Desktop/tarea3-inversiones/returns_detected.csv")

# Provide download links
output_files = {
    "cumulative_plot": "/Users/camilleauvity/Desktop/tarea3-inversiones/cumulative_returns.png",
    "metrics_csv": "/Users/camilleauvity/Desktop/tarea3-inversiones/metrics_detected.csv",
    "returns_csv": "/Users/camilleauvity/Desktop/tarea3-inversiones/returns_detected.csv",
    "weight_plots": weight_plot_paths
}

print("Done. Output files saved:", output_files)
output_files

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Pesos_VT_Parte2.xlsx")
df['Date'] = pd.to_datetime(df['Date'])

plt.figure(figsize=(12, 6))
plt.stackplot(df['Date'],
              df['MKT'], df['VAL'], df['CAP'],
              df['MOM'], df['QUA'], df['VOL'],
              labels=['MKT', 'VAL', 'CAP', 'MOM', 'QUA', 'VOL'],
              alpha=0.7)

plt.title("Pesos de los activos en la cartera (VT)")
plt.xlabel("Fecha")
plt.ylabel("Ponderación")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
