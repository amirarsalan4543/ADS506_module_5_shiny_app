from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from shiny import App, reactive, render, ui

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).parent / "AustralianWines.csv"


# Load CSV into a clean monthly table keyed by Varietal
def load_wine_data():
    df = pd.read_csv(DATA_PATH, na_values=["*"])
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"sparkling": "Sparkling"})
    df["Month"] = pd.to_datetime(df["Month"], format="%b-%y")

    long_df = df.melt(id_vars="Month", var_name="Varietal", value_name="Sales")
    long_df["Sales"] = pd.to_numeric(long_df["Sales"], errors="coerce")
    long_df = long_df.sort_values(["Varietal", "Month"]).reset_index(drop=True)

    # Fill the 2 missing Rose months (Jul-Aug 1994)
    long_df["Sales"] = long_df.groupby("Varietal")["Sales"].transform(lambda s: s.interpolate())
    return long_df


# Accuracy helper: RMSE, MAE, MAPE (%)
def calc_metrics(actual, pred):
    err = np.array(actual) - np.array(pred)
    rmse = round(float(np.sqrt(np.mean(err ** 2))), 2)
    mae = round(float(np.mean(np.abs(err))), 2)
    mape = round(float(np.mean(np.abs(err / np.array(actual))) * 100), 2)
    return rmse, mae, mape


# 1) Linear regression with trend + monthly seasonality
def fit_tslm(train_s, steps):
    n = len(train_s)
    train_df = pd.DataFrame({"y": train_s.values, "t": np.arange(n), "m": train_s.index.month})
    fit = smf.ols("y ~ t + C(m)", data=train_df).fit()

    future_idx = pd.date_range(train_s.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    future_df = pd.DataFrame({"t": np.arange(n, n + steps), "m": future_idx.month})
    pred = fit.get_prediction(future_df).summary_frame(alpha=0.05)

    fc = pd.DataFrame({
        "Month": future_idx,
        "Mean": pred["mean"].values,
        "Low95": pred["obs_ci_lower"].values,
        "High95": pred["obs_ci_upper"].values,
    })
    return fit.fittedvalues.values, fc, "TSLM(trend + season)"


# 2) Auto-ETS: pick lowest AICc from standard seasonal forms
def fit_ets(train_s, steps):
    candidates = [
        ("add", "add", "add"),
        ("add", None, "add"),
        ("mul", "add", "mul"),
        ("mul", None, "mul"),
    ]
    best_fit, best_cfg, best_aicc = None, None, np.inf
    for err, tr, seas in candidates:
        try:
            fit = ETSModel(train_s, error=err, trend=tr, seasonal=seas, seasonal_periods=12).fit(disp=False)
            if fit.aicc < best_aicc:
                best_fit, best_cfg, best_aicc = fit, (err, tr, seas), fit.aicc
        except Exception:
            continue

    if best_fit is None:
        best_cfg = ("add", None, "add")
        best_fit = ETSModel(train_s, error="add", seasonal="add", seasonal_periods=12).fit(disp=False)

    code = {None: "N", "add": "A", "mul": "M"}
    spec = f"ETS({code[best_cfg[0]]},{code[best_cfg[1]]},{code[best_cfg[2]]})"

    future_idx = pd.date_range(train_s.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    summary = best_fit.get_prediction(start=len(train_s), end=len(train_s) + steps - 1).summary_frame(alpha=0.05)
    low_col = "pi_lower" if "pi_lower" in summary.columns else summary.columns[2]
    high_col = "pi_upper" if "pi_upper" in summary.columns else summary.columns[3]

    fc = pd.DataFrame({
        "Month": future_idx,
        "Mean": summary["mean"].values,
        "Low95": summary[low_col].values,
        "High95": summary[high_col].values,
    })
    return best_fit.fittedvalues.values, fc, spec


# 3) Auto-ARIMA: pick lowest AIC from standard seasonal ARIMA orders
def fit_arima(train_s, steps):
    candidates = [
        ((0, 1, 1), (0, 1, 1, 12)),
        ((1, 1, 0), (0, 1, 1, 12)),
        ((1, 1, 1), (0, 1, 1, 12)),
        ((1, 0, 0), (0, 1, 1, 12)),
    ]
    best_fit, best_order, best_aic = None, None, np.inf
    for order, s_order in candidates:
        try:
            fit = SARIMAX(
                train_s, order=order, seasonal_order=s_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            if fit.aic < best_aic:
                best_fit, best_order, best_aic = fit, (order, s_order), fit.aic
        except Exception:
            continue

    if best_fit is None:
        best_order = ((0, 1, 1), (0, 1, 1, 12))
        best_fit = SARIMAX(train_s, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)

    (p, d, q), (P, D, Q, m) = best_order
    spec = f"ARIMA({p},{d},{q})({P},{D},{Q})[{m}]"

    future_idx = pd.date_range(train_s.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    fc_res = best_fit.get_forecast(steps=steps)
    ci = fc_res.conf_int(alpha=0.05)

    fc = pd.DataFrame({
        "Month": future_idx,
        "Mean": fc_res.predicted_mean.values,
        "Low95": ci.iloc[:, 0].values,
        "High95": ci.iloc[:, 1].values,
    })
    return best_fit.fittedvalues.values, fc, spec


# Fit selected models for each varietal and collect outputs
def run_models(df, varietals, train_end, horizon, selected_models):
    forecasts, metrics, specs = [], [], []
    fitters = {"TSLM": fit_tslm, "ETS": fit_ets, "ARIMA": fit_arima}

    for var in varietals:
        sub = df[df["Varietal"] == var].sort_values("Month")
        series = sub.set_index("Month")["Sales"].asfreq("MS")
        train_s = series[series.index <= train_end]
        val_s = series[series.index > train_end]

        if len(train_s) < 24:
            continue

        steps = max(len(val_s), horizon)
        for name in selected_models:
            fitted, fc, spec = fitters[name](train_s, steps)

            fc_out = fc.iloc[:horizon].copy()
            fc_out["Varietal"] = var
            fc_out["Model"] = name
            forecasts.append(fc_out)
            specs.append({"Varietal": var, "Model": name, "Specification": spec})

            rmse, mae, mape = calc_metrics(train_s.values, fitted)
            metrics.append({"Varietal": var, "Model": name, "Window": "Train", "RMSE": rmse, "MAE": mae, "MAPE (%)": mape})

            if len(val_s) > 0:
                rmse, mae, mape = calc_metrics(val_s.values, fc["Mean"].iloc[:len(val_s)].values)
                metrics.append({"Varietal": var, "Model": name, "Window": "Validation", "RMSE": rmse, "MAE": mae, "MAPE (%)": mape})

    fc_df = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()
    return fc_df, pd.DataFrame(metrics), pd.DataFrame(specs)


WINE_DF = load_wine_data()
VARIETALS = sorted(WINE_DF["Varietal"].unique().tolist())
COLORS = {"TSLM": "tab:blue", "ETS": "tab:red", "ARIMA": "tab:green"}

# --- UI ---
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h5("Controls"),
        ui.input_checkbox_group("varietals", "Varietals", choices=VARIETALS, selected=["Sparkling", "Red"]),
        ui.input_slider("years", "Date Range (Years)", min=1980, max=1994, value=[1980, 1994], sep=""),
        ui.input_slider("train_end_year", "Train Split End Year", min=1982, max=1993, value=1992, sep=""),
        ui.input_slider("horizon", "Forecast Horizon (Months)", min=6, max=36, value=24, step=1),
        ui.input_checkbox_group("models", "Models", choices=["TSLM", "ETS", "ARIMA"], selected=["TSLM", "ETS", "ARIMA"]),
        ui.hr(),
        ui.download_button("download_forecasts", "Download Forecasts (CSV)"),
        width=270,
    ),
    ui.navset_tab(
        ui.nav_panel("Overview", ui.card(ui.output_plot("overview_plot", height="500px"))),
        ui.nav_panel("Forecasts", ui.card(ui.output_plot("forecast_plot", height="620px"))),
        ui.nav_panel(
            "Specs & Accuracy",
            ui.card(ui.card_header("Model Specifications"), ui.output_table("specs_table")),
            ui.card(ui.card_header("Train vs. Validation Accuracy"), ui.output_table("metrics_table")),
        ),
        ui.nav_panel(
            "About",
            ui.card(
                ui.markdown(
                    """
                    **Data:** Monthly Australian wine sales (`AustralianWines.csv`, 1980–1994) organized by
                    `Month` and `Varietal`. Two missing `Rose` values (`*` in Jul–Aug 1994) are linearly interpolated.

                    **Models:** Auto-selected `TSLM`, `ETS`, and seasonal `ARIMA` fit on the training window,
                    evaluated on Train and Validation windows (`RMSE`, `MAE`, `MAPE`), with 95% prediction intervals.

                    **Note:** Built with Shiny for Python (`pandas`, `numpy`, `matplotlib`, `statsmodels`).
                    An LLM was used to help draft boilerplate code.
                    """
                )
            ),
        ),
    ),
    title="Australian Wine Sales Forecasting",
)


# --- Server ---
def server(input, output, session):
    @reactive.calc
    def filtered_df():
        vars_sel = list(input.varietals())
        if not vars_sel:
            return pd.DataFrame(columns=WINE_DF.columns)
        start_y, end_y = input.years()
        mask = (
            WINE_DF["Varietal"].isin(vars_sel)
            & (WINE_DF["Month"].dt.year >= start_y)
            & (WINE_DF["Month"].dt.year <= end_y)
        )
        return WINE_DF.loc[mask].copy()

    @reactive.calc
    def model_results():
        df = filtered_df()
        vars_sel, mods_sel = list(input.varietals()), list(input.models())
        if df.empty or not vars_sel or not mods_sel:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        train_end = pd.Timestamp(year=input.train_end_year(), month=12, day=1)
        return run_models(df, vars_sel, train_end, input.horizon(), mods_sel)

    @render.plot
    def overview_plot():
        df = filtered_df()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        if df.empty:
            ax.text(0.5, 0.5, "Select at least one varietal.", ha="center", va="center")
            ax.axis("off")
            return fig

        for var, grp in df.groupby("Varietal"):
            ax.plot(grp["Month"], grp["Sales"], label=var)
        ax.set_title("Monthly Australian Wine Sales")
        ax.set_xlabel("Month")
        ax.set_ylabel("Sales")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    @render.plot
    def forecast_plot():
        df = filtered_df()
        fc_df, _, _ = model_results()
        vars_sel = list(input.varietals())

        if df.empty or fc_df.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.text(0.5, 0.5, "Select at least 1 varietal, 1 model, and 2+ years of training data.", ha="center", va="center")
            ax.axis("off")
            return fig

        fig, axes = plt.subplots(len(vars_sel), 1, figsize=(9.5, 3.2 * len(vars_sel)), squeeze=False)
        train_end = pd.Timestamp(year=input.train_end_year(), month=12, day=1)

        for i, var in enumerate(vars_sel):
            ax = axes[i, 0]
            hist = df[df["Varietal"] == var].sort_values("Month")
            train_part = hist[hist["Month"] <= train_end]
            val_part = hist[hist["Month"] > train_end]

            ax.plot(train_part["Month"], train_part["Sales"], color="gray", label="Train Actual")
            if not val_part.empty:
                ax.plot(val_part["Month"], val_part["Sales"], color="black", linestyle="--", label="Validation Actual")
            ax.axvline(train_end, color="gray", linestyle=":", label="Train/Val Split")

            for mod, grp in fc_df[fc_df["Varietal"] == var].groupby("Model"):
                c = COLORS[mod]
                ax.plot(grp["Month"], grp["Mean"], color=c, label=f"{mod} Forecast")
                ax.fill_between(grp["Month"], grp["Low95"], grp["High95"], color=c, alpha=0.15, label=f"{mod} 95% PI")

            ax.set_title(f"{var} — Forecast Comparison")
            ax.set_ylabel("Sales")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8, ncol=2)

        axes[-1, 0].set_xlabel("Month")
        fig.tight_layout()
        return fig

    @render.table
    def specs_table():
        _, _, specs_df = model_results()
        return specs_df if not specs_df.empty else pd.DataFrame({"Message": ["Adjust controls to fit models."]})

    @render.table
    def metrics_table():
        _, metrics_df, _ = model_results()
        return metrics_df if not metrics_df.empty else pd.DataFrame({"Message": ["Adjust controls to view accuracy."]})

    @render.download(filename="wine_forecasts.csv")
    def download_forecasts():
        fc_df, _, _ = model_results()
        yield fc_df.to_csv(index=False)


app = App(app_ui, server)
