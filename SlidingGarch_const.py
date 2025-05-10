# mean=const, AR
# o=1: GJR, o=0: GARCH

# GJR-GARCH, mean=Const
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import chi2
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

class SlidingGARCH:
    def __init__(self, tickers=["^GSPC", "GLD", "USO"], start="2020-01-01", end="2025-01-01",
                 freq="1d", return_type="log", mean="Zero", vol="GARCH",
                 p=1, o=None, q=1, dist="t", window=250, power=2.0):
        """
        Parameters:
          tickers: List of stock ticker symbols (e.g., ["^GSPC", "GLD", "USO"]).
          start, end: Data start and end dates.
          freq: Data frequency (passed to yfinance's interval, e.g., "1d", "1wk").
          return_type: "log" for log returns, "pct" for simple percentage returns.
          mean: Mean model (e.g., "Zero", "AR").
          vol: Volatility model type (e.g., "GARCH").
          p, o, q: GARCH model orders; if no leverage effect is needed, set o=0.
          dist: Residual distribution (e.g., "StudentsT", "skewt", "ged").
          window: Sliding window length (number of days).
          power: The power parameter for the GARCH model (e.g., 1.0 for IGARCH constraint).
        """
        self.tickers = tickers
        self.start = start
        self.end = end
        self.freq = freq
        self.return_type = return_type.lower()  # "log" or "pct"
        self.mean = mean
        self.vol = vol
        self.p = p
        self.o = o if o is not None else 0
        self.q = q
        self.dist = dist
        self.window = window
        self.power = power

        self.returns_dict = self.get_data()  # Dictionary of returns for each ticker and portfolio
        self.results_dict = {}  # To store results_df for each asset
        self.last_result_dict = {}  # To store last_result for each asset
        self.var_dict = {}  # To store VaR_series for each asset

    def get_data(self):
        """
        Download stock data using yfinance and compute returns for each ticker and portfolio.
        """
        returns_dict = {}
        csv_file = "data-ggu1.csv"
        if os.path.exists(csv_file):
            print(f"Loading data from {csv_file} …")
            data = pd.read_csv(
                csv_file,
                header=[0,1,2],
                index_col=0,
                parse_dates=True
            )
        # data = yf.download(self.tickers, start=self.start, end=self.end, interval=self.freq, threads=False)
        for ticker in self.tickers:
            if self.return_type == "log":
                close_series = data['Close'][ticker].iloc[:, 0]
                ret = np.log(close_series / close_series.shift(1)).dropna()
                print(f"Downloaded data for {ticker} using log returns, sample length: {len(ret)}")
            elif self.return_type == "pct":
                ret = (close_series / close_series.shift(1) - 1).dropna()
                print(f"Downloaded data for {ticker} using percentage returns, sample length: {len(ret)}")
            else:
                raise ValueError("Unknown return_type. Choose 'log' or 'pct'.")
            returns_dict[ticker] = ret

        # Calculate equally weighted GLD-USO portfolio returns
        gld_returns = returns_dict["GLD"]
        uso_returns = returns_dict["USO"]
        portfolio_returns = 0.5 * gld_returns + 0.5 * uso_returns
        # Ensure portfolio_returns is a Pandas Series
        returns_dict["GLD-USO"] = pd.Series(portfolio_returns, name="GLD-USO").dropna()
        print(f"Calculated GLD-USO portfolio returns, sample length: {len(portfolio_returns)}")
        #print(f"Calculated GLD-USO portfolio returns, sample length: {len(portfolio_returns.dropna())}")

        return returns_dict

    def preprocess(self, ticker):
        """
        Preprocess the raw return data for a specific ticker:
          - Plot ACF & PACF for returns to check autocorrelation.
          - Plot ACF & PACF for squared returns.
          - Perform Ljung-Box tests (for returns and squared returns).
          - Conduct ADF test for stationarity.
        """
        ret = self.returns_dict[ticker]
        #print(ret.head())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(ret, ax=axes[0], lags=20, title=f"ACF of Returns ({ticker})")
        plot_pacf(ret, ax=axes[1], lags=20, title=f"PACF of Returns ({ticker})")
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(ret**2, ax=axes[0], lags=20, title=f"ACF of Squared Returns ({ticker})")
        plot_pacf(ret**2, ax=axes[1], lags=20, title=f"PACF of Squared Returns ({ticker})")
        plt.tight_layout()
        plt.show()

        lb_ret = acorr_ljungbox(ret, lags=[10], return_df=True)
        lb_ret_sq = acorr_ljungbox(ret**2, lags=[10], return_df=True)
        print(f"Ljung-Box test for {ticker} returns (lag=10): p = {lb_ret['lb_pvalue'].values[0]:.4f}")
        print(f"Ljung-Box test for {ticker} squared returns (lag=10): p = {lb_ret_sq['lb_pvalue'].values[0]:.4f}")

        adf_stat, adf_p, _, _, _, _ = adfuller(ret)
        print(f"ADF test for {ticker} statistic: {adf_stat:.4f}, p-value = {adf_p:.4f}")

    def FHS(self, ticker):
      """
      Filtered Historical Simulation (FHS) for a specific ticker.
      返回:
        df_VaR: 包含 Date 索引和 VaR 列的 DataFrame
        failure_rate: 整个预测期间的失败率
        kupiec_p: Kupiec 检验的 p 值
      """
      returns = self.returns_dict[ticker].dropna().squeeze()
      window = self.window

      dates = []
      VaR_values = []

      for t in range(window, len(returns)):
          date_t = returns.index[t]
          window_returns = returns[t - window:t]

          model = arch_model(
              window_returns * 100,
              mean=self.mean,
              vol=self.vol,
              p=self.p,
              o=self.o,
              q=self.q,
              power=self.power,
              dist=self.dist
          )
          try:
              fit_result = model.fit(disp='off')
          except Exception as e:
              print(f"Window ending {date_t} optimization failed: {e}")
              continue

          # 预测下一期波动率
          forecast = fit_result.forecast(horizon=1)
          sigma_t_plus_1 = np.sqrt(forecast.variance.iloc[-1, 0])
          sigma_t = fit_result.conditional_volatility.iloc[-window:]
          standardized_returns = window_returns / sigma_t
          adjusted_returns = standardized_returns * sigma_t_plus_1

          # 计算 95% VaR（负号已包含）
          VaR = np.quantile(adjusted_returns, 0.05)

          dates.append(date_t)
          VaR_values.append(VaR)

      # 构建带日期索引的 DataFrame
      df_VaR = pd.DataFrame({'VaR': VaR_values}, index=pd.to_datetime(dates))
      df_VaR.index.name = 'Date'

      # 计算 failure_rate 和 Kupiec p-value
      actual_losses = returns.loc[df_VaR.index]
      exceptions = actual_losses < df_VaR['VaR']
      num_exceptions = exceptions.sum()
      total_observations = len(df_VaR)
      failure_rate = num_exceptions / total_observations

      confidence_level = 0.95
      p = 1 - confidence_level
      L0 = (1 - p) ** (total_observations - num_exceptions) * p ** num_exceptions
      L1 = (1 - failure_rate) ** (total_observations - num_exceptions) * failure_rate ** num_exceptions
      likelihood_ratio = -2 * (np.log(L0) - np.log(L1))
      kupiec_p = 1 - chi2.cdf(likelihood_ratio, df=1)

      # 保存结果
      self.var_dict[ticker] = df_VaR

      return df_VaR, failure_rate, kupiec_p

    def run_sliding_window(self, ticker):
        """
        Estimate the GARCH model using a sliding window approach for a specific ticker.
        """
        window = self.window
        returns = self.returns_dict[ticker]
        n = len(returns)
        dates = []
        omega_list = []
        alpha1_list = []
        beta1_list = []
        nu_list = []
        lam_list = []
        ks_p_list = []
        arch_p_list = []
        cond_vol_list = []

        for i in range(window, n):
            window_data = returns.iloc[i - window:i]
            model = arch_model(
                window_data * 100,
                mean=self.mean,
                vol=self.vol,
                p=self.p,
                o=self.o,
                q=self.q,
                power=self.power,
                dist=self.dist
            )
            try:
                res = model.fit(disp='off')
            except Exception as e:
                print(f"Window ending {returns.index[i]} optimization failed: {e}")
                continue

            dates.append(returns.index[i])
            omega_list.append(res.params.get('omega', np.nan))
            alpha1_list.append(res.params.get('alpha[1]', np.nan))
            beta1_list.append(res.params.get('beta[1]', np.nan))
            nu_val = res.params.get('eta', res.params.get('nu', np.nan))
            nu_list.append(nu_val)
            lam_val = res.params.get('lambda', np.nan)
            lam_list.append(lam_val)
            cv = res.conditional_volatility
            cond_vol_list.append(cv.iloc[-1])
            std_resid = res.resid / res.conditional_volatility
            dist_obj = res.model.distribution
            if dist_obj.num_params == 1:
                F_empirical = dist_obj.cdf(std_resid, [nu_val])
            elif dist_obj.num_params >= 2:
                F_empirical = dist_obj.cdf(std_resid, [nu_val, lam_val])
            else:
                F_empirical = None

            try:
                ks_stat, ks_p = stats.kstest(F_empirical, 'uniform', args=(0, 1))
            except Exception as e:
                ks_p = np.nan
            ks_p_list.append(ks_p)

            try:
                lb_test = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)
                arch_p = lb_test['lb_pvalue'].values[0]
            except Exception as e:
                arch_p = np.nan
            arch_p_list.append(arch_p)

            self.last_result_dict[ticker] = res

        self.results_dict[ticker] = pd.DataFrame({
            'omega': omega_list,
            'alpha1': alpha1_list,
            'beta1': beta1_list,
            'nu': nu_list,
            'lambda': lam_list,
            'KS_p': ks_p_list,
            'ARCH_p': arch_p_list
        }, index=dates)

        plt.figure(figsize=(6, 4))
        plt.plot(dates, cond_vol_list, label='Last σₜ of Each Window')
        plt.title(f'Last Conditional Volatility Over Time ({ticker})')
        plt.xlabel('Date')
        plt.ylabel('Conditional Volatility')
        plt.legend()
        plt.tight_layout()
        plt.show()

        if self.last_result_dict[ticker] is not None:
            print(f"\n===== Final Sliding Window Model Summary for {ticker} =====")
            print(self.last_result_dict[ticker].summary())
            print("Maximized Log Likelihood:", -self.last_result_dict[ticker].loglikelihood)
        return self.results_dict[ticker]

    def plot_last_qq(self, ticker):
        """
        Plot QQ-plot for the last sliding window of a specific ticker.
        """
        if ticker not in self.last_result_dict or self.last_result_dict[ticker] is None:
            print(f"No result from the last window for {ticker}. Please run run_sliding_window() first.")
            return

        res = self.last_result_dict[ticker]
        std_resid = res.resid / res.conditional_volatility
        dist_obj = res.model.distribution
        nu_val = res.params.get('eta', res.params.get('nu', np.nan))
        lam_val = res.params.get('lambda', np.nan)

        z_sorted = np.sort(std_resid)
        n_points = len(z_sorted)
        p_vals = (np.arange(n_points) + 0.5) / n_points

        if dist_obj.num_params == 1:
            q_theory = dist_obj.ppf(p_vals, [nu_val])
        elif dist_obj.num_params >= 2:
            q_theory = dist_obj.ppf(p_vals, [nu_val, lam_val])
        else:
            print(f"Unknown number of distribution parameters for {ticker}")
            return

        plt.figure(figsize=(6, 4))
        plt.scatter(q_theory, z_sorted, s=10, c='b', label='Empirical Quantiles')
        min_val = min(q_theory[0], z_sorted[0])
        max_val = max(q_theory[-1], z_sorted[-1])
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='45° line')
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Empirical Quantiles")
        plt.title(f"QQ-Plot of the Last Window ({ticker})")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Create a SlidingGARCH object with multiple tickers
    sliding_model = SlidingGARCH(
        tickers=["^GSPC", "GLD", "USO"],
        start="2020-01-01",
        end="2025-01-01",
        freq="1d",
        return_type="log",
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="skewt",
        window=250,
        power=2.0
    )

    # Preprocess data for each ticker
    for ticker in sliding_model.tickers + ["GLD-USO"]:
        print(f"\nPreprocessing for {ticker}:")
        sliding_model.preprocess(ticker)

    # Run sliding window estimation for each ticker and portfolio
    for ticker in sliding_model.tickers + ["GLD-USO"]:
        print(f"\nRunning sliding window for {ticker}:")
        results_df = sliding_model.run_sliding_window(ticker)
        print(f"Sliding window model parameters for {ticker}:")
        #print(results_df.head())

    tickers = sliding_model.tickers + ["GLD-USO"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    ax_ks, ax_arch = axes

    for ticker in tickers:
        df = sliding_model.results_dict[ticker]
        ax_ks.plot(df.index, df['KS_p'], label=ticker)
        ax_arch.plot(df.index, df['ARCH_p'], label=ticker)

    # 添加 5% 水平线
    ax_ks.axhline(0.05, color='red', linestyle='--', label='5% level')
    ax_arch.axhline(0.05, color='red', linestyle='--', label='5% level')

    # 设置标题和标签
    ax_ks.set_title("KS p-value Over Time")
    ax_arch.set_title("ARCH p-value Over Time")
    for ax in axes:
        ax.set_xlabel("Date")
        ax.set_ylabel("p-value")
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # Plot QQ-plot for each asset
    for ticker in sliding_model.tickers + ["GLD-USO"]:
        sliding_model.plot_last_qq(ticker)

    # —— 1) 初始化两个空 DataFrame ——
    all_var_df    = pd.DataFrame()  # 存放所有 ticker 的 VaR
    all_actual_df = pd.DataFrame()  # 存放所有 ticker 的实际收益

    # —— 2) 对每个 ticker 计算 FHS ——
    for ticker in sliding_model.tickers + ["GLD-USO"]:
        print(f"\nRunning FHS for {ticker}:")
        df_VaR, failure_rate, kupiec_p = sliding_model.FHS(ticker)
        print(df_VaR)
        print(f"Failure rate for {ticker}: {failure_rate:.4f}")
        print(f"Kupiec p-value for {ticker}: {kupiec_p:.4f}")

        # —— 2.1 合并 VaR ——
        all_var_df[ticker] = df_VaR['VaR']

        # —— 2.2 对齐并合并实际收益 ——
        returns = sliding_model.returns_dict[ticker].dropna().squeeze()
        actual  = returns.loc[df_VaR.index]        # 只取 VaR 对应的同一天
        all_actual_df[ticker] = actual

    # —— 3) 保存 VaR 到单独的文件 ——
    var_filename = f"{sliding_model.vol}-{sliding_model.mean}-{sliding_model.o}-VaR.csv"
    all_var_df.to_csv(var_filename, index_label='Date')
    print(f"Saved combined VaR to {var_filename}")

    # —— 4) 保存实际收益到另一个文件 ——
    actual_filename = f"{sliding_model.vol}-{sliding_model.mean}-{sliding_model.o}-actual.csv"
    all_actual_df.to_csv(actual_filename, index_label='Date')
    print(f"Saved combined actual returns to {actual_filename}")

    # Plot VaR vs Actual Returns for all assets
    plt.figure(figsize=(12, 6))

    for ticker in sliding_model.tickers + ["GLD-USO"]:
        # 取原始收益
        returns = sliding_model.returns_dict[ticker].dropna().squeeze()
        # 拿到该 ticker 的 df_VaR
        df_VaR = sliding_model.var_dict[ticker]
        # 对齐实际收益到 VaR 的日期
        actual = returns.loc[df_VaR.index]

        # 画实际收益
        plt.plot(df_VaR.index, actual, label=f"Actual Returns ({ticker})")
        # 画 95% VaR（注意 VaR 本身是正值，图上向下画负号）
        plt.plot(df_VaR.index, df_VaR["VaR"], label=f"95% VaR ({ticker})")

    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.title("95% VaR Forecast vs Actual Returns for All Assets")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

