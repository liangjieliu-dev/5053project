# iGarch
# mean=AR, o=1

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import chi2
#from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

from scipy.optimize import minimize

def igarch_neg_loglik(params, returns):
    mu, omega, alpha = params
    T = len(returns)
    sigma2 = np.empty(T)
    sigma2[0] = np.var(returns)
    resid = returns - mu
    loglik = 0.0
    for t in range(1, T):
        sigma2[t] = omega + alpha * resid[t-1]**2 + (1 - alpha) * sigma2[t-1]
        if sigma2[t] <= 0:
            return 1e6
        loglik += np.log(2 * np.pi) + np.log(sigma2[t]) + (resid[t]**2) / sigma2[t]
    return 0.5 * loglik

class SlidingGARCH:
    def __init__(self, tickers=["^GSPC", "GLD", "USO"], start="2024-01-01", end="2025-01-01",
                 freq="1d", return_type="log", mean="Zero", vol="GARCH",
                 p=1, o=None, q=1, dist="skewt", window=250, power=2.0):
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
      returns = self.returns_dict[ticker].dropna().squeeze()
      window = self.window

      dates = []
      VaR_values = []

      for t in range(window, len(returns)):
          date_t = returns.index[t]
          window_returns = returns[t - window:t]

          # —— 2.1 用自定义 IGARCH 拟合 ——
          ret_vals = (window_returns * 100).values
          mu0, omega0, alpha0 = np.mean(ret_vals), 1e-6, 0.1
          bounds = [(None, None), (1e-8, None), (1e-8, 0.9999)]
          res = minimize(igarch_neg_loglik, [mu0, omega0, alpha0],
                        args=(ret_vals,), bounds=bounds, method='L-BFGS-B')
          mu_est, omega_est, alpha_est = res.x
          beta_est = 1 - alpha_est

          # —— 2.2 递推计算条件方差序列 ——
          T = len(ret_vals)
          sigma2 = np.empty(T)
          sigma2[0] = np.var(ret_vals)
          resid = ret_vals - mu_est
          for t in range(1, T):
              sigma2[t] = omega_est + alpha_est * resid[t-1]**2 + beta_est * sigma2[t-1]

          # 用最后 window 点的 sigma 作为当期波动率
          # 预测下期波动率
          sigma_t_plus_1 = np.sqrt(omega_est
                                  + alpha_est * resid[-1]**2
                                  + beta_est * sigma2[-1])

          # 然后照原逻辑：
          standardized = window_returns / np.sqrt(sigma2)
          adjusted    = standardized * sigma_t_plus_1
          VaR = np.quantile(adjusted, 0.05)

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
      window = self.window
      returns = self.returns_dict[ticker]
      n = len(returns)

      dates = []
      omega_list = []
      alpha1_list = []
      beta1_list = []
      ks_p_list = []
      arch_p_list = []

      for i in range(window, n):
          window_data = returns.iloc[i - window:i]
          ret_vals = (window_data * 100).values

          # Initial parameters and bounds
          mu0 = ret_vals.mean()
          omega0 = 1e-6
          alpha0 = 0.1
          bounds = [(None, None), (1e-8, None), (1e-8, 0.9999)]

          # MLE fit for IGARCH(1,1)
          res = minimize(
              igarch_neg_loglik,
              x0=[mu0, omega0, alpha0],
              args=(ret_vals,),
              bounds=bounds,
              method='L-BFGS-B'
          )

          if not res.success:
              dates.append(returns.index[i])
              omega_list.append(np.nan)
              alpha1_list.append(np.nan)
              beta1_list.append(np.nan)
              ks_p_list.append(np.nan)
              arch_p_list.append(np.nan)
              continue

          mu_est, omega_est, alpha_est = res.x
          beta_est = 1.0 - alpha_est

          # Recursively compute sigma2 series
          T = len(ret_vals)
          sigma2 = np.empty(T)
          sigma2[0] = np.var(ret_vals)
          resid = ret_vals - mu_est
          for t in range(1, T):
              sigma2[t] = omega_est + alpha_est * resid[t - 1]**2 + beta_est * sigma2[t - 1]

          # Standardized residuals
          cond_vol = np.sqrt(sigma2)
          std_resid = resid / cond_vol

          # KS test on normal CDF
          F_emp = stats.norm.cdf(std_resid)
          try:
              _, ks_p = stats.kstest(F_emp, 'uniform', args=(0, 1))
          except:
              ks_p = np.nan
          ks_p_list.append(ks_p)

          # Ljung-Box test on squared standardized residuals
          try:
              lb = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)
              arch_p = lb['lb_pvalue'].iloc[0]
          except:
              arch_p = np.nan
          arch_p_list.append(arch_p)

          dates.append(returns.index[i])
          omega_list.append(omega_est)
          alpha1_list.append(alpha_est)
          beta1_list.append(beta_est)

      # Store final parameters
      self.last_result_dict[ticker] = {
          'mu': mu_est,
          'omega': omega_est,
          'alpha': alpha_est,
          'beta': beta_est
      }

      # Build results DataFrame
      self.results_dict[ticker] = pd.DataFrame({
          'omega':  omega_list,
          'alpha1': alpha1_list,
          'beta1':  beta1_list,
          'KS_p':   ks_p_list,
          'ARCH_p': arch_p_list
      }, index=pd.to_datetime(dates))

      # Print summary
      print(f"\n===== Final IGARCH(1,1) Summary for {ticker} =====")
      print(f"μ = {mu_est:.6f}, ω = {omega_est:.6e}, α = {alpha_est:.6f}, β = {beta_est:.6f}")

      return self.results_dict[ticker]

if __name__ == "__main__":
    # Create a SlidingGARCH object with multiple tickers
    sliding_model = SlidingGARCH(
        tickers=["^GSPC", "GLD", "USO"],
        start="2024-01-01",
        end="2025-01-01",
        freq="1d",
        return_type="log",
        mean="AR",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="t",
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
    #plt.show()
    plt.savefig(f"KS+ARCH-{sliding_model.vol}-{sliding_model.mean}-{sliding_model.o}.png")

    # # Plot QQ-plot for each asset
    # for ticker in sliding_model.tickers + ["GLD-USO"]:
    #     sliding_model.plot_last_qq(ticker)

    # —— 1) 初始化两个空 DataFrame ——
    all_var_df    = pd.DataFrame()  # 存放所有 ticker 的 VaR
    all_actual_df = pd.DataFrame()  # 存放所有 ticker 的实际收益

    # —— 2) 对每个 ticker 计算 FHS ——
    for ticker in sliding_model.tickers + ["GLD-USO"]:
        print(f"\nRunning FHS for {ticker}:")
        df_VaR, failure_rate, kupiec_p = sliding_model.FHS(ticker)
        # print(df_VaR)
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
    # actual_filename = f"{sliding_model.vol}-{sliding_model.mean}-{sliding_model.o}-actual.csv"
    # all_actual_df.to_csv(actual_filename, index_label='Date')
    # print(f"Saved combined actual returns to {actual_filename}")

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
    plt.savefig(f"VAR-{sliding_model.vol}-{sliding_model.mean}-{sliding_model.o}.png")
    #plt.show()