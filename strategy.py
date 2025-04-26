import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time  # 引入 time 模块用于可能的 API 调用间隔

# --- Tushare 初始化 ---
# !pip install tushare # 如果尚未安装，请取消注释并运行
import tushare as ts

# !!! 重要：请将 'YOUR_TUSHARE_TOKEN' 替换为你的 Tushare API Token !!!
TUSHARE_TOKEN = 'YOUR_TUSHARE_TOKEN'
if TUSHARE_TOKEN == 'YOUR_TUSHARE_TOKEN':
    print("警告：请在代码中设置你的 Tushare API Token！")
    pro = None
else:
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

# --- 数据获取函数 (使用 Tushare) ---

def get_csi800_constituents(date_str: str) -> list[str]:
    """
    使用 Tushare 获取指定日期的沪深800 (000906.SH) 成分股。
    """
    if pro is None:
        print("错误：Tushare API 未初始化。")
        return []
    try:
        tushare_date = pd.Timestamp(date_str).strftime('%Y%m%d')
        df = pro.index_weight(index_code='000906.SH', trade_date=tushare_date)
        if df is None or df.empty:
            print(f"警告：在 {tushare_date} 未获取到沪深800成分股数据，尝试前一个交易日。")
            prev_date = (pd.Timestamp(date_str) - pd.Timedelta(days=1)).strftime('%Y%m%d')
            df = pro.index_weight(index_code='000906.SH', trade_date=prev_date)

        if df is None or df.empty:
            print(f"错误：无法获取 {date_str} 或之前的沪深800成分股数据。")
            return []

        tickers = df['con_code'].unique().tolist()
        print(f"获取到 {len(tickers)} 支沪深800成分股 ({tushare_date} 或之前)。")
        return tickers
    except Exception as e:
        print(f"从 Tushare 获取成分股时出错: {e}")
        return []

def get_price_volume_data(tickers: list[str], start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """
    使用 Tushare 获取每日价格（收盘价）和成交量数据。
    返回一个带有 MultiIndex (日期, 股票代码) 的 pandas DataFrame。
    """
    if pro is None:
        print("错误：Tushare API 未初始化。")
        return pd.DataFrame()
    if not tickers:
        return pd.DataFrame()

    tushare_start_date = pd.Timestamp(start_date_str).strftime('%Y%m%d')
    tushare_end_date = pd.Timestamp(end_date_str).strftime('%Y%m%d')

    print(f"正在从 Tushare 获取 {len(tickers)} 支股票从 {tushare_start_date} 到 {tushare_end_date} 的日线数据...")
    try:
        ticker_str = ','.join(tickers)
        df = pro.daily(ts_code=ticker_str, start_date=tushare_start_date, end_date=tushare_end_date,
                       fields='ts_code,trade_date,close,vol')

        if df is None or df.empty:
            print("警告：从 Tushare 未获取到价格/成交量数据。")
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.rename(columns={'ts_code': 'ticker', 'vol': 'volume'})
        df = df[['date', 'ticker', 'close', 'volume']]
        df = df.set_index(['date', 'ticker'])
        df = df.sort_index()
        df['volume'] = df['volume'] * 100
        print(f"成功获取并处理了 {len(df)} 条价格/成交量数据。")
        return df

    except Exception as e:
        print(f"从 Tushare 获取价格/成交量数据时出错: {e}")
        return pd.DataFrame()

def get_earnings_data(tickers: list[str], report_date_str: str) -> pd.DataFrame:
    """
    使用 Tushare 获取在给定 report_date_str *之前* 公告的最新可用季度每股收益 (EPS)。
    还需要一年前同期的 EPS。
    返回一个以 'ticker' 为索引，包含 ['latest_eps', 'prior_year_eps'] 列的 DataFrame。
    """
    if pro is None:
        print("错误：Tushare API 未初始化。")
        return pd.DataFrame()
    if not tickers:
        return pd.DataFrame()

    target_ann_date = pd.Timestamp(report_date_str).strftime('%Y%m%d')
    results = {}
    print(f"正在从 Tushare 获取 {len(tickers)} 支股票截至 {target_ann_date} 的盈利数据...")

    ticker_count = 0
    total_tickers = len(tickers)
    for ticker in tickers:
        ticker_count += 1
        if ticker_count % 100 == 0:
            print(f"  处理盈利数据进度: {ticker_count}/{total_tickers}")
            time.sleep(0.5)

        try:
            start_date_eps = (pd.Timestamp(report_date_str) - pd.Timedelta(days=3*365)).strftime('%Y%m%d')
            df_fina = pro.fina_indicator(ts_code=ticker, start_date=start_date_eps, end_date=target_ann_date,
                                         fields='ts_code,ann_date,end_date,eps,update_flag')

            if df_fina is None or df_fina.empty:
                results[ticker] = {'latest_eps': np.nan, 'prior_year_eps': np.nan}
                continue

            df_fina = df_fina[df_fina['ann_date'] <= target_ann_date]
            if df_fina.empty:
                results[ticker] = {'latest_eps': np.nan, 'prior_year_eps': np.nan}
                continue

            latest_report = df_fina.sort_values(by='ann_date', ascending=False).iloc[0]
            latest_eps = latest_report['eps']
            latest_end_date = latest_report['end_date']

            year = int(latest_end_date[:4])
            month_day = latest_end_date[4:]
            prior_year_end_date = f"{year - 1}{month_day}"

            prior_report = df_fina[df_fina['end_date'] == prior_year_end_date]

            if not prior_report.empty:
                prior_report = prior_report.sort_values(by='ann_date', ascending=False).iloc[0]
                prior_year_eps = prior_report['eps']
            else:
                try:
                    df_prior = pro.fina_indicator(ts_code=ticker, period=prior_year_end_date,
                                                  fields='ts_code,ann_date,end_date,eps')
                    if df_prior is not None and not df_prior.empty:
                         df_prior = df_prior[df_prior['ann_date'] <= target_ann_date]
                         if not df_prior.empty:
                              prior_year_eps = df_prior.sort_values(by='ann_date', ascending=False).iloc[0]['eps']
                         else:
                              prior_year_eps = np.nan
                    else:
                         prior_year_eps = np.nan
                except Exception as e_prior:
                    print(f"查询 {ticker} 去年同期 ({prior_year_end_date}) EPS 时出错: {e_prior}")
                    prior_year_eps = np.nan

            results[ticker] = {'latest_eps': latest_eps, 'prior_year_eps': prior_year_eps}

        except Exception as e:
            print(f"从 Tushare 获取 {ticker} 盈利数据时出错: {e}")
            results[ticker] = {'latest_eps': np.nan, 'prior_year_eps': np.nan}
            time.sleep(1)

    print(f"完成盈利数据获取。成功处理 {len(results)}/{total_tickers} 支股票。")
    return pd.DataFrame.from_dict(results, orient='index')

def calculate_simplified_sue(earnings_df: pd.DataFrame) -> pd.Series:
    """
    根据季度 EPS 的同比变化计算简化的 SUE（标准化未预期盈余）。
    处理潜在的除以零或接近零的去年同期 EPS 的情况。
    返回一个以 ticker 为索引，包含 'sue' 值的 Series。
    """
    prior_eps_adjusted = earnings_df['prior_year_eps'].replace(0, 0.001)
    prior_eps_adjusted[np.abs(prior_eps_adjusted) < 0.001] = np.sign(prior_eps_adjusted) * 0.001

    sue = (earnings_df['latest_eps'] - earnings_df['prior_year_eps']) / np.abs(prior_eps_adjusted)
    return sue.rename('sue')

def calculate_cgo(price_volume_df: pd.DataFrame, current_date_str: str, lookback_days: int = 500) -> pd.Series:
    """
    计算资本利得悬置 (Capital Gain Overhang, CGO)。
    CGO = (当前价格 - 估计平均成本) / 当前价格
    估计平均成本通过回溯期内的成交量加权平均价 (VWAP) 近似。
    返回一个以 ticker 为索引，包含 'cgo' 值的 Series。
    """
    current_date = pd.Timestamp(current_date_str)
    start_date = current_date - pd.Timedelta(days=lookback_days * 1.5)

    price_volume_df = price_volume_df.sort_index()
    relevant_data = price_volume_df[price_volume_df.index.get_level_values('date') <= current_date]

    current_prices = relevant_data.loc[(current_date, slice(None)), 'close']
    if current_prices.empty:
        potential_dates = relevant_data.index.get_level_values('date').unique()
        if len(potential_dates) > 0:
            last_available_date = potential_dates.max()
            print(f"警告：{current_date_str} 没有价格数据。使用 {last_available_date} 的数据")
            current_prices = relevant_data.loc[(last_available_date, slice(None)), 'close']
        else:
             print(f"错误：找不到截至 {current_date_str} 的价格数据")
             return pd.Series(dtype=float, name='cgo')

    cgo_results = {}
    for ticker in current_prices.index:
        ticker_data = relevant_data.loc[(slice(None), ticker), ['close', 'volume']].droplevel('ticker')
        ticker_data = ticker_data.last(f'{lookback_days}D')

        if ticker_data.empty or ticker_data['volume'].sum() == 0 or ticker_data['close'].isna().all():
            cgo_results[ticker] = np.nan
            continue

        ticker_data = ticker_data.dropna()
        if ticker_data.empty:
             cgo_results[ticker] = np.nan
             continue

        weighted_price = (ticker_data['close'] * ticker_data['volume']).sum()
        total_volume = ticker_data['volume'].sum()

        if total_volume == 0:
             cgo_results[ticker] = np.nan
             continue

        estimated_avg_cost = weighted_price / total_volume
        current_price = current_prices.loc[ticker]

        if pd.isna(current_price) or current_price == 0:
            cgo_results[ticker] = np.nan
            continue

        cgo = (current_price - estimated_avg_cost) / current_price
        cgo_results[ticker] = cgo

    return pd.Series(cgo_results, name='cgo')

def get_quintiles(data: pd.Series) -> pd.Series:
    """将数据点分配到五分位数 (1=最低, 5=最高)。"""
    return pd.qcut(data, 5, labels=False, duplicates='drop') + 1

def run_strategy(rebalance_date_str: str, cgo_lookback_days: int = 500):
    """
    为给定的再平衡日期运行投资组合选择逻辑。
    """
    print(f"\n为再平衡日期运行策略：{rebalance_date_str}")

    tickers = get_csi800_constituents(rebalance_date_str)
    if not tickers:
        print("错误：无法获取股票池。")
        return [], []

    cgo_start_date = (pd.Timestamp(rebalance_date_str) - pd.Timedelta(days=cgo_lookback_days * 1.5)).strftime('%Y-%m-%d')
    price_volume_data = get_price_volume_data(tickers, cgo_start_date, rebalance_date_str)
    earnings_data = get_earnings_data(tickers, rebalance_date_str)

    sue = calculate_simplified_sue(earnings_data)
    cgo = calculate_cgo(price_volume_data, rebalance_date_str, cgo_lookback_days)

    factors_df = pd.DataFrame({'sue': sue, 'cgo': cgo})
    factors_df = factors_df.dropna()
    print(f"过滤缺失因子后的股票数量：{len(factors_df)}")
    if len(factors_df) < 10:
        print("错误：没有足够具有有效因子数据的股票来构建投资组合。")
        return [], []

    try:
        factors_df['sue_quintile'] = get_quintiles(factors_df['sue'])
        factors_df['cgo_quintile'] = get_quintiles(factors_df['cgo'])
    except Exception as e:
        print(f"计算五分位数时出错：{e}。请检查数据分布。")
        print("因子数据头部：\n", factors_df.head())
        print("因子数据描述：\n", factors_df.describe())
        return [], []

    long_portfolio = factors_df[
        (factors_df['sue_quintile'] == 5) & (factors_df['cgo_quintile'] == 5)
    ].index.tolist()

    short_portfolio = factors_df[
        (factors_df['sue_quintile'] == 1) & (factors_df['cgo_quintile'] == 1)
    ].index.tolist()

    print(f"多头组合规模：{len(long_portfolio)}")
    print(f"空头组合规模：{len(short_portfolio)}")

    return long_portfolio, short_portfolio

if __name__ == "__main__":
    rebalance_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    if pro is None:
        print("\n错误：Tushare API 未成功初始化。请检查你的 Tushare Token 设置。")
        print("脚本无法继续执行。")
    else:
        long_stocks, short_stocks = run_strategy(rebalance_date)

        print("\n--- 投资组合选择 (使用 Tushare 数据) ---")
        print(f"日期：{rebalance_date}")
        print("\n多头组合股票：")
        print(long_stocks if long_stocks else "无")
        print("\n空头组合股票：")
        print(short_stocks if short_stocks else "无")

        print("\n提醒：此脚本使用了简化的因子（SUE, CGO）和 Tushare 数据。")
        print("请根据需要验证数据质量，并考虑实现 RCGO 和完整的历史回测以进行实际分析。")
        print("注意 Tushare 积分和 API 调用频率限制。")
