# Part 1. Task Discription

# Part 2. Install Python Packages

## 2.1. Install packages

* Yahoo Finance API
* pandas
* numpy
* matplotlib
* stockstats
* OpenAI gym
* stable-baselines
* tensorflow
* pyfolio

### 数据处理 & 分析相关

#### `yfinance`（Yahoo Finance API）

- **作用**：从 Yahoo Finance 下载股票/指数等金融数据。
- **在 FinRL 中用途**：
  - 下载训练/测试所需的历史数据（如股票价格、指数、VIX 等）

#### `pandas`

- **作用**：数据处理神器，用于表格（DataFrame）处理。
- **在 FinRL 中用途**：
  - 存储并处理股票数据、收益率、交易记录、账户价值等。

#### `matplotlib`

- **作用**：用于绘图。
- **在 FinRL 中用途**：
  - 可视化账户收益、回测结果等图表。

#### `stockstats`

- **作用**：为股票 DataFrame 添加各种技术指标（如 MACD、RSI、BOLL 等）。
- **在 FinRL 中用途**：
  - 自动计算并加入训练特征中的技术指标。

### 强化学习核心相关

#### `gym`（OpenAI Gym）

- **作用**：提供标准化的 RL 环境接口。
- **在 FinRL 中用途**：
  - 创建股票交易环境（`StockTradingEnv`），定义状态、动作、奖励。

#### `stable-baselines`（或 `stable-baselines3`）

- **作用**：一套强化学习算法的实现库（如 DDPG、PPO、A2C）。
- **在 FinRL 中用途**：
  - 用于训练智能体（Agent）进行股票买卖决策。

#### `tensorflow`

- **作用**：深度学习框架。
- **在 FinRL 中用途**：
  - 支撑底层的强化学习模型训练（如 DDPG 的 actor/critic 网络）。

> FinRL 有的版本可以选 `tensorflow` 或 `pytorch`，新版本偏向使用 `stable-baselines3` + `PyTorch`。

### 回测与评估相关

#### `pyfolio`

- **作用**：量化投资策略分析库。
- **在 FinRL 中用途**：
  - 分析模型的收益率、波动率、夏普比率等指标。
  - 生成完整的策略绩效报告（tear sheet）。

# Part 3. Download Data
Yahoo Finance provides stock data, financial news, financial reports, etc. Yahoo Finance is free.
* FinRL uses a class **YahooDownloader** in FinRL-Meta to fetch data via Yahoo Finance API
* Call Limit: Using the Public API (without authentication), you are limited to 2,000 requests per hour per IP (or up to a total of 48,000 requests a day).

雅虎财经提供股票数据、财经新闻、财经报道等。雅虎金融是免费的。

* FinRL使用FinRL Meta中的类**YahooDownloader**通过Yahoo Finance API获取数据
* 调用限制：使用公共API（无身份验证），每个IP每小时最多只能有2000个请求（或每天最多有48000个请求）。

```tex
YF.download() has changed argument auto_adjust default to True
YF deprecation warning: set proxy via new config function: yf.set_config(proxy=proxy)
[*********************100%***********************]  1 of 1 completed
结构信息：
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 3230 entries, 2009-01-02 to 2021-10-29
Data columns (total 5 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   (Close, AXP)   3230 non-null   float64
 1   (High, AXP)    3230 non-null   float64
 2   (Low, AXP)     3230 non-null   float64
 3   (Open, AXP)    3230 non-null   float64
 4   (Volume, AXP)  3230 non-null   int64  
dtypes: float64(4), int64(1)
memory usage: 151.4 KB

维度： (3230, 5)

列名： MultiIndex([( 'Close', 'AXP'),
            (  'High', 'AXP'),
            (   'Low', 'AXP'),
            (  'Open', 'AXP'),
            ('Volume', 'AXP')],
           names=['Price', 'Ticker'])

索引： DatetimeIndex(['2009-01-02', '2009-01-05', '2009-01-06', '2009-01-07',
               '2009-01-08', '2009-01-09', '2009-01-12', '2009-01-13',
...
1  2009-01-05  14.828887  15.632117  14.674419  15.408140  16019200  AXP    0
2  2009-01-06  15.678460  16.512585  15.454483  16.273161  13820200  AXP    1
3  2009-01-07  15.992727  16.140735  15.447431  15.587650  15699900  AXP    2
4  2009-01-08  15.424060  15.712288  15.112463  15.611019  12255100  AXP    3
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...

```

# Part 4: Preprocess Data

We need to check for missing data and do feature engineering to convert the data point into a state.

我们需要检查丢失的数据，并进行特征工程以将数据点转换为状态。

## 新增技术指标

* ***\*Adding technical indicators\****. In practical trading, various information needs to be taken into account, such as historical prices, current holding shares, technical indicators, etc. Here, we demonstrate two trend-following technical indicators: MACD and RSI.
* 新增技术指标。在实际交易中，需要考虑各种信息，如历史价格、当前持有股份、技术指标等。在这里，我们展示了两个趋势跟踪技术指标：MACD和RSI。

```python
fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = INDICATORS,
                    use_vix=True,
                    use_turbulence=True,
                    user_defined_feature = False)

processed = fe.preprocess_data(df)
```

```tex
结构信息：
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 3229 entries, 2009-01-02 to 2021-10-28
Data columns (total 5 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   (Close, ^VIX)   3229 non-null   float64
 1   (High, ^VIX)    3229 non-null   float64
 2   (Low, ^VIX)     3229 non-null   float64
 3   (Open, ^VIX)    3229 non-null   float64
 4   (Volume, ^VIX)  3229 non-null   int64  
dtypes: float64(4), int64(1)
memory usage: 151.4 KB

维度： (3229, 5)

列名： MultiIndex([( 'Close', '^VIX'),
            (  'High', '^VIX'),
            (   'Low', '^VIX'),
            (  'Open', '^VIX'),
            ('Volume', '^VIX')],
           names=['Price', 'Ticker'])

索引： DatetimeIndex(['2009-01-02', '2009-01-05', '2009-01-06', '2009-01-07',
               '2009-01-08', '2009-01-09', '2009-01-12', '2009-01-13',
               '2009-01-14', '2009-01-15',
               ...
               '2021-10-15', '2021-10-18', '2021-10-19', '2021-10-20',
               '2021-10-21', '2021-10-22', '2021-10-25', '2021-10-26',
               '2021-10-27', '2021-10-28'],
              dtype='datetime64[ns]', name='Date', length=3229, freq=None)

前几行数据：
Price           Close       High        Low       Open Volume
Ticker           ^VIX       ^VIX       ^VIX       ^VIX   ^VIX
Date                                                         
2009-01-02  39.189999  39.820000  36.880001  39.580002      0
2009-01-05  39.080002  40.220001  38.299999  39.240002      0
2009-01-06  38.560001  39.330002  37.340000  38.060001      0
2009-01-07  43.389999  43.820000  40.119999  40.290001      0
2009-01-08  42.560001  44.599998  42.560001  43.380001      0
Shape of DataFrame:  (3229, 8)
Successfully added vix
         date       open       high        low      close    volume  tic  day  \
0  2009-01-02  14.342317  15.076039  14.211020  14.929295  10955700  AXP    4   
1  2009-01-05  14.828887  15.632117  14.674419  15.408140  16019200  AXP    0   
2  2009-01-06  15.678460  16.512585  15.454483  16.273161  13820200  AXP    1   
3  2009-01-07  15.992727  16.140735  15.447431  15.587650  15699900  AXP    2   
4  2009-01-08  15.424060  15.712288  15.112463  15.611019  12255100  AXP    3   

       macd    boll_ub    boll_lb      rsi_30      cci_30       dx_30  \
0  0.000000        NaN        NaN         NaN         NaN         NaN   
1  0.010743  15.845907  14.491527  100.000000   66.666667  100.000000   
2  0.040513  16.899101  14.174629  100.000000  100.000000  100.000000   
3  0.028310  16.662981  14.436142   65.187520   40.792220   98.976368   
4  0.022013  16.527669  14.596038   65.609719    4.744625   58.989947   

   close_30_sma  close_60_sma        vix  
0     14.929295     14.929295  39.189999  
1     15.168717     15.168717  39.080002  
2     15.536865     15.536865  38.560001  
3     15.549562     15.549562  43.389999  
4     15.561853     15.561853  42.560001  
Successfully added turbulence index
```

**VIX** 是芝加哥期权交易所（CBOE）推出的 **波动率指数（Volatility Index）**，也被俗称为：

> 🧠 **“恐慌指数（Fear Index）”**

### 指标的意义

#### 第一部分：基础市场数据（行情数据）

| 列名     | 含义                                                         |
| -------- | ------------------------------------------------------------ |
| `date`   | 日期                                                         |
| `open`   | 当天开盘价                                                   |
| `high`   | 当天最高价                                                   |
| `low`    | 当天最低价                                                   |
| `close`  | 当天收盘价                                                   |
| `volume` | 当天成交量（交易的股数）                                     |
| `tic`    | 股票代码（如 AXP 表示 American Express）                     |
| `day`    | 一个辅助变量，表示这个日期是当前交易周期（如一周）内的第几天（0~4） |

#### 第二部分：技术指标（Technical Indicators）

这些是根据股价计算出的特征，帮助模型捕捉趋势、波动和动量等市场行为。

| 列名                 | 含义                                                         |
| -------------------- | ------------------------------------------------------------ |
| `macd`               | **移动平均收敛散度指标（Moving Average Convergence Divergence）**：用于识别价格趋势的方向与强度 |
| `boll_ub`, `boll_lb` | **布林带上下轨（Bollinger Bands Upper/Lower）**：用于衡量价格的波动性，价格突破上下轨可能意味着超买/超卖 |
| `rsi_30`             | **相对强弱指标（Relative Strength Index, RSI）**，30日版本：评估当前股价是否超买或超卖（0-100之间） |
| `cci_30`             | **顺势指标（Commodity Channel Index）**，30日版本：分析当前价格与平均水平的偏离程度 |
| `dx_30`              | **方向指标（Directional Movement Index）**：衡量趋势的强度（而非方向） |

#### 第三部分：移动平均线

| 列名           | 含义                                                         |
| -------------- | ------------------------------------------------------------ |
| `close_30_sma` | 过去30天的**收盘价简单移动平均**：用于平滑价格数据，观察趋势 |
| `close_60_sma` | 过去60天的收盘价简单移动平均                                 |

## Adding turbulence index

* ***\*Adding turbulence index\****. Risk-aversion reflects whether an investor prefers to protect the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the turbulence index that measures extreme fluctuation of asset price.
* 风险规避反映了投资者是否倾向于保护资本。当面对不同的市场波动水平时，它也会影响一个人的交易策略。为了控制最坏情况下的风险，如2007–2008年的金融危机，FinRL采用了衡量资产价格极端波动的动荡指数。

```python
list_ticker = processed["tic"].unique().tolist()
#提取所有的股票代码（tic），比如：["AAPL", "GOOG", "MSFT"]

list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
# 获取所有从最早到最晚的日期（完整的日历），比如：["2020-01-01", ..., "2021-12-31"]

combination = list(itertools.product(list_date,list_ticker))
# 生成所有日期和股票代码的笛卡尔积，也就是所有可能的 (日期, 股票) 组合。
"""
("2020-01-01", "AAPL")
("2020-01-01", "GOOG")
("2020-01-02", "AAPL")
("2020-01-02", "GOOG")
...
"""
processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
"""
创建一个完整的 dataframe，然后把原始 processed 数据 左连接进去。
有数据的地方就合并进来；
没有的地方就会变成 NaN。
"""

processed_full = processed_full[processed_full['date'].isin(processed['date'])]
# 只保留原始数据中真正存在的日期（过滤掉周末、节假日等你本来没有数据的日子）。
processed_full = processed_full.sort_values(['date','tic'])
# 按时间和股票代码排序，保证训练时顺序整齐。
processed_full = processed_full.fillna(0)
# 把缺失值（NaN）填成 0。

```

在原始的 `processed` 数据中，**某些时间点某些股票可能没有数据**（例如停牌、未上市等）。
 而强化学习训练通常需要一个完整的时间 × 股票的矩阵。

# Part 5. Build A Market Environment in OpenAI Gym-style
The training process involves observing stock price change, taking an action and reward's calculation. By interacting with the market environment, the agent will eventually derive a trading strategy that may maximize (expected) rewards.

Our market environment, based on OpenAI Gym, simulates stock markets with historical market data.

训练过程包括观察股价变化、采取行动和计算奖励。通过与市场环境的互动，Agent最终会得出一种可能使（预期）回报最大化的交易策略。
我们的市场环境基于OpenAI Gym，使用历史市场数据模拟股票市场。

## Data Split

We split the data into training set and testing set as follows:

Training data period: 2009-01-01 to 2020-07-01

Trading data period: 2020-07-01 to 2021-10-31

```python
train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)
print(len(train))
print(len(trade))
"""
2893
336
"""
```

```python
stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

# Stock Dimension: 1, State Space: 11
```

```python
buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension
"""
stock_dimension：股票数量，比如你模拟多少只股票。
buy_cost_list 和 sell_cost_list：每只股票买入和卖出的手续费比例，这里都设为0.1%（0.001）。
num_stock_shares：每只股票当前持有的股数，初始化为0。
"""

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}
"""
hmax：每次最大买卖多少股，限制为最多100股。
initial_amount：初始资金，100万元。
num_stock_shares：持仓股数列表，初始都为0。
buy_cost_pct / sell_cost_pct：买卖手续费比例列表。
state_space：环境的状态空间维度（代表状态向量大小，包含价格、技术指标等）。
stock_dim：股票数量。
tech_indicator_list：技术指标列表，用于辅助决策。
action_space：动作空间大小，一般是股票数量（每支股票都有买卖动作）。
reward_scaling：奖励缩放系数，强化学习用来调整奖励数值大小。
"""

e_train_gym = StockTradingEnv(df = train, **env_kwargs)
"""
StockTradingEnv 是一个类，定义了股票交易的环境规则。
df=train：传入训练集数据（行情、指标等）。
**env_kwargs：把前面定义的参数逐个传进去。
"""
```

## Environment for Training

**创建环境**

```python
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))
"""
e_train_gym.get_sb_env()
这是调用 StockTradingEnv 类实例 e_train_gym 的一个方法 get_sb_env()。
get_sb_env() 方法通常会返回一个与 Stable Baselines（一个强化学习库）兼容的环境对象。
返回值通常是一个元组，形如 (env_train, other_info)，所以用
"""
```

# Part 6: Train DRL Agents

\* The DRL algorithms are from **Stable Baselines 3**. Users are also encouraged to try **ElegantRL** and **Ray RLlib**.

\* FinRL includes fine-tuned standard DRL algorithms, such as DQN, DDPG, Multi-Agent DDPG, PPO, SAC, A2C and TD3. We also allow users to

design their own DRL algorithms by adapting these DRL algorithms.

* DRL算法来自**稳定基线3**。还鼓励用户尝试使用**ElegantRL**和**Ray RLlib**。

* FinRL包括经过微调的标准DRL算法，如DQN、DDPG、Multi-Agent DDPG、PPO、SAC、A2C和TD3。我们还允许用户通过对这些DRL算法的自适应，设计出自己的DRL算法。

```python
agent = DRLAgent(env = env_train)
# 创建agent

if_using_a2c = False
if_using_ddpg = False
if_using_ppo = False
if_using_td3 = False
if_using_sac = True
```

### Agent Training: 5 algorithms (A2C, DDPG, PPO, TD3, SAC)

### Agent 1: A2C

```python
agent = DRLAgent(env = env_train)
# 依赖环境 创建agent
model_a2c = agent.get_model("a2c")
# agent注入特定模型

if if_using_a2c:
  # set up logger
  tmp_path = RESULTS_DIR + '/a2c'
  new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_a2c.set_logger(new_logger_a2c)
# 配置日志
"""
如果 if_using_a2c 为 True：

在 RESULTS_DIR 目录下创建一个子目录 'a2c' 用于保存日志文件。

使用 configure() 函数（通常是 Stable Baselines3 中的一个日志配置函数）来配置日志记录方式，这里日志会输出到终端 (stdout)、保存为 CSV 文件(csv)，并且支持 TensorBoard 格式(tensorboard)。

将这个新的日志记录器绑定给 A2C 模型 model_a2c，这样训练过程中的日志信息就会记录到这些位置。

"""
trained_a2c = agent.train_model(model=model_a2c, 
                             tb_log_name='a2c',
                             total_timesteps=50000) if if_using_a2c else None
# trained_a2c 训练的模型
"""
如果 if_using_a2c 为 True, 就调用 agent.train_model() 方法，训练模型 model_a2c，训练步数为 50000，TensorBoard 日志名称为 'a2c'。
如果 if_using_a2c 为 False，则变量 trained_a2c 被赋值为 None。
agent.train_model() 应该是你代码里负责执行强化学习训练的函数。
"""

```

## In-sample Performance（样本内表现）

Assume that the initial capital is $1,000,000.

如果你用的是 2009 年到 2020 年的数据来训练你的策略，那么这段时间的策略表现就叫 **in-sample performance**。

### Set turbulence threshold

Set the turbulence threshold to be greater than the maximum of insample turbulence data. If current turbulence index is greater than the threshold, then we assume that the current market is volatile

**设置湍流阈值**

将湍流阈值设置为大于采样湍流数据的最大值。如果当前动荡指数大于阈值，则我们假设当前市场波动

```python
data_risk_indicator = processed_full[(processed_full.date<TRAIN_END_DATE) & (processed_full.date>=TRAIN_START_DATE)]
"""
从 processed_full 数据框中筛选出在训练时间区间 [TRAIN_START_DATE, TRAIN_END_DATE) 内的数据，赋值给 data_risk_indicator。
这一步限定了只看训练期内的所有数据。
"""

insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=['date'])
"""
对 data_risk_indicator 按照日期 date 去重，只保留每个日期的第一条记录，得到 insample_risk_indicator。
这样可以让每一天只对应一个风险指标值（假设同一天多个股票数据中，这两个指标是一样的，或者只需一条代表）。
"""

insample_risk_indicator.vix.describe()
"""
计算 vix 指标的统计信息，包括计数、均值、标准差、最小值、四分位数和最大值。
vix 是市场波动率指数，用来反映市场风险情绪。
"""

insample_risk_indicator.vix.quantile(0.996)
"""
计算 vix 的第 99.6 百分位数（即极高风险阈值）。
这是用来找到非常高风险波动率的临界点。
"""

insample_risk_indicator.turbulence.describe()
"""
计算 turbulence 指标的统计信息，类似于 vix 的描述。
turbulence 是用来度量市场异常波动的指标，反映市场的“动荡”程度。
"""

insample_risk_indicator.turbulence.quantile(0.996)
"""
计算 turbulence 指标的第 99.6 百分位数。
用来捕捉极端动荡风险的阈值。
"""
```

### Trading (Out-of-sample Performance样本外表现)

用来**测试模型泛化能力**的数据集上，模型的表现。
注意：这些数据**模型从未见过**！

We update periodically in order to take full advantage of the data, e.g., retrain quarterly, monthly or weekly. We also tune the parameters along the way, in this notebook we use the in-sample data from 2009-01 to 2020-07 to tune the parameters once, so there is some alpha decay here as the length of trade date extends. 

Numerous hyperparameters – e.g. the learning rate, the total number of samples to train on – influence the learning process and are usually determined by testing some variations.

我们定期更新以充分利用数据，例如，每季度、每月或每周重新训练。我们也会一路调整参数，在本笔记中，我们使用2009-01至2020-07的样本内数据调整参数一次，因此随着交易日期的延长，这里会出现一些alpha衰减。
很多超参数（如学习率、训练样本数）都会影响训练过程，通常是通过多次实验得出的。

>本 notebook 中只用一次固定的样本内数据进行了调参和训练，虽然在真实应用中我们会定期更新模型并调参，但这里为了简化流程没有这么做，所以策略效果可能随时间衰减（alpha decay），但这是一个常见的权衡。

| 阶段     | 数据范围    | 用途        | 类型          |
| -------- | ----------- | ----------- | ------------- |
| 训练阶段 | 2009 - 2020 | 训练 + 调参 | In-sample     |
| 测试阶段 | 2020 - 2023 | 模型验证    | Out-of-sample |

```python
e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
# env_trade, obs_trade = e_trade_gym.get_sb_env()
"""
StockTradingEnv 这个环境在回测过程中会参考某个风险指标列来判断当前市场是否“过于波动 / 危险”，从而决定是否让模型继续交易，或者减仓、清仓。
你这里传入的是 'vix'：
表示用 trade 这个 dataframe 里的 'vix' 列（VIX指数）作为风险指标。
如果某天 VIX 指数 高于 turbulence_threshold = 70，可能会触发风控逻辑，比如：
限制买入
减仓
清仓避险
risk_indicator_col='turbulence'
那就说明你想用 turbulence index（市场动荡指标） 来作为判断依据。
"""

trained_moedl = trained_sac
df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_moedl, 
    environment = e_trade_gym)
"""
用训练好的 DRL 模型（比如 A2C、PPO、SAC 等）在测试数据（e_trade_gym）上跑一遍。

得到两个结果：
df_account_value：每日账户总资产的记录（衡量模型表现）
df_actions：模型每天对每只股票的买卖决策（动作）
"""

df_account_value.shape
"""
看一下 df_account_value 这个 DataFrame 有多少行、多少列。
比如输出 (900, 2)，说明有 900 天的数据，2 列（通常是日期和资产值）。
"""

df_actions.head()
"""
看一下模型刚开始几天做出的交易决策。
df_actions 通常包含：
每只股票的买/卖/不动的动作值
有时还有时间戳等信息
"""
```

# Part 7: Backtesting Results
Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.

**回测（Backtesting）在量化交易中的作用：**

“回测”是在历史数据上模拟策略的表现**，用来评估一个交易策略是否有效。

**回测在评估交易策略的表现中起着关键作用。**
 我们更倾向于使用**自动化的回测工具**，因为它能**减少人为错误**。
 我们通常使用 **Quantopian 的 pyfolio 包** 来对交易策略进行回测。
 它使用起来很方便，而且提供了**多种图表**，可以**全面展示策略的表现**。

**pyfolio 的作用：**

- 一种 Python 的回测结果分析库；
- 能生成像 **累计收益曲线、夏普比率、最大回撤**、每月收益等图表；
- 和很多强化学习模型/策略库都能配合使用。

## 7.1 BackTestStats

`stats` 是 **statistics（统计数据）** 的缩写。在这段代码里，它用来存放**回测结果的各类绩效指标**。

pass in df_account_value, this information is stored in env class

“传入 `df_account_value`，这些信息已经保存在 `env`（环境）类中。”

**backtest_stats**

```python
print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=df_account_value)
"""
调用 backtest_stats() 函数来计算回测的绩效指标，比如：
年化收益率（Annual return）
夏普比率（Sharpe ratio）
最大回撤（Max drawdown）
胜率（Win rate）
累计收益（Cumulative return）
"""

perf_stats_all = pd.DataFrame(perf_stats_all)
# 把返回的回测指标变成一个 DataFrame 表格，方便后续查看和保存。

perf_stats_all.to_csv("./"+RESULTS_DIR+"/perf_stats_all_"+now+'.csv')
"""
保存回测结果到本地 CSV 文件，路径大概像这样：
./results/perf_stats_all_20250521-20h41.csv
"""
```
**baseline stats**

```python
#baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
        ticker="^DJI", 
        start = df_account_value.loc[0,'date'],
        end = df_account_value.loc[len(df_account_value)-1,'date'])
"""
get_baseline() 函数是用来抓取某个指数在指定时间段的价格数据。
ticker="^DJI"：表示选用 道琼斯指数（Dow Jones Industrial Average）作为基准；
start 和 end：用策略回测开始和结束的日期作为基准数据的时间范围，这两个日期来自 df_account_value：
"""

stats = backtest_stats(baseline_df, value_col_name = 'close')
"""
这一步是用 backtest_stats() 来计算基准指数的绩效指标，比如：
年化收益率
波动率
夏普比率
最大回撤
累计收益
这里设置 value_col_name = 'close'，表示使用基准数据中的 close 列作为账户价值参考。
"""

```

#### 1. **`backtest_stats`**

- 这是你对 **你自己的交易策略**（用 DRL 模型训练出来的策略）进行回测后得到的绩效统计数据。
- 通常基于：`df_account_value`（也就是 DRL 策略在交易过程中每一天的账户总资产）

#### 2. **`baseline stats`**

- 这是你选定的一个 **基准资产（benchmark）**，比如道琼斯指数 `^DJI`，的回测绩效。
- 是为了衡量你的策略表现是否“超过市场”。
- 通常基于：`baseline_df`，也就是你下载的基准指数的每日价格数据。

#### 两者比较的意义是：

| 目的               | 比较内容                          | 想了解什么？                     |
| ------------------ | --------------------------------- | -------------------------------- |
| 策略是否有“阿尔法” | 你的策略 vs. 基准指数             | 是否跑赢市场                     |
| 回报率谁更高       | Annual return / Cumulative return | 哪个长期收益更强                 |
| 风险调整后收益     | Sharpe ratio                      | 哪个在控制波动的同时获得更好收益 |
| 风险谁更大         | Max drawdown / Volatility         | 谁更稳定，谁的最大亏损更小       |

#### 举个实际比较的例子：

| 指标          | DRL 策略 | Baseline (^DJI) |
| ------------- | -------- | --------------- |
| Annual Return | 15.3%    | 9.8%            |
| Sharpe Ratio  | 1.45     | 0.88            |
| Max Drawdown  | -7.2%    | -15.6%          |
| Volatility    | 0.12     | 0.18            |

策略收益更高、回撤更小、夏普比率更高 → 策略优于市场。

如果反过来，说明策略并没有带来更好的表现，可能还要调参或换模型。

## 7.2 BackTestPlot

```python
print("==============Compare to DJIA===========")
%matplotlib inline

# 确保是 datetime 类型
df_account_value['date'] = pd.to_datetime(df_account_value['date'])
# 把账户净值数据里的日期列变成标准的 datetime 类型，方便之后画图和处理。

# 不要设置 index
# 手动传 baseline_start 和 baseline_end（确保是 datetime）
start_date = pd.to_datetime(df_account_value.loc[0, 'date'])
end_date = pd.to_datetime(df_account_value.loc[len(df_account_value)-1, 'date'])
# 把账户净值数据里的日期列变成标准的 datetime 类型，方便之后画图和处理。

# 调用 backtest_plot
backtest_plot(
    df_account_value,
    baseline_ticker='^DJI',
    baseline_start=start_date,
    baseline_end=end_date
)
"""
使用 FinRL 中封装的函数 backtest_plot，绘制对比图：
横轴：时间（从 start_date 到 end_date）
纵轴：账户价值（你策略 vs 道琼斯指数）
数据来源：
你策略的账户净值：df_account_value
道琼斯指数的收盘价（会自动下载）
"""

```

















