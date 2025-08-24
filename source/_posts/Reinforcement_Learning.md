---
title: 《深度学习4-强化学习》学习笔记
cover: /imgs/Reinforcement_Learning.png
swiper_index: 3 #置顶轮播图顺序，非负整数，数字越大越靠前
categories: # 分类
	- 强化学习  # 只能由一个
tags:
    - 算法
    - 强化学习
    - 学习
mathjax: true
---
<meta name="referrer" content="no-referrer" />

# Chapter1：老虎机问题

## 一般问题

**问题描述：** 假如有*n*个老虎机，每个老虎机有一定的得分概率分布，如果你可以玩很多次，每次选择一个老虎机玩，如何做可以使得自己的得分最高？

**数学表示：**

*   奖励：$R$，即老虎机的返回值
*   行动：$a$，即对老虎机的选择
*   行动价值：$q(a)=\mathbb{E}[R|A]$，采取某个行动所得到奖励的期望值，$q(a)$表示真实的期望值，$Q(a)$表示估计的期望值

问题的本质是**对老虎机概率分布的估计**，即估计每台老虎机的$Q$值，简单来说是对老虎机输出的数学期望进行估计，根据**大数定律**，可以使用输出的**平均值**估计其数学期望，老虎机输出的平均值计算公式如下所示，随着n增加，其估计理论上也是越来越接近真实数学期望

$$
Q_n=\frac{R_1+R_2+R_3+...+R_n}{n}=Q_{n-1}+\frac{R_n-Q_{n-1}}{n}
$$

**Bandit类实现：**

*   `rate`代表输出概率列表，列表的索引是输出的值，例如\[0.2,0.4,0.4]，则其0.2的概率输出0，以此类推，`rand`控制概率列表的长度
*   `value`代表输出的**数学期望**
*   `play`函数表示玩一次老虎机，按照概率随机输出

```python
import numpy as np
np.random.seed(11)
class Bandit:
    def __init__(self):
        rand = np.random.rand(10)
        sum_rand = sum(rand)
        self.rate = [i/sum_rand for i in rand]
        self.value = sum([index*i for i,index in enumerate(self.rate)])

    def play(self):
        rand = np.random.rand()
        for index,value in enumerate(self.rate):
            rand-=value
            if rand<0:
                return index
```

**Agent类实现：**

*   `Q_list`表示对老虎机输出数学期望的估计
*   `n_list`记录玩老虎机的次数
*   `epsilon`表示探索的概率，如果每次都是选择Q值最高的老虎机，则会陷入局部最优。比如最开始Q值都是0，现在随机选择一个老虎机，其Q值会变的大于0，如果每次选择Q最高的老虎机，则会一直选择这个老虎机，导致无法获得最优解。

```python
class Agent:
    def __init__(self,epsilon,action_size):
        self.Q_list = np.zeros(action_size)
        self.n_list = np.zeros(action_size)
        self.epsilon = epsilon
    def update(self,action,reward):
        self.n_list[action]+=1
        self.Q_list[action]+=(reward-self.Q_list[action])/self.n_list[action]

    def decide_action(self):
        if np.random.rand()<self.epsilon:
            return np.random.randint(0,len(self.Q_list))
        else:
            return np.argmax(self.Q_list)
```

**实践操作：**

```python
import matplotlib.pyplot as plt
steps = 1000
num_of_bandit = 10
bandits = [Bandit() for i in range(num_of_bandit)]
bandits_values = [bandit.value for bandit in bandits]
best_choice = np.argmax(bandits_values)



agent = Agent(0.1,num_of_bandit)
win_rate = []
rewards = []

total_reward = 0


for n in range(steps):
    action = agent.decide_action()
    reward = bandits[action].play()
    agent.update(action,reward)
    total_reward+=reward
    rewards.append(total_reward)
    win_rate.append(total_reward/(n+1))

print(total_reward)
print(bandits_values)
plt.plot(win_rate)
plt.show()
```

理论上其得分的平均值会慢慢趋近于数学期望最高的老虎机的数学期望：

<!-- <p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d0c2f8c306ac444788b44c1db7bb8497~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=z3ywlzLtPcySv7gpK8Q%2B2XOHbz0%3D" alt="image.png" width="70%"></p> -->

<img src="https://gitee.com/leeMX111/reinforcement_learning_imgs/raw/master/202505051427942.png" height="400px" />

## 非稳态问题

上个例子中，老虎机的概率分布不会变化，属于**稳态问题**，而真实场景中，环境的状态会发生改变，也就是老虎机的概率分布会变化，这叫**非稳态问题**

对于非稳态问题，Q值的更新方式变化为：

$$
\begin{aligned} 
Q_n&=Q_{n-1}+\alpha(R_n-Q_{n-1})
\\
&= \alpha R_n+\alpha (1-\alpha)R_{n-1}+...\alpha (1-\alpha)^{n-1} R_1+\alpha (1-\alpha)^nQ_0
\end{aligned} 
$$

在稳态问题中，使用平均值对$Q_n$进行更新，体现为所有时刻的$R$对其权值都是一样的，都是$1/n$,而在非稳态问题中，我们希望最近的$R$对$Q_n$的影响更大，所以使用$\alpha$将之前的$R$的权重变小，这种权重指数级减少的计算，称为**指数移动平均**。

**NonStatBandit类实现：**

*   使用`change_rate`在每一次调用`play`时，给老虎机的概率分布增加噪声

```python
class NonStatBandit:
    def __init__(self):
        rand = np.random.rand(10)
        sum_rand = sum(rand)
        self.rate = [i/sum_rand for i in rand]
        self.value = sum([index*i for i,index in enumerate(self.rate)])

    def play(self):
        rand = np.random.rand()

        change_rate = np.random.randn(10)
        avg_change_rate = np.average(change_rate)
        change_rate = [0.1*(i-avg_change_rate)  for i in change_rate]
        self.rate = [self.rate[i]+change_rate[i] for i in range(len(self.rate))]


        for index,value in enumerate(self.rate):
            rand-=value
            if rand<0:
                return index
```

**AlphaAgent类实现：**

```python
class AlphaAgent:
    def __init__(self,epsilon,action_size,alpha):
        self.Q_list = np.zeros(action_size)
        self.epsilon = epsilon
        self.alpha = alpha
    def update(self,action,reward):
        self.Q_list[action]+=(reward-self.Q_list[action])*self.alpha

    def decide_action(self):
        if np.random.rand()<self.epsilon:
            return np.random.randint(0,len(self.Q_list))
        else:
            return np.argmax(self.Q_list)
```

**两种方法对比：**

这里每种方法都进行多次实验，次数为`runs`，每次实验的步数为`steps`,统计不同实验下，每一个`step`的收益平均值

```python
steps = 1000
runs = 200
all_rates = np.zeros((runs,steps))
for run in range(runs):
    steps = 1000
    num_of_bandit = 10
    # bandits = [Bandit() for i in range(num_of_bandit)]
    bandits = [NonStatBandit() for i in range(num_of_bandit)]
    agent = Agent(0.1,num_of_bandit)
    win_rate = []
    rewards = []
    total_reward = 0

    for n in range(steps):
        action = agent.decide_action()
        reward = bandits[action].play()
        agent.update(action,reward)
        total_reward+=reward
        rewards.append(total_reward)
        win_rate.append(total_reward/(n+1))

    all_rates[run] = win_rate

avg_rates = np.average(all_rates,axis=0)
print(avg_rates)
#
all_rates_nonstat = np.zeros((runs,steps))
for run in range(runs):
    steps = 1000
    num_of_bandit = 10
    bandits = [NonStatBandit() for i in range(num_of_bandit)]
    agent = AlphaAgent(0.1,num_of_bandit,0.8)
    win_rate = []
    rewards = []
    total_reward = 0

    for n in range(steps):
        action = agent.decide_action()
        reward = bandits[action].play()
        agent.update(action,reward)
        total_reward+=reward
        rewards.append(total_reward)
        win_rate.append(total_reward/(n+1))

    all_rates_nonstat[run] = win_rate

avg_rates_nonsta = np.average(all_rates_nonstat,axis=0)

plt.plot(avg_rates,label ='sample average')
plt.plot(avg_rates_nonsta,label ='alpha const update')
plt.legend()
plt.show()
```

可以看到，使用移动平均的效果更好：

<!-- <p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/7c4b05a9a7f04678b804c25f28a829c8~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=oKoeqmYndkmGhQr4L4v83sx%2BPCs%3D" alt="image.png" width="70%"></p> -->

<img src="https://gitee.com/leeMX111/reinforcement_learning_imgs/raw/master/202505051444520.png" height="400px" />

# Chapter2：马尔可夫决策过程

本章讨论的是**environment**随**Agent**的**Action**而发生变化的问题

## 数学表达

*   **状态迁移函数：** 给定当前状态$s$和行动$a$，输出下一个状态$s^{'}$的函数，既$s^{'}=f(s,a)$
*   **状态迁移概率：** 给定当前状态$s$和行动$a$，输出下一个状态$s^{'}$的概率，既$P(s^{'}|s,a)$，因为即使采取了行动，下个状态不是确定的，比如当确定向左移动，但是因为打滑之类的问题而没有发生移动
    *   状态迁移需要上一时刻的信息，既$s^{'}$只收到状态$s$和行动$a$的影响，这是**马尔可夫性**的体现
*   **奖励函数：** $r(s,a,s^{'})$，在下面的例子中，$r(s,a,s^{'})=r(s^{'})$
    *   奖励函数不是确定的，比如在$s^{'}$时，0.8的几率被攻击，奖励为-10等
*   **策略：** 只基于**当前状态**对下一步行动的确定，使用$\pi(a|s)$表示在$s$状态下采取$a$行动的概率

## MDP的目标

*   **回合制任务：** 有结束的任务，例如下围棋
*   **连续性任务：** 没有结束的任务，例如仓库存储

## 收益（return）

收益是指在给定策略时，某个状态下，未来时间内的奖励的总和，公式表示为：

$$
G_t=R_t+\gamma R_{t+1} + \gamma ^2 R_{t+2}+\gamma ^3 R_{t+3}...
$$

$\gamma$被称为`折现率（discount rate）`，具体原因有两个：

*   防止在连续性任务重，收益无限大的情况
*   收益实质上是对未来收益的一种预测，显然距离$t$越远的预测是越不可信的，且这种设置使得收益更关注当前的收益，这是一种**贪心**的体现

策略或状态变化在一些问题中是随机的，所以我们需要使用**数学期望**描述收益，称为**状态价值函数**：

$$
\nu _{\pi}(s) = \mathbb{E}_{\pi}[G_t|S_t = s]
$$

**策略**$\pi$和**初始状态**$s$是需要给定的，然后计算这个条件下未来的收益的数学期望，显然我们需要找到使得这个期望最大的策略$\pi$

**最优策略：** 如果策略$\pi ^{'}$在任意的$s$下，$\nu _{\pi^{'}}(s)$都是最大的，那么$\pi ^{'}$为最优策略。**可以证明在MDP问题中，一定存在一个最优策略**

这里举一个例子，只存在两个状态，当吃到苹果时+1，苹果会无限刷新，撞到墙-1：

<!-- <p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/08174fc2ba604fd38c05daa43a963df1~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=ZWLLSjIDJqlTbovhQKvJVG%2FYoGY%3D" alt="Screenshot_20241202_144609_com.newskyer.draw.jpg" width="50%"></p> -->

<img src="https://gitee.com/leeMX111/reinforcement_learning_imgs/raw/master/202505051807716.png" height="260px" />


策略一共有四种，值得注意的是，**这个例子中状态转移和策略都是确定的**：

|            | $s=L1$ | $s=L2$ |
| ---------- | ------ | ------ |
| $\pi_1(s)$ | Right  | Right  |
| $\pi_2(s)$ | Right  | Left   |
| $\pi_3(s)$ | Left   | Right  |
| $\pi_4(s)$ | Left   | Left   |

随便算一个,假设$\gamma=0.9$

$$
\nu_{\pi_1}(s=L1) = 1+0.9×(-1)+0.9^2×(-1)...=-8
$$

计算所有策略，很显然$\pi_2(s)$是最优策略

# Chapter3：贝尔曼方程

上面的例子中，状态转移和策略都是确定的，当面临随机性时，我们不能使用上面的方法计算，而需要使用**贝尔曼方程**。

## 贝尔曼方程

回忆收益公式：

$$
G_t=R_t+\gamma R_{t+1} + \gamma ^2 R_{t+2}+\gamma ^3 R_{t+3}... = R_t +\gamma G_{t+1}
$$

那么状态价值函数可以写为：

$$
\nu _{\pi}(s) = \mathbb{E}_{\pi}[G_t|S_t = s] = \mathbb{E}_{\pi}[R_t|S_t = s]+\gamma \mathbb{E}_{\pi}[G_{t+1}|S_t = s]
$$

其中

$$
\mathbb{E}_{\pi}[R_t|S_t = s] = \sum_{a} \sum_{s^{'}} \pi(a|s)p(s^{'}|s,a)r(s,a,s^{'})
\\
\mathbb{E}_{\pi}[G_{t+1}|S_t = s] = \sum_{a} \sum_{s^{'}}\pi(a|s)p(s^{'}|s,a)\mathbb{E}_{\pi}[G_{t+1}|S_{t+1} = s]
$$

*   第一个部分：本质上就是求出每一种可能性的概率及其奖励，然后加和
*   第二个部分：确定初始时刻状态，下一时刻开始的收益，应该等于确认下一时刻状态，下一时刻收益×初始时刻到下一时刻状态的概率2

那么**贝尔曼方程**为：

$$
\begin{aligned} 
\nu _{\pi}(s) &= \mathbb{E}_{\pi}[R_t|S_t = s]+\gamma \mathbb{E}_{\pi}[G_{t+1}|S_t = s]
\\&=\sum_{a} \sum_{s^{'}} \pi(a|s)p(s^{'}|s,a)r(s,a,s^{'})+\gamma \sum_{a} \sum_{s^{'}}\pi(a|s)p(s^{'}|s,a)\nu _{\pi}(s^{'})
\\&=\sum_{a} \sum_{s^{'}}\pi(a|s)p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{\pi}s^{'}\}
\end{aligned}
$$

**贝尔曼方程表示的是“当前状态$s$的价值函数$\nu_{\pi}(s)$”和“下一状态$s^{'}$的价值函数$\nu_{\pi}(s^{'})$”之间的关系，其最大的意义在于将无限的计算转换为有限的联立方程。**

对于上面那个例子，假设策略为0.5概率向左移动，0.5向右移动，采用贝尔曼方程求解，这里由于状态转移是固定的，那么当$s^{'}=f(s,a)$时，贝尔曼方程可以写为：

$$
\nu _{\pi}(s)=\sum_{a} \pi(a|s)\{r(s,a,s^{'})+\gamma \nu _{\pi}(s^{'})\}
$$

这里行动只有往左或者往右，那么其还可写成：

$$
\begin{aligned} 
\nu _{\pi}(s)=&\pi(a=Left|s)\{r(s,a=Left,s^{'})+\gamma \nu _{\pi}(s^{'}\}
\\&+\pi(a=Right|s)\{r(s,a=Right,s^{'})+\gamma \nu _{\pi}(s^{'})\}
\end{aligned}
$$

那么：

$$
\nu _{\pi}(L1)=0.5\{-1+0.9\nu _{\pi}(L1)\}+0.5\{1+0.9\nu _{\pi}(L2)\}
\\
\nu _{\pi}(L2)=0.5\{0+0.9\nu _{\pi}(L1)\}+0.5\{-1+0.9\nu _{\pi}(L2)\}
$$

求解可得，$\nu _{\pi}(L1)=-2.25, \nu _{\pi}(L2)=-2.75$

## 行动价值函数（Q函数）

状态价值函数需要两个条件，即**策略**和**状态**，在这个基础上再考虑一个条件，也就是第一次的**行动$a$**，就构成了**Q函数：**

$$
q_{\pi}(s,a) = \mathbb{E}_{\pi}[G_{t+1}|S_t = s,A_t = a] 
$$

Q函数中行动的选择不受到策略的影响，是自己选择的，后面的行动就是按照策略来了，那么显然Q函数和状态价值函数有如下关系：

$$
\nu _{\pi}(s) = \sum_{a}\pi(a|s)q_{\pi}(s,a)
$$

按照同样的分析，可以获得Q函数的表达式：

$$
\begin{aligned} 
q_{\pi}(s,a)&=\mathbb{E}_{\pi}[R_t|S_t = s,A_t = a]+\gamma \mathbb{E}_{\pi}[G_{t+1}|S_t = s,A_t = a]
\\&=\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{\pi}(s^{'})\}
\end{aligned}
$$

那么，**使用Q函数的贝尔曼方程**为：

$$
q_{\pi}(s,a)=\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \sum_{a^{'}}\pi(a^{'}|s^{'})q_{\pi}(s^{'},a^{'})\}
$$

## 贝尔曼最优方程

回忆贝尔曼方程：

$$
\begin{aligned} 
\nu _{\pi}(s) =\sum_{a} \pi(a|s)\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{\pi}(s^{'})\}
\end{aligned}
$$

最优策略应该使得$\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{\pi}(s^{'}\}$最大（因为最优策略是确定的，那么$\pi(a|s)$部分可以省略），于是**贝尔曼最优方程为：**

$$
\begin{aligned} 
\nu _{*}(s) =\max_a \sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{*}(s^{'})\}
\end{aligned}
$$

同理，**Q函数的贝尔曼最优方程**为：

$$

q_{*}(s,a)=\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \max_{a^{'}} q_{*}(s^{'},a^{'})\}

$$

**最优策略**应该是可以使得价值函数最大的策略，用数学语言描述就是：

$$
\begin{aligned}
\pi_{*}(s) &= \argmax_{a}\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{*}(s^{'})\}
\\&=\argmax_{a}q_{*}(s,a)
\end{aligned}
$$

**最优收益的获取：**

还是之前的例子，由于状态转移是确定的，那么最优贝尔曼方程可以写成：

$$
\begin{aligned} 
\nu _{*}(s) =\max_a\{r(s,a,s^{'})+\gamma \nu _{*}(s^{'})\}
\end{aligned}
$$

于是：

$$
\nu_{*}(L1)=max\{-1+0.9\nu_{*}(L1),1+0.9\nu_{*}(L2)\}\\
\nu_{*}(L2)=max\{-1+0.9\nu_{*}(L2),1+0.9\nu_{*}(L1)\}
$$

求解可得，$\nu_{*}(L1)=5.26,\nu_{*}(L2)=4.73$

**最优策略的获取：**

最优策略$\pi_{*}(s)$表示为：

$$
\pi_{*}(s) = \argmax_{a}(\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{\pi}(s^{'})\})
$$

在这个例子中，当$s=L1$时，如果其行动是往左走，则：

$$
\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{\pi}(s^{'})\}=-1+0.9×5.26=3.734
$$

如果往右走，则：

$$
\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{\pi}(s^{'})\}=1+0.9×4.73=5.257
$$

显然，$\pi_{*}(L1)=Right$，同理$\pi_{*}(L2)=Left$

## 总计

**【贝尔曼方程】**

$$
\begin{aligned} 
&\nu _{\pi}(s) =\sum_{a} \sum_{s^{'}}\pi(a|s)p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{\pi}(s^{'}\}
\\&q_{\pi}(s,a)=\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \sum_{a^{'}}\pi(a^{'}|s^{'})q_{\pi}(s^{'},a^{'})\}
\end{aligned}
$$

**【贝尔曼最优方程】**

$$
\begin{aligned} 
&\nu _{*}(s) =\max_a \sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{*}(s^{'})\}
\\&q_{*}(s,a)=\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \max_{a^{'}} q_{*}(s^{'},a^{'})\}
\end{aligned}

$$

**【最优策略】**

$$
\begin{aligned}
\pi_{*}(s) &= \argmax_{a}\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{*}(s^{'})\}
\\&=\argmax_{a}q_{*}(s,a)
\end{aligned}
$$

# Chapter4：动态规划法

按照上面的做法，得到最优贝尔曼方程的解需要解一个联立方程，在上述的例子中环境只有两个，所有这个联立方程可求，当环境变复杂，问题变得多变，解方程的计算量**指数级上升**，所以我们需要用**动态规划**的方法计算**最优贝尔曼方程**或**最优策略**。

在强化学习中通常设计两项任务：

*   **策略评估：** 求给定策略$\pi$的价值函数$\nu _{\pi}(s)$
*   **策略控制：** 控制策略并将其调整为最优策略

## 迭代策略评估

回忆贝尔曼方程：

$$
\begin{aligned} 
\nu _{\pi}(s) =\sum_{a} \pi(a|s)\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{\pi}(s^{'})\}
\end{aligned}
$$

使用DP进行策略评估的思路是从贝尔曼方程衍生出来的，思路是将贝尔曼方程变形为“**更新式**”，表达为：

$$
V_{k+1}(s)=\sum_{a} \sum_{s^{'}}\pi(a|s)p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma V _{k}(s^{'})\}
$$

主要特点是用**下一个状态的当前迭代的价值函数**$V _{k}(s^{'})$来更新**当前状态的下一个迭代的价值函数**$V_{k+1}(s)$

例如对于上面那个例子，一开始的初始价值函数为$V_0(L1)$和$V_0(L2)$，这个$0$代表第0次迭代，然后用上面那个公式计算$V_1(L1)$和$V_2(L2)$，以此类推。

> **如何证明这个迭代式的有效性？(来自豆包)**
> <img src="https://gitee.com/leeMX111/reinforcement_learning_imgs/raw/master/202505052042849.png" height="1200px" />

在上面的例子中，不存在状态转移概率，所有可以简写为：

$$
V_{k+1}(s)=\sum_{a}\pi(a|s)\{r(s,a,s^{'})+\gamma V _{k}(s^{'})\}
$$

对上面的例子进行迭代评估实验：

*   这里使用`delta`表征上下两次迭代的改变量，当改变量小于阈值时，停止迭代

```python
V = {'l1':0,'l2':0}
new_V = V.copy()
n = 1
while(1):
    new_V['l1'] = 0.5*(-1+0.9*V['l1'])+0.5*(1+0.9*V['l2'])
    new_V['l2'] = 0.5 * (0 + 0.9 * V['l1']) + 0.5 * (-1 + 0.9 * V['l2'])
    delta = max(abs(new_V['l1']-V['l1']),abs(new_V['l2']-V['l2']))
    V = new_V.copy()
    n+=1
    if delta<0.0001:
        print(n)
        break

print(V)
# 77
# {'l1': -2.249167525908671, 'l2': -2.749167525908671}
```

## 迭代策略评估的其他方式-覆盖方式

*   **上述方式**：上面的方法是全部计算出新迭代的所有状态的价值函数，再进行下一轮迭代
*   **覆盖方式**：在计算出新的某个状态的价值函数后，直接将其用于计算其他状态的价值函数。
    *   例如我们先利用$V_0(L1)$和$V_0(L2)$计算出了$V_1(L1)$，然后利用$V_1(L1)$和$V_0(L2)$计算$V_1(L2)$，以此类推。

```python
V = {'l1':0,'l2':0}
n = 1
while(1):
    t = 0.5*(-1+0.9*V['l1'])+0.5*(1+0.9*V['l2'])
    delta = abs(t-V['l1'])
    V['l1'] = t

    t = 0.5 * (0 + 0.9 * V['l1']) + 0.5 * (-1 + 0.9 * V['l2'])
    delta = max(abs(t - V['l2']),delta)
    V['l2']=t

    n+=1
    if delta<0.0001:
        print(n)
        break

print(V)
# 61
# {'l1': -2.2493782177156936, 'l2': -2.7494201578106514}
```

## GridWorld

在之前的例子中，我们使用了一个只有两个状态的问题，在这一章中使用一个更为复杂的场景，包含一个奖励点，一个惩罚点和一个无法移动点：
 
<img src="https://gitee.com/leeMX111/reinforcement_learning_imgs/raw/master/202508241405517.png" height="300px" />

```python
class GridWorld:  
    def __init__(self):
        self.RewardMap = [[0,0,0,1],
                          [0,None,0,-1],
                          [0,0,0,0]]
        self.Actions = {'up':[-1,0],
                       'down':[1,0],
                       'left':[0,-1],
                       'right':[0,1]}
        self.GoalState = [0,3]
        self.StartState = [2,0]
        self.WallState = [1,1]

        self.Width = len(self.RewardMap[0])
        self.Height = len(self.RewardMap)

    def NextState(self,State,Action):
        Move = self.Actions[Action]
        NextState = [State[0] + Move[0],State[1] + Move[1]]
        if NextState[0]<0 or NextState[1]<0 or NextState[0]>=self.Height or NextState[1]>=self.Width:
            NextState = State
        elif NextState == self.WallState:
            NextState = State

        return NextState

    def GetReward(self,NextState):
        return self.RewardMap[NextState[0]][NextState[1]]

```

下面使用**迭代策略评估**来估计在**随机策略**下的价值函数值，回忆一次价值函数的迭代公式：

$$
V_{k+1}(s)=\sum_{a}\pi(a|s)\{r(s,a,s^{'})+\gamma V _{k}(s^{'})\}
$$

```python
def DPOneStep(pi,V,env,gamma=0.9):
    for Row in range(env.Height):
        for Col in range(env.Width):
            State = (Row,Col)
            if State == env.GoalState:
                V[State] = 0
                continue

            NewV = 0
            ActionProbs = pi[State]
            for Action , ActionProb in ActionProbs.items():
                NextState = env.NextState(State,Action)
                Reward = env.GetReward(NextState)
                NewV += ActionProb*(Reward+gamma*V[NextState])

            V[State] = NewV
    return V

def PolicyEval(pi,V,env,gamma=0.9,threshold=0.0001):
    while(1):
        OldV = V.copy()
        V = DPOneStep(pi,V,env,gamma=gamma)

        Delta = 0
        for Key,Value in V.items():
            T = abs(Value-OldV[Key])
            if T>Delta:
                Delta=T

        if Delta<threshold:
            break
    return V
```

<p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/ac17ce599ca046c084abae43371429fc~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=hQTVnSVrEew%2FqXp4nnkhmmKSh0k%3D" alt="image.png" width="70%"></p>

## 策略迭代法

最优策略$\pi_{*}(s)$表示为：

$$
\begin{aligned}
\pi_{*}(s) &= \argmax_{a}\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{*}(s^{'})\}
\\&=\argmax_{a}q_{*}(s,a)
\end{aligned}
$$

同理，也可以以此为基础提出最优策略的迭代公式：

$$
\begin{aligned}
\pi^{'}(s) &= \argmax_{a}\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{\pi}(s^{'})\}
\end{aligned}
$$

其中$\nu _{\pi}$ 表示当前策略下的价值函数，$\pi^{'}(s)$表示下一次迭代的策略。

> 至此，我们得到了两个迭代公式：
>
> *   **价值函数迭代公式：** 给定**策略**即可求出当前策略的**价值函数**
> *   **最优策略迭代公式：** 给定**价值函数**即可迭代更新**策略**
>
> 可以观察到上述两个公式是互相运用的，基于这个前提，策略迭代的基本思路是**重复评估和改进**，具体的说是用$\pi_0$评估得到$V_0$，然后再用$V_0$更新策略得到$\pi_1$，以此类推。

**实施策略迭代法：**

在上面的例子中不存在状态转移概率，所以策略的迭代公式为：

$$
\begin{aligned}
\pi^{'}(s) &= \argmax_{a}\{r(s,a,s^{'})+\gamma \nu _{\pi}(s^{'})\}
\end{aligned}
$$

*   `GreedyPolicy`函数实现了**最优策略迭代公式**，主要思路是计算出每个$Action$的$\{r(s,a,s^{'})+\gamma \nu _{\pi}(s^{'})\}$，然后存储到`ActionScore`中，最后得到最大值的$Action$
*   `PolicyIter`函数实现了**重复评估和改进**，迭代结束的标准是策略收敛，当前策略即是**最优策略**。

```python
import numpy as np
from collections import defaultdict
def GreedyPolicy(env,V,gamma = 0.9):
    pi = {}
    for Row in range(env.Height):
        for Col in range(env.Width):
            State = [Row,Col]
            ActionScore = {}
            for Action,StateChange in env.Actions.items():
                NextState = env.NextState(State,Action)
                Reward = env.GetReward(NextState)
                ActionScore[str(Action)]=Reward+gamma*V[str(NextState)]

            MaxScore = -1 #确保BestAction有值
            for Action,Score in ActionScore.items():
                if Score>MaxScore:
                    MaxScore = Score
                    BestAction = Action
            pi[str(State)] = {'left':0,'right':0,'up':0,'down':0}
            pi[str(State)][BestAction] = 1
    return pi

def PolicyIter(pi,V,env,gamma=0.9,threshold=0.001):
    while(1):
        V = PolicyEval(pi,V,env,gamma, threshold)
        NewPi = GreedyPolicy(env,V,gamma)

        if NewPi==pi:
            break
        pi = NewPi
        # print(V,pi)
    return pi,V




env = GridWorld()

V = defaultdict(lambda :0)
pi = defaultdict(lambda :{'left':0.25,'right':0.25,'up':0.25,'down':0.25})
V = PolicyEval(pi, V, env)

BestPi,BestV = PolicyIter(pi,V,env)
print(BestPi)
print(BestV)

for State,Action in BestPi.items():
    print(State,Action)
```

## 价值迭代法

观察两个迭代公式：

$$
V^{'}(s)=\sum_{a} \sum_{s^{'}}\pi(a|s)p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma V(s^{'})\}
$$

$$
\begin{aligned}
\pi^{'}(s) &= \argmax_{a}\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{\pi}(s^{'})\}
\end{aligned}
$$

观察到$\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{\pi}(s^{'})\}$部分的计算是**重复的**，可以合并为：

$$
V^{'}(s)=\max_{a}\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma V(s^{'})\}
$$

回忆贝尔曼最优方程：

$$
\begin{aligned} 
\nu _{*}(s) =\max_a \sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{*}(s^{'})\}
\end{aligned}
$$

可以看到**上面的合并公式就是贝尔曼最优方程的迭代形式**

**价值迭代法的实现：**

*   `ValueIterOneStep`函数实现了上述迭代公式，将$\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma V(s^{'})\}$存储到`ActionScores`，选出最大的值即可

```python
def ValueIterOneStep(V,env,gamma = 0.9):
    for Row in range(env.Height):
        for Col in range(env.Width):
            State = (Row,Col)
            if State == env.GoalState:
                V[State] = 0
                continue

            ActionScores = []
            for Action,StateChange in env.Actions.items():
                NextState = env.NextState(State,Action)
                Reward = env.GetReward(NextState)
                ActionScores.append(Reward+gamma*V[NextState])

            V[State] = max(ActionScores)
    return V
def ValueIter(V,env,gamma=0.9,threshold=0.001):
    while(1):
        OldV=V.copy()
        V = ValueIterOneStep(V,env,gamma)

        Delta = 0
        for Key,Value in V.items():
            T = abs(Value-OldV[Key])
            if T>Delta:
                Delta=T

        if Delta<threshold:
            break
    return V

env = GridWorld()

V = defaultdict(lambda :0)
pi = defaultdict(lambda :{'left':0.25,'right':0.25,'up':0.25,'down':0.25})
V = PolicyEval(pi, V, env)

V = ValueIter(V,env)
pi = GreedyPolicy(env,V)
```

<img src="https://gitee.com/leeMX111/reinforcement_learning_imgs/raw/master/202508241359358.png" height="400px" />
## 总结

动态规划法本质上就是利用**贝尔曼方程的迭代形式**，其要求是**已知环境模型**，一步一步迭代而避免了求解复杂的联立方程，具体分为三个方法：

*   **策略评估**：已知策略迭代计算价值函数
*   **策略迭代法**：策略和价值函数互相迭代
*   **价值迭代法**：先迭代计算出最优价值函数，然后再直接得到最优策略

# Chapter5：蒙特卡洛方法

大部分情况下环境模型是位置的，我们必须从环境中的反复的采集来完成我们的估计，**蒙特卡洛方法是对数据进行反复采样并根据结果进行估计的方法的总称。**

模型的表示方法分为**分布模型**和**样本模型**

*   **分布模型**：表示出模型的具体概率分布
*   **样本模型**：通过采样样本以估计模型的概率分布

例如对于2个骰子的点数和模型，我们当然可以一一列举可能出现的36种可能性，表示每种点数和的概率，这个是分布模型。也可以不断的实验，通过采用的结果计算期望值，这就是**蒙特卡洛方法**

> 在前面的老虎机部分时，我们用采样的平均值估计数学期望，这就是蒙特卡洛方法

## 使用蒙特卡洛方法计算价值函数

回顾价值函数的定义：

$$
\nu _{\pi}(s) = \mathbb{E}_{\pi}[G_t|S_t = s]
$$

按照“**使用平均值进行期望值估计**”的思想，价值函数估计值可以表示为：

$$
\begin{aligned}
V_{\pi}(s) &= \frac{G^{(1)}+G^{(2)}+...+G^{(n)}}{n}
\\&= V_{n-1}(s)+\frac{1}{n}\{G^{(n)}-V_{n-1}(s)\}  
\end{aligned} 
$$

其中$G^{(n)}$表示第$n$轮测试的收益，例如从某个初始点开始，进行$n$次测试，每次到达目标点为止（注意此处只能适用**回合制问题**）

在这个思路的前提下，假设一共有$m$个$State$，那么一共需要进行$m×n$次测试，显然这样的更新方式是效率很低的。例如从$A$状态出发，经过了$B$和$C$到达终点，其实这一次测试可以更新三个状态的价值函数，然而上面的思想却只更新$A$状态的价值函数。

假设从$A$状态出发，经过了$B$和$C$到达终点，$A$到$B$的收益是$R_0$，以此类推，那么各个状态的价值函数估计可以表示为：

$$
\begin{aligned} 
G_A &= R_0 + \gamma R_1 + \gamma^2 R_2 = R_0+\gamma G_B
\\G_B&= R_1+\gamma R_2 = R_1+\gamma G_C
\\G_C&=R_2
\end{aligned} 
$$

因为更新$G_A$需要用到$G_B$，为了更新的独立性，我们**从后往前**更新：

$$
\begin{aligned} 
G_C&=R_2
\\G_B&= R_1+\gamma G_C
\\G_A &= R_0+\gamma G_B
\end{aligned} 
$$

仍然在**GridWorld**问题中应用，创建`RandomAgent`类实现：

*   `GetAction`：随机获取下一个行动
*   `Move`：记录一次移动的状态和收益，并更新当前状态
*   `Update`：更新价值函数

```python
class RandomAgent:
    def __init__(self,StartState):
        self.Cnts = defaultdict(lambda : 0)
        self.Records = []
        self.StartState = StartState
        self.State = StartState #指示当前的状态

        self.V = defaultdict(lambda : 0)
        self.gamma = 0.9
        self.pi = defaultdict(lambda : {'left':0.25,'right':0.25,'up':0.25,'down':0.25})

    def GetAction(self,State):
        ActionProbs = self.pi[State]
        return np.random.choice(list(ActionProbs.keys()),p=list(ActionProbs.values()))

    def Move(self,NextState,State,Reward):
        self.Records.append([State,Reward])
        self.State = NextState

    def Update(self):
        G = 0
        for Data in reversed(self.Records):
            State, Reward = Data
            G = Reward + self.gamma * G
            self.Cnts[State] += 1
            self.V[State] += (G - self.V[State]) / self.Cnts[State]

    def Reset(self): # 返回起点
        self.State = self.StartState
        self.Records = []
```

运行上述方法：

```python
steps = 1000
env = GridWorld()
Agent = RandomAgent(env.StartState)

for step in range(steps):
    Agent.Reset()
    while(1):
        State = Agent.State
        Action = Agent.GetAction(State)
        NextState = env.NextState(State,Action)
        Reward = env.GetReward(NextState)
        Agent.Move(NextState,State,Reward)
        if NextState == env.GoalState:
            Agent.Update()
            break
        State = NextState
```

<p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/810aecad26ba43f491907fb7c1b12456~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=nTPOol4DIR6ioXFE45VW4qsn0FY%3D" alt="image.png" width="50%"></p>

## 使用蒙特卡洛方法实现策略控制

回忆最优策略公式：

$$
\begin{aligned}
\pi_{*}(s) &= \argmax_{a}\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{*}(s^{'})\}
\\&=\argmax_{a}q_{*}(s,a)
\end{aligned}
$$

在本例子中，我们不知道环境参数，也就是不知道$p(s^{'}|s,a)$部分，所以只能实验**Q函数公式**来进行策略控制。

与价值函数迭代类似，Q函数迭代公式为：

$$
\begin{aligned}
Q_{n}(s,a) &= \frac{G^{(1)}+G^{(2)}+...+G^{(n)}}{n}
\\&= Q_{n-1}(s,a)+\frac{1}{n}\{G^{(n)}-Q_{n-1}(s,a)\}  
\end{aligned} 
$$

**蒙特卡洛方法实现策略控制：**

*   `GreedyProbs`实现了策略的迭代更新，主要就是找到Q函数最大的`Action`
*   `Update`函数增加了对策略的更新，其余不变

```python
def GreedyProbs(Q,State,Actions = ['left','right','up','down']):
    qs = [Q[(State,Action)] for Action in Actions]
    BestAction = np.argmax(qs)
    ActionProbs = {Action: 0 for Action in Actions}
    ActionProbs[Actions[BestAction]] = 1
    return ActionProbs

class McAgent:
    def __init__(self,StartState):
        self.Cnts = defaultdict(lambda : 0)
        self.Records = []
        self.StartState = StartState
        self.State = StartState #指示当前的状态

        self.Q = defaultdict(lambda : 0)
        self.gamma = 0.9
        self.pi = defaultdict(lambda : {'left':0.25,'right':0.25,'up':0.25,'down':0.25})

    def GetAction(self,State):
        ActionProbs = self.pi[State]
        return np.random.choice(list(ActionProbs.keys()),p=list(ActionProbs.values()))

    def Move(self,NextState,Action,State,Reward):
        self.Records.append([State,Action,Reward])
        self.State = NextState

    def Update(self):
        G = 0
        for Data in reversed(self.Records):
            State, Action, Reward = Data
            G = Reward + self.gamma * G
            Key = (State, Action)
            self.Cnts[Key] += 1
            self.Q[Key] += (G - self.Q[Key]) / self.Cnts[Key]
            self.pi[State] = GreedyProbs(self.Q,State)

    def Reset(self): # 返回起点
        self.State = self.StartState
        self.Records = []
        
steps = 1000
env = GridWorld()
Agent = McAgent(env.StartState)

for step in range(steps):
    Agent.Reset()
    while(1):
        State = Agent.State
        Action = Agent.GetAction(State)
        NextState = env.NextState(State,Action)
        Reward = env.GetReward(NextState)
        Agent.Move(NextState,Action,State,Reward)
        if NextState == env.GoalState:
            Agent.Update()
            break
        State = NextState
```

使用上述方法有两个弊端：

*   在`GreedyProbs`函数中不应该使用绝对的贪婪算法，这样会导致两种弊端：
    *   一旦确定从初始点到终点的最佳路径，那么这个路径不会变化，对于不在这个路径上的状态，其价值函数是无法更新的
    *   如果在第一次更新时，在初始点的最佳行动是往左（在这个问题中是可能的，因为撞墙是不会有惩罚的，只会停在原地不动），那么程序会陷入死循环，即一直撞墙
*   在 `Update`函数中不应该使用完全平均，而应该使用**移动平均**，具体原因可以类似于老虎机问题中的**非稳态情况**

## $\varepsilon$-greedy算法

其实就是在选取最优行动时，不完全实现贪婪，而保持一些“探索性”，具体实现方式是`e_GreedyProbs`函数中，输出不再是`[0,0,1,0]`的形式，而是类似`[0.1,0.1,0.7,0.1]`这样在**当前最优行动的选择概率最大的情况下，又有一定的概率选择其他行动的结果**。

```python
def e_GreedyProbs(Q,State,Epsilon=0, Actions = ['left','right','up','down']):
    qs = [Q[(State,Action)] for Action in Actions]
    BaseProbs = Epsilon/len(Actions)

    BestAction = np.argmax(qs)
    ActionProbs = {Action: BaseProbs for Action in Actions}
    ActionProbs[Actions[BestAction]] += (1 - Epsilon)
    return ActionProbs
```

同时将`Update`函数更新，主要变化是引入常数$\alpha$以更新$Q$值：

```python
def Update(self):
    G = 0
    for Data in reversed(self.Records):
        State, Action, Reward = Data
        G = Reward + self.gamma * G
        Key = (State, Action)
        self.Cnts[Key] += 1
        self.Q[Key] += (G - self.Q[Key]) * self.Alpha
        self.pi[State] = e_GreedyProbs(self.Q,State,self.Epsilon)
```

这样就不会有死循环的问题，也可以保证所有状态都有概率走到。

## 异策略型

在上面的方法中，我们采用$\varepsilon$-greedy算法进行策略的更新，事实上这是一种妥协，我们必须设置一定程度的“探索”概率以完成所有的状态更新。**理想情况下我们想用完全的“贪婪”，不希望有“探索”行为。**，那么可以采用**异策略型**。

先介绍一些概念：

*   **目标策略**：是我们希望学习和优化的策略。它通常是在理论分析或者算法设计中定义的理想策略，用于计算目标值（如在计算价值函数的目标值时），以此来引导行为策略朝着更好的方向学习。
*   **行为策略**：智能体在与环境交互过程中用于生成动作的策略。简单来说，它决定了智能体在每个状态下如何实际选择动作
*   **同策略型**：目标策略和行为策略没有区分
*   **异策略型**：有区分，分布是两个策略分布

在策略的更新中，我们想要最后的策略是完全的“贪婪”，但是如果我们真的按照完全"贪婪”则无法完成训练。所以我们可以设置一个策略，他具有“探索”行为，称之为**行为策略**，智能体的实际行动还是按照这个**行为策略**。又设置另外一个策略，完全按照"贪婪",称之为**目标策略**，是我们希望的策略更新方向。

由于行为策略和目标策略不同，在估计目标策略的价值函数时会遇到问题。因为按照行为策略收集的数据来直接估计目标策略的价值函数是不准确的。**重要性采样**就是用于解决这个问题的技术。

**重要性采集**

假如我们要估计$\pi$分布的数学期望，那最简单的办法是采集这个分布的值，然后算平均数：

$$
sampling:x_i\sim\pi

\\\mathbb{E}_{\pi}(x) =\sum x \pi(x)= \frac{x_1+x_2+x_3...x_n}{n}
$$

假如$x$不是从$\pi$分布中采集的，而是从$b$分布中采集的，那如何估计$\pi$分布的数学期望呢？

$$
sampling:x_i\sim b
\\
\mathbb{E}_{\pi}(x) =\sum x \frac{\pi(x)}{b(x)}b(x) = \mathbb{E}_{b}([x\frac{\pi(x)}{b(x)}]) = \mathbb{E}_{b}(\rho x)
$$

也就是说，**我们通过行为策略采集到了数据，但是我们需要估计的是目标策略的价值函数**，所以通过采集$\rho x$来估计目标策略的价值函数。

当这两个分概率分布差别较大时，$\frac{\pi(x)}{b(x)}$不稳定，导致采集到的值$x\frac{\pi(x)}{b(x)}$与真实的数学期望之间的差距较大，采集到的值的**方差很大**，意味着方法的稳定性较差，**保证两个分布的概率分布尽可能一样**是解决这个问题的方法，**在这里，两个分布的主要差异是探索系数，当`Epsilon`设置合理时，可以达到这个目的。**

下面介绍了蒙特卡洛方法中异型策略的实现方式，由于蒙特卡洛的数据是一个回合，所以从前面时间步到后

<p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/9eb65ba61e8247faa708116378a39946~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=QtXbVVprxRJKc%2Fua5gkTlWqtUjQ%3D" alt="6D1165383C26BCE9768763B80D225663.png" width="70%"></p>

```python
class McOffAgent:
    def __init__(self,StartState):
        self.Cnts = defaultdict(lambda : 0)
        self.Records = []
        self.StartState = StartState
        self.State = StartState #指示当前的状态

        self.Epsilon = 0.2
        self.Alpha = 0.1
        self.b = defaultdict(lambda : {'left':0.25,'right':0.25,'up':0.25,'down':0.25})

        self.Q = defaultdict(lambda : 0)
        self.gamma = 0.9
        self.pi = defaultdict(lambda : {'left':0.25,'right':0.25,'up':0.25,'down':0.25})

    def GetAction(self,State):
        ActionProbs = self.b[State]
        return np.random.choice(list(ActionProbs.keys()),p=list(ActionProbs.values()))

    def Move(self,NextState,Action,State,Reward):
        self.Records.append([State,Action,Reward])
        self.State = NextState

    def Update(self):
        G = 0
        rho = 1
        for Data in reversed(self.Records):
            State, Action, Reward = Data
            G = Reward + self.gamma * G * rho
            Key = (State, Action)

            self.Q[Key] += (G - self.Q[Key]) * self.Alpha
            rho *= self.pi[State][Action] / self.b[State][Action]
            self.pi[State] = e_GreedyProbs(self.Q,State,0)
            self.b[State] = e_GreedyProbs(self.Q,State,self.Epsilon)

    def Reset(self): # 返回起点
        self.State = self.StartState
```

# Chapter6：TD 方法

> **TD：Temporal Difference，时间差分**

蒙特卡洛方法的弊端在于，其只能应用于**回合制问题**，且当一个回合过长时，其更新速度很慢。
面对连续性问题或回合很长的回合制问题时，**TD方法**更有效。

*   动态规划法

*   蒙特卡洛方法

*   TD方法

$$
\begin{aligned} 
\nu _{\pi}(s) &=\sum_{a} \sum_{s^{'}}\pi(a|s)p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \nu _{\pi}(s^{'})\}
\\&=\mathbb{E}_{\pi}[R_t + \gamma V_{\pi}(S_{t+1})|S_t = s] 
\end{aligned}
$$

TD方法使用的更新公式为：

$$
V_{\pi}^{'} = V_{\pi}(S_t) + \alpha(R_t + \gamma V_{\pi}(S_{t+1})-V_{\pi}(S_{t}))
$$

仍然在**GridWorld**中进行实验，主要关注`Update`方法

```python
class TDAgent:
    def __init__(self,StartState):
        self.Cnts = defaultdict(lambda : 0)
        self.Records = []
        self.StartState = StartState
        self.State = StartState #指示当前的状态

        self.Alpha = 0.1

        self.V = defaultdict(lambda : 0)
        self.gamma = 0.9
        self.pi = defaultdict(lambda : {'left':0.25,'right':0.25,'up':0.25,'down':0.25})

    def GetAction(self,State):
        ActionProbs = self.pi[State]
        return np.random.choice(list(ActionProbs.keys()),p=list(ActionProbs.values()))


    def Update(self,Reward,State,NextState,GoalState):
        if NextState == GoalState:
            Target = Reward + self.gamma * 0
        else:            
            Target = Reward + self.gamma * self.V[NextState]
        self.V[State] +=self.Alpha * (Target - self.V[State])


    def Reset(self): # 返回起点
        self.State = self.StartState
        self.Records = []
```

```python
steps = 10000
env = GridWorld()
Agent = TDAgent(env.StartState)

for step in range(steps):
    Agent.Reset()
    while(1):
        State = Agent.State
        Action = Agent.GetAction(State)
        NextState = env.NextState(State,Action)
        Reward = env.GetReward(NextState)
        Agent.Update(Reward, State, NextState,env.GoalState) #每一步都要更新
        if NextState == env.GoalState:
            break
        Agent.State = NextState
```

<p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/00566e26e5ed492bb4665a73b2eb1946~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=oJdSOF%2BCEfq4Dln6%2B3MUtmMdOtc%3D" alt="image.png" width="50%"></p>

## SARSA

同样，在未知环境模型的情况下，我们还是实验Q函数来进行策略估计和策略控制

回忆TD方法的策略估计公式：

$$
V_{\pi}^{'}(S_t) = V_{\pi}(S_t) + \alpha(R_t + \gamma V_{\pi}(S_{t+1})-V_{\pi}(S_{t}))
$$

将其运用到**Q函数**中可以表示为：

$$
Q_{\pi}^{'}(S_t,A_t) = Q_{\pi}(S_t,A_t) + \alpha(R_t + \gamma Q_{\pi}(S_{t+1},A_{t+1})-Q_{\pi}(S_t,A_t))
$$

**使用SARSA方法进行策略控制：**

*   `Update`函数实现了Q函数的更新
    *   `Records`中保存了$(S_t,A_t)$和$(S_{t+1},A_{t+1})$
*   策略的更新采用了$\varepsilon$-greedy算法

```python
class SarsaAgent:
    def __init__(self,StartState):
        self.Cnts = defaultdict(lambda : 0)
        self.Records = []
        self.StartState = StartState
        self.State = StartState #指示当前的状态
        self.Alpha = 0.8
        self.Q = defaultdict(lambda : 0)
        self.gamma = 0.9
        self.pi = defaultdict(lambda : {'left':0.25,'right':0.25,'up':0.25,'down':0.25})
        self.Epsilon = 0.2
        self.Records = []
    def GetAction(self,State):
        ActionProbs = self.pi[State]
        return np.random.choice(list(ActionProbs.keys()),p=list(ActionProbs.values()))
        
    def Update(self,Reward,State,Action,GoalState):
        self.Records.append((State,Action,Reward))
        if len(self.Records)<2:
            return
        State,Action,Reward = self.Records[-2]
        NextState,NextAction,_ = self.Records[-1]
        if NextState == GoalState:
            Target = Reward + self.gamma * 0
        else:
            Target = Reward + self.gamma * self.Q[(NextState,NextAction)]
        self.Q[(State,Action)] +=self.Alpha * (Target - self.Q[(State,Action)])
        self.pi[State] = e_GreedyProbs(self.Q,State,self.Epsilon)

    def Reset(self): # 返回起点
        self.State = self.StartState
        self.Records = []
```

<!-- <img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/391531a734d8444fbcc8eef3ba006499~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=LTcZ0x5cfx2bb%2FnQyuGxr9xZhyM%3D" alt="TD1.png" width="50%"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/
104965c33fca4f7292edcc74fbe7e922~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=1zoz6zr6RkDSlEWDgqZzUgjgJMM%3D" alt="TD2.png" width="50%">
 -->
<!-- 
<img src="https://gitee.com/leeMX111/reinforcement_learning_imgs/raw/master/202505052045599.png" alt="TD1.png" width="45%"><img src="https://gitee.com/leeMX111/reinforcement_learning_imgs/raw/master/202505052045647.png" alt="TD2.png" width="45%"> -->
<div style="display: flex; align - items: center;">
    <img src="https://gitee.com/leeMX111/reinforcement_learning_imgs/raw/master/202505052045599.png" alt="TD1.png" width="50%">
    <img src="https://gitee.com/leeMX111/reinforcement_learning_imgs/raw/master/202505052045647.png" alt="TD2.png" width="50%">
</div>

## 异策略型的SARSA

在TD问题中，由于只考虑了上下两层状态，所以权重表示为：

$$
\rho = \frac{\pi(A_{t+1}|S_{t+1})}{b(A_{t+1}|S_{t+1})}
$$

因此其更新公式为

$$
sampling:A_{t+1}\sim b
\\
Q_{\pi}^{'}(S_t,A_t) = Q_{\pi}(S_t,A_t) + \alpha\{\rho(R_t + \gamma Q_{\pi}(S_{t+1},A_{t+1}))-Q_{\pi}(S_t,A_t)\}
$$

```python
class SarsaOffAgent:
    def __init__(self,StartState):
        self.Cnts = defaultdict(lambda : 0)
        self.Records = []
        self.StartState = StartState
        self.State = StartState #指示当前的状态

        self.Alpha = 0.8

        self.Q = defaultdict(lambda : 0)
        self.gamma = 0.9
        self.pi = defaultdict(lambda : {'left':0.25,'right':0.25,'up':0.25,'down':0.25})
        self.b = defaultdict(lambda: {'left': 0.25, 'right': 0.25, 'up': 0.25, 'down': 0.25})
        self.Epsilon = 0.2
        # self.Records = deque(maxlen=2)
        self.Records = []
    def GetAction(self,State):
        ActionProbs = self.b[State]
        return np.random.choice(list(ActionProbs.keys()),p=list(ActionProbs.values()))


    def Update(self,Reward,State,Action,GoalState):
        self.Records.append((State,Action,Reward))
        if len(self.Records)<2:
            return
        State,Action,Reward = self.Records[-2]
        NextState,NextAction,_ = self.Records[-1]

        if NextState == GoalState:
            Target = Reward + self.gamma * 0
            rho = 1
        else:
            rho = self.pi[(NextState,NextAction)] / self.b[(NextState,NextAction)]
            Target = rho * (Reward + self.gamma * self.Q[(NextState, NextAction)])

        self.Q[(State,Action)] +=self.Alpha * (Target - self.Q[(State,Action)])
        self.pi[State] = e_GreedyProbs(self.Q,State,0)
        self.b[State] = e_GreedyProbs(self.Q,State,self.Epsilon)


    def Reset(self): # 返回起点
        self.State = self.StartState
        self.Records = []

steps = 10000
env = GridWorld()
Agent = SarsaAgent(env.StartState)

for step in range(steps):
    Agent.Reset()
    while(1):
        State = Agent.State
        Action = Agent.GetAction(State)
        NextState = env.NextState(State,Action)
        Reward = env.GetReward(NextState)
        Agent.Update(Reward, State,Action,env.GoalState)
        if NextState ==env.GoalState:
            Agent.Update(None,NextState,None,env.GoalState)
            break
        Agent.State = NextState
```

## Q学习（重要）

重要性采集事实上容易变得不稳定，尤其当两者策略的概率分布差别变大时，权重$\rho$的变化就会大，SARSA中的更新方向就会发生变化，从而使得Q函数的更新变得不稳定，**Q学习**就是解决这个问题的方法。**Q学习具有下列三个特点**：

*   **采用TD方法**
*   **异策略型**
*   **不使用重要性采样**

为了联合SARS了解Q学习，回忆贝尔曼方程

$$
\begin{aligned} 
q_{\pi}(s,a)&=\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \sum_{a^{'}}\pi(a^{'}|s^{'})q_{\pi}(s^{'},a^{'})\}
\\& =\mathbb{E}_{\pi}[R_t + \gamma Q_{\pi}(S_{t+1},A_{t+1})|S_t = s,A_t=a] 
\end{aligned}
$$

其考虑到了状态转移概率下$p(s^{'}|s,a)$的所有下一个状态，又考虑到了策略$\pi(a^{'}|s^{'})$下的所有下一个动作。

<p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/7baf6a8d2dfa4ed7871b9e282401842e~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=uqRShQZ0hW5bpGdvl3fXqRwncMQ%3D" alt="EE424612F2271E59A66E76468882F639.png" width="50%"></p>

回忆SARSA的公式：

$$
Q_{\pi}^{'}(S_t,A_t) = Q_{\pi}(S_t,A_t) + \alpha\bigg\{R_t + \gamma Q_{\pi}(S_{t+1},A_{t+1})-Q_{\pi}(S_t,A_t)\bigg\}
$$

**考虑SARSA的本质，其实是贝尔曼方程的一种采样**，其先基于$p(s^{'}|s,a)$对下一个状态进行采样，然后基于$\pi(a|s)$对下一步的行动采样，于是Q函数的更新方向就是$R_t + \gamma Q_{\pi}(S_{t+1},A_{t+1})$

<p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/5a7eef18350c4f88858d2113369ce0e8~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=GCiodCIte91T9Au%2BdVVRxkU9BC4%3D" alt="3F0B72ABDF5BE8A633B5F855DD6AE4C2.png" width="50%"></p>

**如果说SARSA对应着贝尔曼方程，那么Q方法就是对应着贝尔曼最优方程**，其在选择下一个动作时，**不再依靠策略进行采样，而是直接选择最优的动作**。

回顾贝尔曼最优方程

$$
\begin{aligned} 
q_{*}(s,a)&=\sum_{s^{'}}p(s^{'}|s,a)\{r(s,a,s^{'})+\gamma \max_{a^{'}} q_{*}(s^{'},a^{'})\}
\\&=\mathbb{E}[R_t + \gamma \max_{a^{'}} q_{*}(s^{'},a^{'})|S_t = s,A_t=a] 
\end{aligned}
$$

将其写成采样形式：

$$
\begin{aligned} 
Q^{'}(S_t,A_t)=\mathbb{E}[R_t + \gamma \max_{a^{'}}Q(S_{t+1},a^{'})] 
\end{aligned}
$$

$$
Q^{'}(S_t,A_t) = Q(S_t,A_t) + \alpha \bigg\{ R_t + \gamma \max_{a^{'}}Q(S_{t+1},a^{'})-Q(S_t,A_t)\bigg\}
$$

**由于对$A_{t+1}$的选择直接使用的$max$，不需要重要性选择进行修正。**

> 读到这里时可能会有读者产生疑问：**在选择动作时，Q学习采样的是$max$，但是在状态转移时为什么却选择基于采样？**
> ![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/17e0dc04095943e6a3f2d1092e433b51~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=ECKotD9VZwR6xehOxpfBJnCRLm0%3D)

**Q学习的实现：**

```python
class QLearningAgent:
    def __init__(self,StartState):
        self.Cnts = defaultdict(lambda : 0)
        self.Records = []
        self.StartState = StartState
        self.State = StartState #指示当前的状态

        self.Alpha = 0.8

        self.Q = defaultdict(lambda : 0)
        self.gamma = 0.9
        self.pi = defaultdict(lambda : {'left':0.25,'right':0.25,'up':0.25,'down':0.25})
        self.b = defaultdict(lambda: {'left': 0.25, 'right': 0.25, 'up': 0.25, 'down': 0.25})
        self.Epsilon = 0.2

    def GetAction(self,State):
        ActionProbs = self.b[State]
        return np.random.choice(list(ActionProbs.keys()),p=list(ActionProbs.values()))

    def Update(self,Reward,State,Action,NextState,GoalState):
        if NextState == GoalState:
            MaxQNextState = 0
        else:
            NextStateScore = [self.Q[(NextState,a)] for a in ['up','down','left','right']]
            MaxQNextState = max(NextStateScore)
        Target = Reward+self.gamma * MaxQNextState
        self.Q[(State,Action)] +=self.Alpha * (Target - self.Q[(State,Action)])
        self.pi[State] = e_GreedyProbs(self.Q,State,0)
        self.b[State] = e_GreedyProbs(self.Q,State,self.Epsilon)


    def Reset(self): # 返回起点
        self.State = self.StartState

steps = 10000
env = GridWorld()
Agent = QLearningAgent(env.StartState)

for step in range(steps):
    Agent.Reset()
    while(1):
        State = Agent.State
        Action = Agent.GetAction(State)
        NextState = env.NextState(State,Action)
        Reward = env.GetReward(NextState)
        Agent.Update(Reward, State,Action,NextState,env.GoalState)
        if NextState == env.GoalState:
            break
        Agent.State = NextState
```

## 样本模型版的Q学习

**样本模型**是对应于**分布模型**，其最大的特点是不保存特定的概率分布。

*   观察Q学习代码，其实`pi`完全没有参与Q函数的更新，由于其在需要的时候，可以由Q函数值马上得到，所以其实可以直接删除
*   对于策略`b`，其本质上也是根据Q函数的$\varepsilon$-greedy算法一种实现，在`GetAction`函数中再现这一过程即可，也可以删除

```python
class QLearningAgent_2:
    def __init__(self,StartState):
        self.Cnts = defaultdict(lambda : 0)
        self.Records = []
        self.StartState = StartState
        self.State = StartState #指示当前的状态

        self.Alpha = 0.8

        self.Q = defaultdict(lambda : 0)
        self.gamma = 0.9
        self.Epsilon = 0.2
        self.ActionSize = ['up','down','left','right']

    def GetAction(self,State):
        if np.random.rand()<self.Epsilon:
            return np.random.choice(self.ActionSize)
        else:
            qs = {a:self.Q[(State,a)] for a in self.ActionSize}
            BestQ = -1
            BestAction = None
            for k,v in qs.items():
                if v>BestQ:
                    BestQ = v
                    BestAction = k
            return BestAction
    def Update(self,Reward,State,Action,NextState,GoalState):
        if NextState == GoalState:
            MaxQNextState = 0
        else:
            NextStateScore = [self.Q[(NextState,a)] for a in self.ActionSize]
            MaxQNextState = max(NextStateScore)
        Target = Reward+self.gamma * MaxQNextState
        self.Q[(State,Action)] +=self.Alpha * (Target - self.Q[(State,Action)])


    def Reset(self): # 返回起点
        self.State = self.StartState
```

# Discussion

前面3-6章基本上都是针对贝尔曼方程进行的一些工作：

<!-- ![强化学习前期.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/79018863bc6843d1b99aea53fd46aa42~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=i3acqOdYW%2FvnhfvcnMgYL1TJWes%3D)
 -->


<img src="https://gitee.com/leeMX111/reinforcement_learning_imgs/raw/master/202505052056144.png" alt="强化学习前期.png">