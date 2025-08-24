---
title: 《深度学习4-强化学习》学习笔记-2
cover: /imgs/GPT2.jpg
swiper_index: 4 #置顶轮播图顺序，非负整数，数字越大越靠前
categories: # 分类
	- 强化学习  # 只能由一个
tags:
    - 算法
    - 强化学习
    - 学习
---
<meta name="referrer" content="no-referrer" />
 
# Chapter7：神经网络和Q学习

在前面的方法中，我们都是用一个表格形式存储Q函数，但是复杂情况下，状态数量太多，无法将其存储为表格形式，用更加紧凑的函数近似Q函数是一种处理方法，其中最有效的便是神经网络

**机器学习和深度学习的基础知识此处省略**

将**求Q函数的过程抽象为神经网络，有两种形式**：

*   输入状态$s$和行动$a$，输出Q函数值$Q(s,a)$
*   输入状态$s$，输入所有行动的Q函数列表

首先我们需要知道神经网络的输入输出，以及其误差函数的设计，这个地方书上写的比较模糊，书上采用了下面这个公式说明，也就是Q学习的迭代公式

$$
Q_{\pi}^{'}(S_t,A_t) = Q_{\pi}(S_t,A_t) + \alpha \bigg\{ R_t + \gamma \max_{a^{'}}Q_{\pi}(S_{t+1},a^{'})-Q_{\pi}(S_t,A_t)\bigg\}
$$

然后书上的代码表达的意思是，**神经网络拟合了Q函数值**，然后误差函数的形式为：

$$
loss( R_t +\gamma \max_{a^{'}}Q_{\pi}(S_{t+1},a^{'}),Q_{\pi}(S_t,A_t))
$$

当然书上的做法是没有问题的，但是但从这两个地方会产生疑惑：这个误差函数的理论依据是什么？跟Q学习的迭代公式好像也没关系啊。其实应该是按照下面这个公式：

$$
\begin{aligned} 
Q^{'}(S_t,A_t)=\mathbb{E}[R_t + \gamma \max_{a^{'}}Q(S_{t+1},a^{'})] 
\end{aligned}
$$

**所以，下一次迭代的$Q^{'}(S_t,A_t)$需要接近$R_t + \gamma \max_{a^{'}}Q(S_{t+1},a^{'})$，由于神经网络本来就是不断更新和迭代的模型，其内部本身就蕴含数学期望的概念，所以上面这个公式才是神经网络更新的基础公式**

**接下来在GridWorld上实现这一操作：**

首先是`onehot`操作，将状态转换为onehot格式以便输入神经网络：

```python
def OneHot(state):
    height = 3
    width = 4
    state_one_hot = np.zeros(height*width)
    state_one_hot[state[0]*width+state[1]] = 1
    return  torch.tensor(state_one_hot[np.newaxis,:],dtype=torch.float)
```

定义模拟Q函数的神经网络`QNet`，由两个线性层组成：

```python
class QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
```

定义`QLearningAgent`:

*   `GetAction`：将状态输入神经网络，输出最大的动作即为选择的动作
*   `Update`：更新神经网络
    *   `target`：$R_t + \gamma \max_{a^{'}}Q(S_{t+1},a^{'})$
    *   `q`：$Q^{'}(S_t,A_t)$

```python
class QLearningAgent:
    def __init__(self,start_state):
        self.start_state = start_state
        self.state = start_state #指示当前的状态

        self.lr = 0.01
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.9
        self.actions =  ['left','right','up','down']
        self.num_states_col = 4
        self.num_states_row = 3

        self.qnet = QNet(self.num_states_col*self.num_states_row,100,len(self.actions))
        self.optimizer = optim.SGD(self.qnet.parameters(),lr = self.lr)

    def GetAction(self,state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            y =  self.qnet(OneHot(state))
            return self.actions[y.data.argmax().numpy()]

    def Update(self, reward, state, action, next_state, goal_state):
        if next_state == goal_state:
            next_q = torch.tensor([0])
        else:
            next_qs = self.qnet(OneHot(next_state))
            next_q = next_qs.max(axis = 1)[0].detach()


        target = self.gamma * next_q + reward
        qs = self.qnet(OneHot(state))

        q = qs[:,self.actions.index(action)]

        loss1 = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss1.backward()
        self.optimizer.step()

        return loss1.data
    def Reset(self): # 返回起点
        self.state = self.start_state
```

```python
env = GridWorld()
agent = QLearningAgent(env.StartState)

steps = 1000
history_loss = []
for step in range(steps):
    agent.Reset()
    total_loss = 0
    cnt = 0
    while(1):
        state = agent.state
        action = agent.GetAction(state)
        next_state = env.NextState(state,agent.actions[action])
        reward = env.GetReward(next_state)
        loss = agent.Update(reward, state,action,next_state,env.GoalState)
        cnt+=1
        total_loss+=loss
        if next_state == env.GoalState:
            break
        agent.state = next_state
    history_loss.append(total_loss/cnt)
```

# Chapter8：DQN

## DQN

DQN是一种运用神经网络模拟Q函数的改进形式，主要改进有两个方面：**经验回放**和**目标网络**

### 经验回放

在一般的有监督学习中，我们一般会采用**小批次**的概念，也就是说每次训练从数据集中拿出一部分数据来进行训练。一个批次中的数据都是独立的，这样可以**防止出现数据偏差**

但是在上面神经网络和Q学习的实践中，数据之间有很强的相关性，比如这一时刻训练的数据是$D_t$，下一时刻训练数据是$D_{t+1}$，显然这两个数据直接是直接相关的（比如$D_t$中的状态和行动都影响了$D_{t+1}$的状态），这导致了一定程度的偏差。

解决方法--**经验回放**：将数据都存储起来，在训练时再随机提取出小批量数据进行训练

### 目标网络

在监督学习中，数据的label是不会发生改变的，但是在chapter7中的做法中，实际上数据的label会随着训练而变化，例如同一个`state`，在不同时间下的`target`是不一样的。在为了弥补这一问题，提出**目标网络**

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/15a2052f6189405bab05964b44a5e7c1~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=JvRZfMNazduhe6ok9UD3Yys8CkE%3D)

**目标网络：** 设立应该跟主神经网络结构一样的网络，**这个网络参数不随时变化**，`target`由这个网络得到，使得训练目标是相对稳定的。**但是这个网络也不能不更新，应该设置一个时间定期与主训练网络同步参数。**

*   主神经网络，其Q函数用$Q_{\theta}$表示，其参数一直在训练和更新
*   目标网络：其Q函数用$Q_{\theta ^ {'}}$表示，其参数不随时更新，只是一段时间后与主神经网络同步
*   主神经网络更新：$Rrward+\gamma \max_{a^{'}}Q_{\theta ^ {'}}(S_{t+1},a^{'})$--逼近-->$Q_{\theta}(S_t,A_t)$

这里用`gym`库模拟倒立摆的问题：

<p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/8305a89de23843ce94847b3caf9daf44~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=wSuC%2F6MBaQQpRBAhU3%2BwFiTdQR4%3D" alt="1a64906badd062445a321ca615c7cfd5.png" width="70%"></p>

*   `ReplayBuffer`：实现经验回放，本质上是使用队列完成，列表也能实现这一功能。

```python
class ReplayBuffer:
    def __init__(self,max_len,batch_size):
        self.deque = deque(maxlen=max_len)
        self.batch_size = batch_size
    def __len__(self):
        return len(self.deque)

    def Add(self,state,action,reward,next_state,done):
        self.deque.append((state,action,reward,next_state,done))

    def GetBatch(self):
        data = random.sample(self.deque,self.batch_size)
        state = torch.tensor(np.stack([x[0] for x in data]))
        action = torch.tensor(np.array([x[1] for x in data]).astype(np.int32))
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32))
        next_state = torch.tensor(np.stack([x[3] for x in data]))
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32))
        return state, action, reward, next_state, done
```

*   `DQNAgent`：实现数据记录和参数更新
    *   `sync_qnet`：将`qnet_target`的参数和`qnet`同步
    *   `Update`：神经网络训练，此处`tagret`是由`qnet_target`得到，而`q`由`qnet`得到，这就实现了目标网络

```python
class DQNAgent:
    def __init__(self,start_state):
        self.start_state = start_state
        self.state = start_state #指示当前的状态

        self.lr = 0.0005
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.98

        self.action_size = 2 #倒立摆的动作只有两种


        self.batch_size = 32
        self.max_len = 10000

        self.qnet = QNet(4,100,self.action_size)
        self.qnet_target = QNet(4, 100, self.action_size)

        self.optimizer = optim.Adam(self.qnet.parameters(),lr = self.lr)
        self.replay_buffer = ReplayBuffer(self.max_len,self.batch_size)

    def sync_qnet(self):
        # self.qnet_target = copy.deepcopy(self.qnet)
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def GetAction(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:

            state = torch.tensor(state[np.newaxis, :])
            qs = self.qnet(state)
            return qs.argmax().item()

    def Update(self, reward, state, action, next_state, done):
        self.replay_buffer.Add(state,action,reward,next_state,done)
        if len(self.replay_buffer)<self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.GetBatch()

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis = 1)[0].detach()

        target = self.gamma * (1-done)*next_q + reward
        qs = self.qnet(state)
        q = qs[np.arange(len(action)),action]

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def Reset(self): # 返回起点
        self.state = self.start_state
```

这里与我们之前的做法有一些差别，`gym`库的`env`提供了`step`函数，不需要我们像之前一样调用各种函数。另外，当前的`state`是保存在`env`中的：

```python
env = gym.make('CartPole-v0')
replay_buffer = ReplayBuffer(10000,32)
start_state = env.reset()
agent = DQNAgent(start_state[0])
steps = 300
history_reward = []

for step in range(steps):
    state=env.reset()[0]
    total_reward = 0
    done = False


    while(1):
        action = agent.GetAction(state)
        next_state, reward, done, info,_ = env.step(action)

        agent.Update(reward, state,action,next_state,done)
        total_reward+=reward

        if done:
            break
        state = next_state
    if step%20 == 0:
        agent.sync_qnet()
    if step % 10 == 0:
        print("episode :{}, total reward : {}".format(step, total_reward))

    history_reward.append(total_reward)
#
```

<p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/bec702ddf6eb4d68a1d575f7464a4d0d~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=cfpdC5huIxY3a%2FIvkzIMubAfNCU%3D" alt="DQN1.png" width="70%"></p>

## Double DQN

回忆DQN的训练方式，$Q_{\theta}(S_t,A_t)$逼近的目标为：

$$
Rrward+\gamma \max_{a^{'}}Q_{\theta ^ {'}}(S_{t+1},a^{'})
$$

**问题是，如果对包含误差的估计值（$Q_{\theta ^ {'}}$）使用max算子，那么与使用真正的Q函数基线计算的情况相比结果会过大。**

解决方法是**Double DQN**，$Q_{\theta}(S_t,A_t)$逼近的目标为：

$$
Rrward+\gamma Q_{\theta ^ {'}}\bigg(S_{t+1},\argmax_aQ_{\theta}(S_{t+1},a)\bigg)
$$

也就是说，对于目标网络的动作选择，来自于主神经网络，而不采用max算子，这样的好处是可以**避免过估计，使得训练更稳定。**

## 优先级经验回放

在之前的经验回放中，我们的做法是把数据放在一起，然后随机选取进行训练，一种优化方法是设置数据的**优先级**

其实本质上是在存储数据时，同时记录这个数据的误差$\delta$，**然后选取数据时，误差越大的数据选取的概率越高。** 可以理解为，误差越大的数据能使得网络的更新更大。

$$
\delta_t =| Rrward+\gamma \max_{a^{'}}Q_{\theta ^ {'}}(S_{t+1},a^{'})-Q_{\theta}(S_t,A_t)|
$$

# chapter9：策略梯度法

前面所有的方法本质上都是对**价值函数的更新迭代从而得到最优策略**，这类方法叫**基于价值的方法**。如果我们的最终目的是得到最优策略，为什么不直接模拟策略呢，得到收益最大的策略，是一个典型的神经网优化问题，这个过程中**不借助价值函数直接表示策略**，称为**基于策略的方法**

**具体思路：** 训练一个神经网络，输入为状态，输入为行动概率，更新的误差为收益的数学期望负数，我们的目的是误差最小，也就是收益最大，这是一个很简单的思想

假设在一个回合中，基于策略行动，我们得到了“状态，行动，奖励”构成的时间序列$\tau$，称之为"**轨迹**”：

$$
\tau = (S_0,A_0,R_0,S_1...)
$$

那么在这个轨迹下，收益为：

$$
G(\tau) = R_0 + \gamma R_1 + \gamma ^2 R_2...
$$

误差函数为表示为：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi}[G(\tau)]
$$

$\tau \sim \pi$表示策略得到的轨迹，然后确定误差函数的梯度为：

$$
\nabla_{\theta} J(\theta)=\mathbb{E}_{\tau \sim \pi} \bigg[\sum_{t=0}^{T}\nabla_{\theta} \log\pi_{\theta}(A_{t}|S_{t})G(\tau)\bigg]
$$

数学期望的计算还是可以使用蒙特卡洛方法进行近似，为了简化，我们这里直接采样1次当作数学期望：

$$
\nabla_{\theta} J(\theta)=\sum_{t=0}^{T}\nabla_{\theta} \log\pi_{\theta}(A_{t}|S_{t})G(\tau)
$$

**还是用倒立摆的问题实践上述方法：**

*   `Policy`：定义神经网络，输入需要进行`softmax`，因为表示的是动作的概率

```python
class Policy(nn.Module):
    def __init__(self, input_size,hidden_size,action_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x
```

*   `Agent`：实现数据记录和参数更新
    *   `Add`：记录数据，记录的是当前步的`reward`和采取行动的概率，当回合结束拿出来训练模型
    *   `Update`：神经网络训练，一个回合训练一次

```python
class Agent:
    def __init__(self,start_state):
        self.start_state = start_state
        self.state = start_state #指示当前的状态

        self.lr = 0.0005
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.98

        self.action_size = 2 #倒立摆的动作只有两种

        self.pi = Policy(4,100,self.action_size)
        self.optimizer = optim.Adam(self.pi.parameters(),lr = self.lr)

        self.records = []


    def GetAction(self, state):

        state = torch.tensor(state[np.newaxis, :])
        probs = self.pi(state)[0]
        action = np.random.choice(len(probs),p = probs.data.numpy())
        return action,probs[action]

    def Add(self,reward,probs):
        self.records.append((reward,probs))

    def Update(self):
        G=0
        loss = 0
        for reward,probs in reversed(self.records):
            G=reward+self.gamma*G

        for reward, probs in self.records:
            loss+= -torch.log(probs)*G
        self.optimizer.zero_grad()
        loss.backward()#求导
        self.optimizer.step()
        self.records = []
```

<p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/f8932c6478284343b7ca5c1f1ff31b6f~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=chyIIHGhaTUi%2BQXBUz7ZkiiEr5o%3D" alt="policy1.png" width="70%"></p>

做完这些，我有一个疑问：**既然`G`是表征策略好坏的直接标准，为什么不能直接对其求梯度呢？**（来自豆包）
![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/453e6043510c4f1d9eb0f5d459ddbe16~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=vWl5%2B8dR11cbDguKPhulKHOCQuE%3D)

## REINFORCE算法

回顾关于策略的梯度法公式：

$$
\nabla_{\theta} J(\theta)=\mathbb{E}_{\tau \sim \pi} \bigg[\sum_{t=0}^{T}\nabla_{\theta} \log\pi_{\theta}(A_{t}|S_{t})G(\tau)\bigg]
$$

其实本质上是对当前策略的评估，如果当前策略较好，其G值就会高，那么

但是我们会发现，在不同的时间t下，$\log\pi_{\theta}(A_{t}|S_{t})$的权重都是一样的，都是这个回合的收益$G(\tau)$，但是其实动作的好坏和之前的收益是无关的，如果我们能**直接体现当前时间的动作的收益**，那么收敛速度会快很多。那么可以将梯度公式改为：

$$
\begin{aligned}
\nabla_{\theta} J(\theta)&=\mathbb{E}_{\tau \sim \pi} \bigg[\sum_{t=0}^{T}\nabla_{\theta} \log\pi_{\theta}(A_{t}|S_{t})G_t\bigg]
\\G_t &= R_t + \gamma R_{t+1} + \gamma ^2 R_{t+2}...
\end{aligned}
$$

那么**每一步的动作都和这个动作本身的收益绑定了**，也更能收敛最优动作。在代码中只需要更改`Update`函数中`G`的计算方式：

```python
def Update(self):
    G=0
    loss = 0
    for reward,probs in reversed(self.records):
        G=reward+self.gamma*G
        loss += -torch.log(probs) * G


    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.records = []
```

<p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/8899635ae9114a1d80fdca53c6f123d5~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=WRODYyyB0KKltOua2oiOWQLLeD8%3D" alt="REINFOCE1.png" width="70%"></p>

> 可以证明上述两种方，在无限样本的情况下，都会收敛到正确的值，但是REINFORCE的方差更小，收敛速度更快，其权重中没有了无关的数据。

## 基线

基线的思路是**利用另外一个函数，通过做差的方式减小数据的方差**

> 可以通过一个例子说明这个原理，假设有三个同学的成绩，分别为90，40，50，方差是466.67，但是如果我们拿这三个同学之前十次考试的平均成绩（82，46，49）作为基准，而这次的成绩可以表示为（8，-6，1），方差缩小到32.67

可以针对收益$G_t$运用这个原理，其中$b(S_t)$可以是任意用状态作为输入的函数，实践中常常使用**价值函数**，那么这种情况下还**需要额外训练价值函数**：

$$
\begin{aligned}
\nabla_{\theta} J(\theta)&=\mathbb{E}_{\tau \sim \pi} \bigg[\sum_{t=0}^{T}G_t\nabla_{\theta} \log\pi_{\theta}(A_{t}|S_{t})\bigg]
\\ &= \mathbb{E}_{\tau \sim \pi} \bigg[\sum_{t=0}^{T}(G_t-b(S_t))\nabla_{\theta} \log\pi_{\theta}(A_{t}|S_{t})\bigg]
\end{aligned}
$$

这里用倒立摆的例子说明基准的意义，假设某个时刻倒立摆的角度很大，在3步后不管采取什么行动都会失败，此时的$G_t=3$，那么按照之前的方法，其还是会趋向于选择一个动作去逼近，但是其实是没有意义的。如果按照基线的思想，此时$b(S_t)=3$，那么其梯度为0，则不会更新，避免了无意义的训练和更新。

## Actor-Critic

> actor：行动者，表示策略；critic：评论者，表示价值函数。Actor-Critic的意思是使用价值函数评价策略的好坏。

强化学习的算法大致可以分为**基于价值**的方法和**基于策略**的方法，在之前的文章中介绍了两类方法，现在考虑一种**使用二者**的方法：

回忆带基线的REINFORCE算法梯度表示：

$$
\begin{aligned}
\nabla_{\theta} J(\theta)= \mathbb{E}_{\tau \sim \pi} \bigg[\sum_{t=0}^{T}(G_t-b(S_t))\nabla_{\theta} \log\pi_{\theta}(A_{t}|S_{t})\bigg]
\end{aligned}
$$

理论上函数$b$可以是任意以状态为输入的函数，此处我们使用基于**神经网络建模的价值函数作为基线**，可以表示为：

$$
\begin{aligned}
\nabla_{\theta} J(\theta)= \mathbb{E}_{\tau \sim \pi} \bigg[\sum_{t=0}^{T}(G_t-V_{\omega}(S_t))\nabla_{\theta} \log\pi_{\theta}(A_{t}|S_{t})\bigg]
\end{aligned}
$$

其中$\omega$表示神经网络参数，$V_{\omega}(S_t)$表示将状态输入神经网络所输出的价值函数值，根据TD方法的原理，训练神经网络时，我们使用$R_t+\gamma V_{\omega}(S_{t+1})$去逼近$V_{\omega}(S_t)$，这样的好吃是只需要采样下一状态的值即可进行训练。

$$
\begin{aligned}
\nabla_{\theta} J(\theta)= \mathbb{E}_{\tau \sim \pi} \bigg[\sum_{t=0}^{T}(R_t+\gamma V_{\omega}(S_{t+1})-V_{\omega}(S_t))\nabla_{\theta} \log\pi_{\theta}(A_{t}|S_{t})\bigg]
\end{aligned}
$$

**接下来还是在倒立摆的例子中实现这一思想：**

*   首先分别定义两个网络模型
    *   这里的`ValueNet`要区别于之前的DQN中的网络，这里是针对价值函数，所以输出只有一个

```python
class PolicyNet(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x

class ValueNet(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
```

*   定义智能体
    *   `ValueNet`：$R_t+\gamma V_{\omega}(S_{t+1})$-->$V_{\omega}(S_t)$
    *   `PolicyNet`：$\max \bigg [((R_t+\gamma V_{\omega}(S_{t+1})-V_{\omega}(S_t)) \log\pi_{\theta}(A_{t}|S_{t})\bigg]$

```python
class Agent:
    def __init__(self,start_state):
        self.start_state = start_state
        self.state = start_state #指示当前的状态

        self.lr = 0.0005
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.98

        self.action_size = 2 #倒立摆的动作只有两种

        self.pi = PolicyNet(4,100,self.action_size)
        self.optimizer_pi = optim.Adam(self.pi.parameters(),lr = self.lr)

        self.value = ValueNet(4,100,1)
        self.optimizer_value = optim.Adam(self.value.parameters(),lr = self.lr)

        self.loss_func = nn.MSELoss()



    def GetAction(self, state):

        state = torch.tensor(state[np.newaxis, :])
        probs = self.pi(state)[0]
        action = np.random.choice(len(probs),p = probs.data.numpy())
        return action,probs[action]


    def Update(self,state,reward,prob,next_state,done):
        state = torch.tensor(state[np.newaxis, :])
        next_state = torch.tensor(next_state[np.newaxis, :])

        target = (reward+(1-done)*self.gamma*self.value(next_state))
        v = self.value(state)
        loss_for_value = self.loss_func(target,v)

        loss_for_pi = -(target - v)*torch.log(prob)

        self.optimizer_pi.zero_grad()
        self.optimizer_value.zero_grad()

        loss_for_pi.backward(retain_graph=True)
        loss_for_value.backward(retain_graph=True)

        self.optimizer_pi.step()
        self.optimizer_value.step()
```

## 基于策略的方法的优点

*   **策略直接模型化，更高效**
    *   我们最终的目的是得到最优策略，对于某些价值函数很复杂却最优策略简单的问题，基于策略的方法显然更高效
*   **在连续的行动空间中也能使用**
    *   之前的例子都是离散的行动空间，例如之前的GridWorld，行动也只有四个，对于连续情况下难以运用， 例如行动是速度和方向等。这种情况可以采用离散化的方法，但是这是很困难的。如果采用基于策略的方法就能实现，例如直接输出速度等。
*   **行动的选择概率平滑地变化**
    *   之前对于策略的选择一般基于greed算法，如果Q函数发生大的变化，则选择的动作也会变化，而策略的方法是通过概率来是西安，更新的过程中会比较平稳，训练也比较稳定。

# Chapter10：进一步学习

## 模型分类

![扫描全能王 2024-12-14 14.04\_2.jpg](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c6b6074715f44ab99aa5a384eabad933~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=v75N%2F3lMuY2jkvNX1%2Fg003%2BkTGo%3D)
有无模型体现的是**是否使用环境模型**，如果未知模型则训练环境模型即可，在本文大部分情况下都是不使用模型的情况。

## A3C

**Asynchronous（异步） Advantage Actor-Critic算法**

A3C需要多个本地神经网络和一个全局神经网络，本地网络在各自环境中独自训练，然后将训练结果的梯度发送到全局网络。全局网络使用这些梯度异步更新权重参数。在更新全局网络的同时，定期同步全局网络和本地网络的权重参数。

<p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/02b8f1815ad84443bcf936417a062660~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=m1mIBoX319nQ3XAzSQg%2FC5t%2Fnvc%3D" alt="扫描全能王 2024-12-14 14.04_1.jpg" width="50%"></p>

A3C的优点：

*   加快训练速度
*   多个Agent独立行动，可以得到更加多样化的数据，减少数据之间的相关性，使得训练更加稳定。

另外，AC3的Actor-Critic将共享神经网络的权重，在靠近输入侧的神经网络的权重是共享的，这样的好处是加快训练速度，减小成本。

<p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/71ea0a4b4eba44d5b92c341a9d810fe1~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=1%2FyUKjEk7W4uk5da0g%2BJ8go8DEg%3D" alt="扫描全能王 2024-12-14 14.04_3.jpg" width="50%"></p>

## A2C

A2C是同步更新参数的方法，不采取异步更新参数的方式。其主要的特点是在本地不需要神经网络，只是独自运行智能代理，将不同环境中的状态作为数据进行批数据汇总进行训练。然后将神经网络的输出中对下一个行动采用，分发给各个环境。

<p align="center"><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/4cd8892c611d4a99946f6c484c875ad8~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=bCgVe3SAgPxZUFuqn52%2FfDzm9iM%3D" alt="扫描全能王 2024-12-14 14.04_4.jpg" width="50%"></p>

其最大的优点是在不减少训练精度的情况下，**本地无需神经网络的训练，可以大大节省GPU资源。**

## DDPG

**Deep Deterministic Policy Gradient**

在传统DQN中，使用到了max算子计算动作，但是这只能处理离散情况，我们需要在连续情况下像一个方法取代这个算子，使其能适应连续性问题。**DDPG的解决方法是设立Actor网络，使其能直接根据状态输出连续的动作值**，其算法特点为：

*   **Actor - Critic 架构**
    *   **Actor 网络（策略网络）** ：根据状态输出行动，这里的行动一般是连续且确定的，参数用$\theta$表示，即$a = μ_{\theta}(s)$
    *   **Critic 网络（价值网络）**：用于评估 Actor 网络生成的动作的价值。它接收状态和动作作为输入，输出一个 Q 值，即$Q = Q_{\omega}(s,a)$
*   **目标网络**
    *   为了使学习过程更加稳定，DDPG 采用了目标网络。它包括目标 Actor 网络和目标 Critic 网络
*   **训练过程**
    *   **Critic 网络训练**：$y_t = r_t + γQ'(s_{t + 1}, μ'(s_{t + 1}|θ')|ω')$，损失函数为$L(ω)=(1/N)∑(Q(s_i,a_i|ω)-y_i)²)$，这里最大的区别是，将$max$算子省略了，取而代之的是Actor 网络输出的动作。主要是因为Actor 网络输出是确定的行动，为了简化计算，就进行替换。
    *   **Actor 网络训练**： $∇_θJ(θ)=(1/N)∑∇_aQ(s_i,a|ω)|_{a = μ(s_i|θ)}∇_θμ(s_i|θ)$，其实直观来解释就是**希望得到使得Q最大的动作，这是一个最大化问题**

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/91911a319b2244b89757a89f90035dc8~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg54us5oap:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzg1NTMwNjg3OTkzMzU4MiJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1747028962&x-orig-sign=%2FINU%2BCLXDV7ghq2tryAxh91%2FnqM%3D)

## DQN的改进

### 分类DQN

之前的DQN中都是输出的Q函数的数学期望，在分类DQN中输出的是Q函数的概率分布

*   **价值分布表示**：在分类 DQN 中，它将 Q - 值表示为一个离散的概率分布。假设动作价值函数的取值范围被划分为$N$个区间（例如$N=51$），对于每个动作$a$在状态$s$下，网络输出一个概率分布$p(s,a)$，其中$p(s,a)_i$表示$Q(s,a)$落在第个$i$区间的概率。
*   **训练过程**：通过最小化预测的概率分布和基于贝尔曼方程计算得到的目标概率分布之间的差异来训练网络。例如，在一个简单的游戏环境中，智能体在某个状态下采取动作后，根据获得的奖励和下一个状态的价值分布来计算目标概率分布，然后调整网络参数，使得预测的概率分布向目标概率分布靠近
*   **优势分析**：
    *   **提高估计精度**：这种方式能够更细致地捕捉 Q - 值的不确定性。相比传统 DQN 只估计一个单一的 Q - 值，分类 DQN 提供了 Q - 值可能取值的概率分布信息，在一些复杂的、具有随机因素的环境中，可以更好地估计价值。
    *   **处理风险偏好**：可以根据概率分布来考虑不同的风险偏好策略。例如，如果智能体是风险厌恶型的，可以选择具有高概率获得相对稳定回报的动作；如果是风险偏好型的，可以选择有一定概率获得高回报的动作，尽管可能伴随着高风险。

### Noisy Network

之前策略的选择是根据$\varepsilon$-greedy算法，超参数$\varepsilon$的选择对模型的结果有很大的影响，且离散的超参数也会使得训练不够稳定。一般来说我们希望理想情况下什么超参数都不要设置，于是我们可以将$\varepsilon$的思想隐蔽入神经网络中。

具体的做法是**通过在神经网络的参数或者激活函数中添加噪声来鼓励智能体在训练过程中探索更多不同的动作**，从而提高学习效率和策略的鲁棒性。

相比之下Noisy Network的优势体现在两个方面：

*   **高效探索连续动作空间**
    *   在连续动作空间场景下，$\varepsilon$-greedy 算法就显得比较笨拙。因为在连续动作空间中，随机选择一个动作可能会导致动作过于离散、跳跃，不利于找到一个平滑的最优策略。**而 Noisy Network 可以通过在网络参数或激活函数上添加噪声，自然地对连续动作进行小幅度的扰动，从而更高效地探索连续动作空间**。例如，在自动驾驶中，车辆的转向和速度控制是连续动作，Noisy Network 可以对控制动作进行微调，而$\varepsilon$-greedy 可能会导致车辆突然转向或者急加速、急刹车等不合理的动作。
*   **基于策略的探索**
    *   Noisy Network 的探索是基于当前策略的。因为噪声是添加在神经网络内部，而神经网络是学习到的策略的表示。所以，**这种探索是与策略紧密相关的，是在策略的基础上进行微调。相比之下，$\varepsilon$-greedy 的随机探索部分与当前策略没有直接关联**。例如，在一个复杂的游戏策略学习中，Noisy Network 会根据当前游戏状态和已经学到的策略，通过噪声来微调动作，而$\varepsilon$-greedy 可能会在不考虑当前策略的情况下随机选择一个动作，可能会破坏已经建立的良好策略模式

### Rainbow

Rainbow 是一种先进的深度强化学习算法，它整合了多种对 DQN（Deep Q - Network）的改进技术，目的是提升算法在强化学习任务中的性能和效率。

*   **组成技术：**
    *   **Double DQN**
    *   **Prioritized Experience Replay**
    *   **Dueling DQN**
    *   **Noisy DQN**
    *   **Categorical DQN**
*   **训练过程和优势**
    *   **训练过程**
        *   Rainbow 算法的训练过程综合了上述各种技术。在经验回放阶段，利用 Prioritized Experience Replay 抽取样本；在网络结构上，采用 Dueling DQN 和 Noisy DQN 的特点；在计算目标 Q 值时，应用 Double DQN 的方法；在 Q - 值估计上，有 Categorical DQN 的概率分布表示。通过这些技术的协同作用，不断更新网络参数，使智能体能够更好地学习策略。
