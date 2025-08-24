---
title: 《GPT图解》学习笔记
cover: /imgs/GPT2.jpg
swiper_index: 1 #置顶轮播图顺序，非负整数，数字越大越靠前
categories: # 分类
	- 大模型  # 只能由一个
tags:
    - 大模型
    - 学习
---

<meta name="referrer" content="no-referrer" />

# Chapter1：N-grams & Bag-of-words

## N-grams模型


N-grams是指将文本分割为连续的长度为N的文本片段，统计每个片段的频数以计算每个片段出现的条件概率，从而计算完整句子的出现概率。

该模型基于这样一种假设，第N个词的出现只与前面N-1个词相关，而与其它任何词都不相关，整句的概率就是各个词出现概率的乘积。这些概率可以通过直接从语料中统计N个词同时出现的次数得到。常用的是二元的Bi-Gram和三元的Tri-Gram，下面具体解释其数学实现：

对于一个有$m$个词语的语句，其条件概率为：

$$
P(w_1,w_2,w_3...w_m) = P(w_1)P(w_2|w_1)P(w_3|w_2,w_1)...P(w_m|w_{m-1},w_{m-2}...w_1)
$$

可以利用**马尔科夫假设**（当前状态只与前面n个状态相关）简化上述公式，具体体现为：

$$
P(w_m|w_1,w_2...w_{m-1}) = P(w_m|w_{m-1},w_{m-2}...w_{m-n})
$$

当n取1时，既每个状态只与前面一个状态相关，公式可以简化为

$$
P(w_1,w_2,w_3...w_m) = P(w_1)P(w_2|w_1)P(w_3|w_2)...P(w_m|w_{m-1})
$$

这就是N-grams模型的数学基础，通过语料中的统计学结果计算一句话的概率，具体应用场景可以是，根据一部分语料预测接下来的完整句子（只需要找到$P$最大的句子表达）

**举个例子**，当N取2时，对于句子“我爱你”，可以分为“我爱”，“爱你”两种文本片段，假设我们有一大堆语料文本，可以统计得到“我X”出现了100次，其中“我爱”出现了60次，则“我爱”片段条件概率为60%，那么当文本最后一个字是“我”时，我们会选择概率最大的“爱”作为后续输出。

下面通过具体例子实现N-Grams：

```python
corpus = [ "我喜欢吃苹果",
        "我喜欢吃香蕉",
        "她喜欢吃葡萄",
        "他不喜欢吃香蕉",
        "他喜欢吃苹果",
        "她喜欢吃草莓"]
def tokenize(text):
    return [char for char in text]
   #分词方式很多，也有很多处理方法，这里为了方便直接取一个字
```

然后需要统计grams的频数，设计函数`count_ngrams`统计频数，可以自定义n统计，当n=2时，片段为“我喜”，“喜欢”等等。

```python
def count_ngrams(corpus,n):
    ngrams_count = {}
    for text in corpus:
        tokens = tokenize(text)
        for i in range(len(tokens)-n+1):
            prefix = ''.join(tokens[i:i+n-1])
            token = tokens[i+n-1]
            if prefix in ngrams_count:
                if token in ngrams_count[prefix]:
                    ngrams_count[prefix][token]+=1
                else:
                    ngrams_count[prefix][token]=1
            else:
                ngrams_count[prefix]={token:1}
    return ngrams_count
bigram_counts = count_ngrams(corpus, 2) # 计算 bigram 词频
print("bigram 词频：") # 打印 bigram 词频
for prefix, counts in bigram_counts.items():
    print("{}: {}".format("".join(prefix), dict(counts)))
# 我: {'喜': 2}
# 喜: {'欢': 6}
# 欢: {'吃': 6}
# 吃: {'苹': 2, '香': 2, '葡': 1, '草': 1}
# 苹: {'果': 2}
# 香: {'蕉': 2}
# 她: {'喜': 2}
# 葡: {'萄': 1}
# 他: {'不': 1, '喜': 1}
# 不: {'喜': 1}
# 草: {'莓': 1}
```

当n=3时，片段为“我喜欢”等，前缀为“我喜”：

```python
# 我喜: {'欢': 2}
# 喜欢: {'吃': 6}
# 欢吃: {'苹': 2, '香': 2, '葡': 1, '草': 1}
# 吃苹: {'果': 2}
# 吃香: {'蕉': 2}
# 她喜: {'欢': 2}
# 吃葡: {'萄': 1}
# 他不: {'喜': 1}
# 不喜: {'欢': 1}
# 他喜: {'欢': 1}
# 吃草: {'莓': 1}
```

根据grams频数计算grams的条件概率，函数为`ngram_probabilities`

```python
def ngram_probabilities(ngrams_count):
    for prefix,tokens in ngrams_count.items():
        tokens_count_sum = sum(tokens.values())
        for token in tokens.keys():
            tokens[token] /= tokens_count_sum
    return ngrams_count

bigram_probs = ngram_probabilities(bigram_counts) # 计算 bigram 出现的概率
print("\nbigram 出现的概率 :") # 打印 bigram 概率
for prefix, probs in bigram_probs.items():
 print("{}: {}".format("".join(prefix), dict(probs)))
# 我: {'喜': 1.0}
# 喜: {'欢': 1.0}
# 欢: {'吃': 1.0}
# 吃: {'苹': 0.3333333333333333, '香': 0.3333333333333333, '葡': 0.16666666666666666, '草': 0.16666666666666666}
# 苹: {'果': 1.0}
# 香: {'蕉': 1.0}
# 她: {'喜': 1.0}
# 葡: {'萄': 1.0}
# 他: {'不': 0.5, '喜': 0.5}
# 不: {'喜': 1.0}
# 草: {'莓': 1.0}
```

最后应用场景是根据部分文本生成接下来的文本，`generate_next_token`函数可以根据前一个片段生成后一个token，具体方式就是选择条件概率最大的文本输出

需要注意的是文本生成的截止条件，如果生成的最后一个字在词表片段中不存在以它开头的前缀时，就停止，例如如果生成的最后一个字的“果”，上述`bigram_probs`中没有以“果”为前缀的片段，则终止输出。

```python
def generate_next_token(prefix,ngrams_probs):
    if prefix in ngrams_probs:
        return max(ngrams_probs[prefix],key=ngrams_probs[prefix].get)
    else:
        return None

def generate_text(prefix,n):
    ngram_counts = count_ngrams(corpus, n)
    ngrams_probs = ngram_probabilities(ngram_counts)
    for prefixs, probs in ngrams_probs.items():
        print("{}: {}".format("".join(prefixs), dict(probs)))

    text = prefix
    while(1):
        ngrams_prefix = text[-(n-1):]
        next_token = generate_next_token(ngrams_prefix,ngrams_probs)
        if next_token is None:
            break
        else:
            text = text+next_token
    return text
```

**缺点：无法捕捉距离较远文本的信息**

## Bag of Words

词袋模型是一种将文本转换为向量的方式，其只关注词语出现的次数而不关注词语的上下文关系，也就是不关心词语的顺序。

举个例子，对于一个句子`i love you very very much`，其通过词袋模型编码后的结果可能为`[1,1,1,2,1,0,0]`,这代表整个词语库共7种词语，这个句子包含了5种词语，词语的频数也有体现。

通常可以用于比较句子之间的相关性

具体实现：
构建一个词语库，统计到共21个词语：

```python
import jieba
corpus=['我特别特别喜欢看电影','这部电影真的是很好看的电影','今天天气真好是难得的好天气','我今天去看了一部电影','电影院的电影都很好看']
tokens = [list(jieba.cut(i)) for i in corpus]
def create_words_table(tokens):
    words_dict = {}
    index = 0
    for sentence in tokens:
        for word in sentence:
            if word not in words_dict:
                words_dict[word] = index
                index+=1
    return words_dict

words_dict = create_words_table(tokens)
print(words_dict)
#  {'我': 0, '特别': 1, '喜欢': 2, '看': 3, '电影': 4, '这部': 5, '真的': 6, '是': 7, '很': 8, '好看': 9, '的': 10, '今天天气': 11, '真好': 12, '难得': 13, '好': 14, '天气': 15, '今天': 16, '去': 17, '了': 18, '一部': 19, '电影院': 20, '都': 21}               
    
```

对每个句子进行向量化，具体方法为统计句子中出现了哪些词语且其频数是多少，在长度为21的向量中进行标注：

```python
def create_words_bag(words_dict,tokens):
    words_bag = []
    for sentence in tokens:
        sentence_vector = [0]*len(words_dict)
        for word in sentence:
            sentence_vector[words_dict[word]]+=1
        words_bag.append(sentence_vector)
    return words_bag

words_bag = create_words_bag(words_dict,tokens)
import numpy as np
print(np.matrix(words_bag))
# [[1 2 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 2 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 1 0 0 1 1 1 1 1 1 0 0 0 0 0 0]
#  [1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0]
#  [0 0 0 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 1 1]]
```

计算句子之间的相关性，使用余弦相似度

```python
import numpy as np
def cosine_similarity(vec1,vec2):
    return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


def similarity_matrix(words_bag):
    similarity_matrix = np.zeros((len(words_bag),len(words_bag)))
    for i in range(len(words_bag)):
        for j in range(len(words_bag)):
            similarity_matrix[i][j] = cosine_similarity(words_bag[i],words_bag[j])
    return similarity_matrix

similarity_matrix = similarity_matrix(words_bag)
print(similarity_matrix)
import matplotlib.pyplot as plt

fig,ax = plt.subplots()
cax = ax.matshow(similarity_matrix,cmap = plt.cm.Blues)
plt.show()
```
<!-- ![](https://gitee.com/leeMX111/images_for_markdown/raw/master/Figure_1.png) -->
<img src="https://gitee.com/leeMX111/images_for_markdown/raw/master/Figure_1.png" height="400px" />


**缺点：对于较大的词语库会造成高稀疏表示，且不关注词语的顺序，会损失部分语义信息**

# Chapter2：Word2Vec

**词语向量化**的一种重要方法，对比与one-hot方法，word2vec可以体现词语之间的相互关系，为后续的语义理解提供了基础。

Word2Vec的基础思想为构造一个**神经网络**，通过一些nlp任务（例如通过周围的词语得到中间词）训练这个神经网络，而我们真正需要的是这个神经网络的**隐藏层**，其可以将输入词语（可以是one-hot编码）映射到一个n维的向量，这个向量是**非稀疏**的，且经过前期的训练，这个向量可以很好的反应这个词语的语义信息。

Word2Vec训练时，一般会有两个NLP任务，既**Skip-Gram**和**CBOW**

*   **Skip-Gram**：使用中间词预测周边其他词
*   **CBOW**：使用周边其他词预测中间词


<img src="https://gitee.com/leeMX111/images_for_markdown/raw/master/v2-35339b4e3efc29326bad70728e2f469c_1440w.png" height="500px" />

tip：从实现来看，上图中的sum应该改为mean才对

训练完之后，我们并不需要整个模型，而只需要**中间层**的参数作为**词语向量化查询表**，也就是上图中两个方法的中间层。

## Skip-Gram

以Skip-Gram为例，**在实现中并非同时生成周边其他词，而是训练n次，每次生成一个词**，例如对于“我爱你”这句话，“爱”的周边词为“我”和“你”，在训练时则训练两次，分别为`“爱”->“我”`和`“爱”->“你”`，这也解释了下图中从`hidden layer`到`output layer`时是使用一样参数的原因。

最终我们只需要中间层参数$W_{V×N}$作为词语向量化表，表示词语库中共有$V$个词语，将每个词语向量化为长度为$N$的向量。

<img src="https://gitee.com/leeMX111/images_for_markdown/raw/master/联想截图_20250319222737.png" height="700px" />


实现：

```python
sentences = ['kate is teacher','mazong is boss','niuzong is boss','xiaobing is student','xiaoxue is student']
tokens = [i.split(' ') for i in sentences]
def create_words_table(tokens):
    words_dict = {}
    index = 0
    for sentence in tokens:
        for word in sentence:
            if word not in words_dict:
                words_dict[word] = index
                index+=1
    return words_dict
words_dict = create_words_table(tokens)
print(words_dict)

# {'kate': 0, 'is': 1, 'teacher': 2, 'mazong': 3, 'boss': 4, 'niuzong': 5, 'xiaobing': 6, 'student': 7, 'xiaoxue': 8}
```

构建skip-gram的数据集，此处的`windowsize`表示周围文本的长度，当其值为2时，表示中心词只能预测周围距离为1的词语，例如“kate”为中心词时，其周围词只有“is”

得到的数据集为多个数组，每个数组的第一个词为中心词，既输入，第二个词为周围词，既输出

```python
def create_skipgram_dataset(token,windowsize = 2):
    dataset=[]
    for sentence in token:
        for word_index,word in enumerate(sentence):
            for i in range(-windowsize+1,windowsize):
                if i<0 and word_index+i>=0:
                    dataset.append([word,sentence[word_index+i]])
                elif i>0 and word_index+i<=len(sentence)-1:
                    dataset.append([word,sentence[word_index+i]])
                else:
                    continue
    return dataset

dataset = create_skipgram_dataset(tokens)
print(dataset)
# [['kate', 'is'], ['is', 'kate'], ['is', 'teacher'], ['teacher', 'is'], ['mazong', 'is'], ['is', 'mazong'], ['is', 'boss'],
# ['boss', 'is'], ['niuzong', 'is'], ['is', 'niuzong'], ['is', 'boss'], ['boss', 'is'], ['xiaobing', 'is'], ['is', 'xiaobing'],
# ['is', 'student'], ['student', 'is'], ['xiaoxue', 'is'], ['is', 'xiaoxue'], ['is', 'student'], ['student', 'is']]
```

将上面的训练集中的输入变为**one-hot编码**，这样才能输入神经网络进行训练，而输出不需要是因为在计算误差时，使用`CrossEntropyLoss`函数会自动进行one-hot编码以计算误差值：

```python
import torch
def one_hot_encoding(word,words_dict):
    tensor = torch.zeros(len(words_dict))
    tensor[words_dict[word]] = 1
    return tensor
    
skip_gram_data = [[one_hot_encoding(context,words_dict),words_dict[output]] for[context,output] in dataset]
print(skip_gram_data)
# [[tensor([1., 0., 0., 0., 0., 0., 0., 0., 0.]), 1], [tensor([0., 1., 0., 0., 0., 0., 0., 0., 0.]), 0], [tensor([0., 1., 0., 0., 0., 0., 0., 0., 0.]), 2],
# [tensor([0., 0., 1., 0., 0., 0., 0., 0., 0.]), 1], [tensor([0., 0., 0., 1., 0., 0., 0., 0., 0.]), 1], [tensor([0., 1., 0., 0., 0., 0., 0., 0., 0.]), 3], 
# [tensor([0., 1., 0., 0., 0., 0., 0., 0., 0.]), 4], [tensor([0., 0., 0., 0., 1., 0., 0., 0., 0.]), 1], [tensor([0., 0., 0., 0., 0., 1., 0., 0., 0.]), 1], 
# [tensor([0., 1., 0., 0., 0., 0., 0., 0., 0.]), 5], [tensor([0., 1., 0., 0., 0., 0., 0., 0., 0.]), 4], [tensor([0., 0., 0., 0., 1., 0., 0., 0., 0.]), 1], 
# [tensor([0., 0., 0., 0., 0., 0., 1., 0., 0.]), 1], [tensor([0., 1., 0., 0., 0., 0., 0., 0., 0.]), 6], [tensor([0., 1., 0., 0., 0., 0., 0., 0., 0.]), 7], 
# [tensor([0., 0., 0., 0., 0., 0., 0., 1., 0.]), 1], [tensor([0., 0., 0., 0., 0., 0., 0., 0., 1.]), 1], [tensor([0., 1., 0., 0., 0., 0., 0., 0., 0.]), 8], 
# [tensor([0., 1., 0., 0., 0., 0., 0., 0., 0.]), 7], [tensor([0., 0., 0., 0., 0., 0., 0., 1., 0.]), 1]]
```

定义神经网络模型，此处定义了两层Linear层：

*   Linear1：input\_2\_hidden，输入大小为词表中词语个数，也就是输入词语进行one-hot编码后的长度，输出为自定义的隐藏层大小
*   Linear2：hidden\_2\_output，输入为隐藏层大小，输入长度也是one-hot编码后的长度，表示各个词语的输出概率

这里不需要定义softmax层，因为误差函数会自动进行softmax：

```python
import torch.nn as nn
class SkipGram(nn.Module):
    def __init__(self, voc_size,embedding_size) -> None:
        super(SkipGram,self).__init__()
        self.input_2_hidden =nn.Linear(voc_size,embedding_size,bias=False)
        # self.input_2_hidden = nn.Embedding(voc_size, embedding_size)
        self.hidden_2_output = nn.Linear(embedding_size,voc_size,bias=False)
    def forward(self,X):
        hidden = self.input_2_hidden(X)
        output = self.hidden_2_output(hidden)
        return output
    
skip_gram_model = SkipGram(voc_size=len(skip_gram_data[0][0]),embedding_size=2)
print(skip_gram_model)
# SkipGram(
#   (input_2_hidden): Linear(in_features=9, out_features=2, bias=False)
#   (hidden_2_output): Linear(in_features=2, out_features=9, bias=False)
# )
```

这里的`input_2_hidden`可看成一个$V×N$的矩阵，输入是一个长度为$V$的向量，那么实际上这一层做的操作即为矩阵乘法，这个向量是一个one-hot向量，矩阵乘法实际上是对这个$V×N$矩阵的查找（选出one-hot向量中为1的元素对应的行）

那么在实现时可以使用`nn.Embedding`代替线性层，这个层的本质是一个`查找表`，输入大小不需要改变，在输入时便不需要进行one-hot编码，直接输入词语对应的索引进行查找即可，简化运算。

```python
def __init__(self, voc_size,embedding_size) -> None:
    super(SkipGram,self).__init__()
    # self.input_2_hidden =nn.Linear(voc_size,embedding_size,bias=False)
    self.input_2_hidden = nn.Embedding(voc_size, embedding_size)
    self.hidden_2_output = nn.Linear(embedding_size,voc_size,bias=False)
```

模型训练：

```python
epochs = 1000
criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
lr = 0.001
import torch.optim as optim
optimizer = optim.SGD(params=skip_gram_model.parameters(),lr = lr)
loss_values = []
for epoch in range(epochs):
    loss_sum = 0
    for [one_hot_input,target] in skip_gram_data:
        X = one_hot_input.float().unsqueeze(0) 
        # tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0.]])
        y_true = torch.tensor([target], dtype=torch.long)
        # tensor([8])
        y_pred = skip_gram_model(X)
        # tensor([[-0.1776, -0.1084, 0.0309, 0.0138, 0.2688, -0.0034, -0.2324, 0.1325,
        #          0.1417]], grad_fn= < MmBackward0 >)
        loss = criterion(y_pred,y_true)
        loss_sum +=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 100 == 0: # 输出每 100 轮的损失，并记录损失
      print(f"Epoch: {epoch+1}, Loss: {loss_sum/len(skip_gram_data)}")
      loss_values.append(loss_sum / len(skip_gram_data))
      
#使用nn.Embedding的训练过程，直接输入索引即可
#for epoch in range(epochs):
#     loss_sum = 0
#     for [center_word,target] in dataset:
#         X = torch.tensor(words_dict[center_word],dtype=torch.long).unsqueeze(0)
#         y_true = torch.tensor([words_dict[target]], dtype=torch.long) # 将周围词转换为索引值
#         y_pred = skip_gram_model(X)
#         loss = criterion(y_pred,y_true)
#         loss_sum +=loss.item()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     if (epoch+1) % 100 == 0: # 输出每 100 轮的损失，并记录损失
#       print(f"Epoch: {epoch+1}, Loss: {loss_sum/len(skip_gram_data)}")
#       loss_values.append(loss_sum / len(skip_gram_data))
   
```

<!-- ![](https://gitee.com/leeMX111/images_for_markdown/raw/master/skip_gram.png) -->
<img src="https://gitee.com/leeMX111/images_for_markdown/raw/master/skip_gram.png" height="450px" />

训练完之后，我们需要的是隐藏层的参数，即`skip_gram_model.input_2_hidden.weight`，这是一个9×2的矩阵，表示将9个词语变为了长度为2的向量。

```python
import matplotlib.pyplot as plt # 导入 matplotlib
# 绘制二维词向量图
plt.rcParams["font.family"]=['SimHei'] # 用来设定字体样式
plt.rcParams['font.sans-serif']=['SimHei'] # 用来设定无衬线字体样式
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
plt.plot(range(1, epochs//100 + 1), loss_values) # 绘图
plt.title(' 训练损失曲线 ') # 图题
plt.xlabel(' 轮次 ') # X 轴 Label
plt.ylabel(' 损失 ') # Y 轴 Label
plt.show() # 显示图

print(skip_gram_model.input_2_hidden.weight)
# tensor([[-0.4476,  0.6655, -0.7532, -0.6657, -1.0122, -0.4196, -0.6324, -0.5355,
#          -0.4030],
#         [ 1.0320, -0.5229,  0.8602,  0.9287,  1.0592,  0.9380,  0.9937,  1.3663,
#           0.9899]], requires_grad=True)
```

可以将每个词语的向量表示汇出：

```python
for word in words_dict:
    print(word)
    print(skip_gram_model.input_2_hidden.weight[:,words_dict[word]].detach().numpy())

fig, ax = plt.subplots() 
for word in words_dict:
    vec = skip_gram_model.input_2_hidden.weight[:,words_dict[word]].detach().numpy()
    ax.scatter(vec[0], vec[1]) # 在图中绘制嵌入向量的点
    ax.annotate(word, (vec[0], vec[1]), fontsize=12) # 点旁添加单词标签
plt.title(' 二维词嵌入 ') # 图题
plt.xlabel(' 向量维度 1') # X 轴 Label
plt.ylabel(' 向量维度 2') # Y 轴 Label
plt.show() # 显示图
```

<!-- ![](https://gitee.com/leeMX111/images_for_markdown/raw/master/skip_gram2.png) -->
<img src="https://gitee.com/leeMX111/images_for_markdown/raw/master/skip_gram2.png" height="450px" />

## CBOW

CBOW是用周围词预测中间词，这里需要注意的是，$C$个中间词是同时输入的，那么可以把输入矩阵看做$I_{C×V}$，隐藏层矩阵为$W_{V×N}$，则输出大小为$C×N$，这里需要做一次**平均操作**，使得输出大小变为$1×N$以输入后续的线性层。

<!-- ![](https://gitee.com/leeMX111/images_for_markdown/raw/master/1231.png) -->
<img src="https://gitee.com/leeMX111/images_for_markdown/raw/master/1231.png" height="700px" />

实现过程与skip_gram类似，只需要进行部分调整：
在生成数据集时，需要实现多对一的数据集：

```python
def create_CBOW_dataset(token, windowsize=2):
    dataset = []
    for sentence in token:
        for word_index, word in enumerate(sentence):
            context = []
            for i in range(-windowsize, windowsize+1):
                if (i < 0 and word_index + i >= 0) or (i > 0 and word_index + i <= len(sentence) - 1):
                    context.append(sentence[word_index + i])
                else:
                    continue
            dataset.append([word,context])
    return dataset


dataset = create_CBOW_dataset(tokens)
print(dataset)
# [['kate', ['is', 'teacher']], ['is', ['kate', 'teacher']], ['teacher', ['kate', 'is']], ['mazong', ['is', 'boss']], ['is', ['mazong', 'boss']], 
#  ['boss', ['mazong', 'is']], ['niuzong', ['is', 'boss']], ['is', ['niuzong', 'boss']], ['boss', ['niuzong', 'is']], ['xiaobing', ['is', 'student']], 
#  ['is', ['xiaobing', 'student']], ['student', ['xiaobing', 'is']], ['xiaoxue', ['is', 'student']], ['is', ['xiaoxue', 'student']], ['student', ['xiaoxue', 'is']]]
```

将数据集进行one-hot编码，且使用`torch.stack`将多个输入进行合并，与`torch.cat`的区别在于`torch.stack`会新增一个维度来进行拼接，这使得它在构建具有批次维度等场景下非常有用，比如在深度学习中构建批次数据时，将多个样本张量堆叠起来。

```python
def create_CBOW_data(dataset):
    CBOW_data = []
    for [center_word,context] in dataset:
        context_one_hot = torch.stack([one_hot_encoding(word,words_dict) for word in context]).float()
        CBOW_data.append([torch.tensor(words_dict[center_word],dtype=torch.long),context_one_hot])
    return CBOW_data
# print(skip_gram_data)
CBOW_data = create_CBOW_data(dataset)
# [[tensor(0), tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0.],[0., 0., 1., 0., 0., 0., 0., 0., 0.]])], 
#  [tensor(1), tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0.],[0., 0., 1., 0., 0., 0., 0., 0., 0.]])],
```

定义网络结构，`input_2_hidden`层将多个输入同时计算，得到`2×embedding_size`的结果，然后中间加入了一个`mean`层，将输出变为1维，值得注意的是，网络的输入输出大小没变。

```python
class CBOW(nn.Module):
    def __init__(self, voc_size, embedding_size) -> None:
        super(CBOW, self).__init__()
        self.input_2_hidden = nn.Linear(voc_size, embedding_size, bias=False)
        self.hidden_2_output = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, X):
        embedding = self.input_2_hidden(X)
        # tensor([[-0.2415, 0.2611],
        #         [0.2320, -0.3655]], grad_fn= < MmBackward0 >)
        hidden = torch.mean(embedding,dim=0)
        # tensor([-0.0047, -0.0522], grad_fn= < MeanBackward1 >)
        output = self.hidden_2_output(hidden.unsqueeze(0))
        return output
```

**Word2Vec的局限性**

*   词向量是固定的，无法处理“一词多义”的情况
*   无法处理未知词汇

# Chapter3：NPLM模型

在NPLM（Neural Probabilistic Language Model）模型中

在

<!-- ![](https://gitee.com/leeMX111/images_for_markdown/raw/master/nplm.png) -->
<img src="https://gitee.com/leeMX111/images_for_markdown/raw/master/nplm.png" height="700px" />

**因此，神经概率语言模型**依然是一个概率语言模型，它是通过**神经网络**来计算**概率语言模型**中的每个参数。

相比于N-gram语言模型，**神经概率语言模型**有以下优点：

1.  **单词之间的相似性可以通过词向量来体现**(相比神经语言模型本身，作为其副产品的词向量反而是更大的惊喜)
2.  **自带平滑处理**

> 在某种程度上，可以说Word2Vec和NPLM在一些方面有相似之处，但它们在设计和应用上仍有一些显著的区别。以下是它们的一些相似点和差异：
>
> **相似点**：
>
> 1.  **基于神经网络**：Word2Vec和NPLM都是基于神经网络的模型，用于学习词向量和处理自然语言文本。
> 2.  **词嵌入**：两者都旨在将单词映射到连续向量空间中，以便捕捉单词之间的语义关系。
>
> **差异点**：
>
> 1.  **预测任务**：Word2Vec的预测任务主要是通过上下文单词预测目标单词（Skip-gram）或通过目标单词预测上下文单词（CBOW），而NPLM是一种神经网络语言模型，主要任务是预测下一个单词出现的概率。
> 2.  **上下文考虑**：NPLM在训练时考虑了前面n-1个单词的上下文信息，以便更好地捕捉长距离依赖关系，而Word2Vec主要关注词与词之间的语义关系，对于长距离依赖的处理不如NPLM。
> 3.  **应用领域**：由于任务和设计的差异，Word2Vec通常用于词向量学习、词义相似度计算等任务，而NPLM更适用于语言建模等需要考虑长距离依赖的任务。

实现：

```python
sentences = ["我 非常 喜欢 玩具", "我 爱 爸爸", "我 讨厌 挨打"]
words_list = list(set(" ".join(sentences).split()))
words_dict =  {word:index for index,word in enumerate(words_list)}
print(words_dict)
# {'我': 0, '喜欢': 1, '爱': 2, '爸爸': 3, '讨厌': 4, '挨打': 5, '玩具': 6}
idx_to_word = {idx: word for idx, word in enumerate(words_dict)}
```

构建训练集，设置`make_batch`生成一个batch训练集，在这里一个batch包含两份数据，`n_step`表示一次性输入模型的token数量，在这里设置为2，也就是说用前面2个token预测下一个token

```python
import torch # 导入 PyTorch 库
import random # 导入 random 库
batch_size = 2 # 每批数据的大小
n_step = 2
def make_batch(n_step):
    input_batch = []  # 定义输入批处理列表
    target_batch = []  # 定义目标批处理列表
    selected_sentences = random.sample(sentences, batch_size) # 随机选择句子
    for sen in selected_sentences:  # 遍历每个句子
        word = sen.split()  # 用空格将句子分隔成多个词
        # 将除最后一个词以外的前面n_step个词的索引作为输入
        input = [words_dict[n] for n in word[-n_step-1:-1]]  # 创建输入数据
        # 将最后一个词的索引作为目标
        target = words_dict[word[-1]]  # 创建目标数据
        input_batch.append(input)  # 将输入添加到输入批处理列表
        target_batch.append(target)  # 将目标添加到目标批处理列表
    input_batch = torch.LongTensor(input_batch) # 将输入数据转换为张量
    target_batch = torch.LongTensor(target_batch) # 将目标数据转换为张量
    return input_batch, target_batch  # 返回输入批处理和目标批处理数据
input_batch, target_batch = make_batch(n_step) # 生成批处理数据
print(" 输入批处理数据：",input_batch)  # 打印输入批处理数据
# 将输入批处理数据中的每个索引值转换为对应的原始词
input_words = []
for input_idx in input_batch:
    input_words.append([idx_to_word[idx.item()] for idx in input_idx])
print(" 输入批处理数据对应的原始词：",input_words)
print(" 目标批处理数据：",target_batch) # 打印目标批处理数据
# 将目标批处理数据中的每个索引值转换为对应的原始词
target_words = [idx_to_word[idx.item()] for idx in target_batch]
print(" 目标批处理数据对应的原始词：",target_words)
# #
# 输入批处理数据： tensor([[1, 5],
#                         [6, 0]])
# 输入批处理数据对应的原始词： [['我', '爱'], ['非常', '喜欢']]
# 目标批处理数据： tensor([7, 3])
# 目标批处理数据对应的原始词： ['爸爸', '玩具']
```

构建模型，这里的重点是第一个线性层的输入大小为`n_step * embedding_size`,也就是将`n_step`个输入进行`embedding`后拼接起来再输入线性层（区别于CBOW，其方法为多个输入编码后取平均）

```python
import torch.nn as nn # 导入神经网络模块
# 定义神经概率语言模型（NPLM）
class NPLM(nn.Module):
    def __init__(self):
        super(NPLM, self).__init__()
        self.C = nn.Embedding(voc_size, embedding_size) # 定义一个词嵌入层
        # 第一个线性层，其输入大小为 n_step * embedding_size，输出大小为 n_hidden
        self.linear1 = nn.Linear(n_step * embedding_size, n_hidden)
        # 第二个线性层，其输入大小为 n_hidden，输出大小为 voc_size，即词汇表大小
        self.linear2 = nn.Linear(n_hidden, voc_size)
    def forward(self, X):  # 定义前向传播过程
        # 输入数据 X 张量的形状为 [batch_size, n_step]
        X = self.C(X)  # 将 X 通过词嵌入层，形状变为 [batch_size, n_step, embedding_size]
        X = X.view(-1, n_step * embedding_size) # 形状变为 [batch_size, n_step * embedding_size]
        # 通过第一个线性层并应用 ReLU 激活函数
        hidden = torch.tanh(self.linear1(X)) # hidden 张量形状为 [batch_size, n_hidden]
        # 通过第二个线性层得到输出
        output = self.linear2(hidden) # output 形状为 [batch_size, voc_size]
        return output # 返回输出结果
#
```

模型训练及预测

```python
n_hidden = 2 # 隐藏层大小
embedding_size = 2 # 词嵌入大小
voc_size = len(words_dict)
model = NPLM() # 创建神经概率语言模型实例
print(' NPLM 模型结构：', model) # 打印模型的结构
#
import torch.optim as optim # 导入优化器模块
criterion = nn.CrossEntropyLoss() # 定义损失函数为交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.1) # 定义优化器为 Adam，学习率为 0.1
# 训练模型
for epoch in range(5000): # 设置训练迭代次数
   optimizer.zero_grad() # 清除优化器的梯度
   input_batch, target_batch = make_batch(n_step) # 创建输入和目标批处理数据
   output = model(input_batch) # 将输入数据传入模型，得到输出结果
   loss = criterion(output, target_batch) # 计算损失值
   if (epoch + 1) % 1000 == 0: # 每 1000 次迭代，打印损失值
     print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
   loss.backward() # 反向传播计算梯度
   optimizer.step() # 更新模型参数



# 进行预测
input_strs = [['我', '讨厌'], ['我', '喜欢']]  # 需要预测的输入序列
# 将输入序列转换为对应的索引
input_indices = [[words_dict[word] for word in seq] for seq in input_strs]
# 将输入序列的索引转换为张量
input_batch = torch.LongTensor(input_indices)
# 对输入序列进行预测，取输出中概率最大的类别
predict = model(input_batch).data.max(1)[1]
# 将预测结果的索引转换为对应的词
predict_strs = [idx_to_word[n.item()] for n in predict.squeeze()]
for input_seq, pred in zip(input_strs, predict_strs):
   print(input_seq, '->', pred)  # 打印输入序列和预测结果

# ['我', '讨厌'] -> 挨打
# ['我', '喜欢'] -> 爸爸
```

# Chapter4：Seq2Seq

部分引用：<https://zhuanlan.zhihu.com/p/147310766>

Seq2Seq本意为`序列—>序列`的一种模型，解决的是一些序列转换的问题，例如机器翻译等等，基本思想是**将输入序列编码为一些向量表示，然后再通过解码获奖这些信息转换为输出序列**

Seq2Seq一般包含两个部分：

*   Encoder：将输入序列进行编码，映射到一个向量空间中，一般会采用`embedding`+`rnn`(或`lstm`等)，输入有两个：输入序列和初始化的`hidden`
*   Decoder: 接收编码器的最后的`hidden`，并将其解码为需要的序列。解码器也有输入序列，在训练时和预测时有不同：

在预测时，将编码器的`hidden`当成解码器的初始隐藏层，并在第一个时间步输入一个**开始信号**，一般为`<sos>`，然后将上一时刻的输出作为下一时刻的输入，这很好理解，根据上一时刻说了什么推断下一时刻要说什么很合理。

![](https://gitee.com/leeMX111/images_for_markdown/raw/master/v2-6c73bb4f24b93d8a640fea0ef60d1919_1440w.jpg)

但是在训练时不能像测试时一样，在一开始时，模型是混乱的，利用模型的输出，将上一时刻的输出作为下一时刻的输入是没有意义的，模型的进步会非常缓慢，所以需要**教师强制（Teacher Forcing)** 机制

训练时，解码器的输入和期望输出基本一致，但是错开一个时间步。教师强制是一种Seq2Seq在训练时的监督方法，decoder在运行是一步一步输出，可以看成一个生成模型，教师强制指的是在训练时，对其每一步都基于正确的引导，使得其能快速的更新参数

例如在一次训练中，解码器期望输出为`I LOVE YOU <eos>`,那么其输入为`<sos> I LOVE YOU`,在第一个时间步，解码器输入`<sos>`，其期望输出为`I`，在第二个时间步，输入为`I`(尽管在第一个时间步的实际输出可能不是`I`)，期望输出为`LOVE`,以此类推，**就好像每一步都有一个老师拿着上一时刻的正确答案引导你下一时刻做出正确的选择。**

![](https://gitee.com/leeMX111/images_for_markdown/raw/master/222.jpg)

**具体实现**

先准备数据，每一个数据包括**编码器输入**，**解码器输入**和**期望解码器输出**，这里模拟一个中文翻译英文的场景：

```python
sentences = [
    ['咖哥 喜欢 小冰', '<sos> KaGe likes XiaoBing', 'KaGe likes XiaoBing <eos>'],
    ['我 爱 学习 人工智能', '<sos> I love studying AI', 'I love studying AI <eos>'],
    ['深度学习 改变 世界', '<sos> DL changed the world', 'DL changed the world <eos>'],
    ['自然 语言 处理 很 强大', '<sos> NLP is so powerful', 'NLP is so powerful <eos>'],
    ['神经网络 非常 复杂', '<sos> Neural-Nets are complex', 'Neural-Nets are complex <eos>']]
word_list_cn, word_list_en = [], []  # 初始化中英文词汇表
# 遍历每一个句子并将单词添加到词汇表中
for s in sentences:
    word_list_cn.extend(s[0].split())
    word_list_en.extend(s[1].split())
    word_list_en.extend(s[2].split())
# 去重，得到没有重复单词的词汇表
word_list_cn = list(set(word_list_cn))
word_list_en = list(set(word_list_en))
# 构建单词到索引的映射
word2idx_cn = {w: i for i, w in enumerate(word_list_cn)}
word2idx_en = {w: i for i, w in enumerate(word_list_en)}
# 构建索引到单词的映射
idx2word_cn = {i: w for i, w in enumerate(word_list_cn)}
idx2word_en = {i: w for i, w in enumerate(word_list_en)}
# 计算词汇表的大小
voc_size_cn = len(word_list_cn)
voc_size_en = len(word_list_en)
print(" 句子数量：", len(sentences)) # 打印句子数
print(" 中文词汇表大小：", voc_size_cn) # 打印中文词汇表大小
print(" 英文词汇表大小：", voc_size_en) # 打印英文词汇表大小
print(" 中文词汇到索引的字典：", word2idx_cn) # 打印中文词汇到索引的字典
print(" 英文词汇到索引的字典：", word2idx_en) # 打印英文词汇到索引的字典
# 句子数量： 5
# 中文词汇表大小： 18
# 英文词汇表大小： 20
# 中文词汇到索引的字典： {'人工智能': 0, '语言': 1, '深度学习': 2, '强大': 3, '很': 4, '复杂': 5, '喜欢': 6, '改变': 7,
#                        '处理': 8, '自然': 9, '小冰': 10, '神经网络': 11, '学习': 12, '我': 13, '咖哥': 14, '爱': 15,
#                        '世界': 16, '非常': 17}
# 英文词汇到索引的字典： {'I': 0, '<eos>': 1, 'are': 2, 'powerful': 3, 'changed': 4, 'AI': 5, 'Neural-Nets': 6, 'NLP': 7,
#                        '<sos>': 8, 'XiaoBing': 9, 'KaGe': 10, 'studying': 11, 'the': 12, 'likes': 13, 'love': 14,
#                        'is': 15, 'DL': 16, 'complex': 17, 'world': 18, 'so': 19}

import torch
import random
def make_data(sentences):
    sentence = random.choice(sentences)
    encoder_input = torch.LongTensor([word2idx_cn[word]for word in sentence[0].split()]).unsqueeze(0)
    decoder_input = torch.LongTensor([word2idx_en[word]for word in sentence[1].split()]).unsqueeze(0)
    encoder_output = torch.LongTensor([word2idx_en[word]for word in sentence[2].split()]).unsqueeze(0)
    return encoder_input,decoder_input,encoder_output

print(make_data(sentences))
# (tensor([[16,  8, 12]]), tensor([[ 3,  0, 12,  8, 17]]), tensor([[ 0, 12,  8, 17, 14]]))
```

构建**encoder**和**decoder**

*   encoder:
    *   输入大小为输入中文词库的大小，在这里是18，`hidden_size`人为定义，为128,输出大小也为128
    *   主要包含一层`embedding`，将输入词语映射到向量中，然后带入`rnn`层进行编码
*   decoder
    *   `embedding`将解码器输入部分进行编码，输出大小为输出英文词库的大小，这里为20
    *   将`embedding`输出和从解码器过来的`hidden`输入rnn
    *   输出接一个线性层，线性层输出大小也为输出英文词库的大小，这里为20，代表每个英文单词的概率

```python
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)

    def forward(self, encoder_input, hidden):
        embedding = self.embedding(encoder_input)
        output, hidden = self.rnn(embedding, hidden)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, decoder_input, hidden):
        embedding = self.embedding(decoder_input)
        output, hidden = self.rnn(embedding, hidden)
        output = self.linear(output)
        return output, hidden


voc_size = len(word_list_en)
hidden_size = 128
encoder = Encoder(len(word_list_cn), hidden_size)
decoder = Decoder(hidden_size, len(word_list_en))

print(encoder, decoder)
# Encoder(
#   (embedding): Embedding(18, 128)
#   (rnn): RNN(128, 128, batch_first=True)
# ) Decoder(
#   (embedding): Embedding(20, 128)
#   (rnn): RNN(128, 128, batch_first=True)
#   (linear): Linear(in_features=128, out_features=20, bias=True)
# )
```

**构建Seq2Seq模型**

*   `forward`函数用于训练
*   `predict`函数用于测试，在《GPT图解》中没有这个函数，取而代之的是输入`<sos><sos><sos>...<eos>`,这样每一次解码器的输入都是`<eos>`，效果是比较差的。
    *   正确的做法应该是将解码器每一步的输出当场下一步的输入

```python
class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,encoder_input,hidden,decoder_input):
        encoder_output,hidden = self.encoder(encoder_input,hidden)
        decoder_output,_ = self.decoder(decoder_input,hidden)
        return decoder_output

    def predict(self, input_seq, start_token, max_length,end_token):
        batch_size = input_seq.size(0)
        hidden = torch.zeros(1, batch_size, self.decoder.hidden_size)  # 初始化隐藏状态
        encoder_output, hidden = self.encoder(input_seq, hidden)
        decoder_input = torch.tensor([start_token] * batch_size).unsqueeze(1)
        # 解码器的第一回合的输入还是start_token，也就是<sos>

        output_seq = []

        for i in range(max_length):
            # 解码器前向传播
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            # 取出输出中概率最大的词的序号
            decoder_input = decoder_output.data.max(2,keepdim=True)[1].squeeze(1)
            # 当前的输入作为下一步的输入
            output_word = int(decoder_input[0][0].detach())
            output_seq.append(output_word)
            # 如果输出是<eos>则结束预测
            if output_word == end_token:
                break

            return output_seq
```

**训练和预测**

```python
def train_seq2seq(model,sentences,epochs,optimizer,loss_func):
    for epoch in range(epochs):
        encoder_input,decoder_input,encoder_output = make_data(sentences)
        hidden = torch.zeros(1,encoder_input.size(0),hidden_size)#torch.Size([1, 1, 128])
        optimizer.zero_grad()
        output = model(encoder_input,hidden,decoder_input)
        loss = loss_func(output.view(-1,output.size(2)),encoder_output.squeeze(0))
        if epoch%20==0:
            print(epoch,loss)
        loss.backward()
        optimizer.step()

import torch
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
loss_func = nn.CrossEntropyLoss()
train_seq2seq(model,sentences,200,optimizer,loss_func)

def test_seq2seq(model,test_input,word2index_cn = word2idx_cn,word2index_en = word2idx_en):
    encoder_input = torch.LongTensor([word2index_cn[i] for i in test_input.split()]).unsqueeze(0)
    hidden = torch.zeros(1, encoder_input.size(0), hidden_size)  # torch.Size([1, 1, 128])
    start_token = word2index_en['<sos>']
    end_token = word2index_en['<eos>']
    print(start_token)
    predict  = model.predict(encoder_input,start_token,10,end_token=end_token)

    predict = [word_list_en[i] for i in predict]
    print(predict)


test_seq2seq(model,'深度学习 很 强大')
```

**Seq2Seq的局限性**

*   编码器将其最后一个状态输出交由解码器进行解码，这要求编码器最后一个状态中包含所有信息，这其实是非常困难的，尤其是输入序列较长的时候，可能会存在信息丢失问题和梯度消失问题
*   编码时序列被编码成了固定长度的向量，解码过程中模型难以关注到序列的重要信息。

# Chapter5：注意力机制

## 点积注意力（Dot-Product Attention）

点积注意力的公式为：

$$
out = softmax(Q·K^T)·V
$$

为了便于理解，这里不展开Q，K，V的描述，先从两个向量解释：

*   对于张量`x1`(*batch\_size, sep\_len1, feature\_dim*)和`x2`(*batch\_size, sep\_len2, feature\_dim*)
*   `x1`与`x2`（转置）进行**点积**，得到初始权重`raw_weights`，大小为(*batch\_size, sep\_len1, sep\_len2*)
*   使用`softmax`对其**行**进行归一化，得到归一化后注意力权重`atten_weights`，大小不变
*   最后跟x2进行加权求和，也就是相乘，得到`atten_out`(*batch\_size, sep\_len1, feature\_dim*)，这就是x1对于x2的**点积注意力**

第2步中的点积实际上是提取`x1`和`x2`不同元素之间的相似度，可以想象`x1`是“衣服感兴趣向量”，例如代表（质量、品牌、美观），值为(0.8,0.1,0.1)，表示其最需要质量。`x2`为“衣服实际状态向量”，由于两个向量*feature\_dim*是一样的，其也代表（质量、品牌、美观），值为（98,1,1），这件衣服的重点在于质量，那么`x1`与`x2`点积结果就会很大。

在`raw_weights`(*batch\_size, sep\_len1, sep\_len2*)中，每个元素表示`x1`中的每个元素对于`x2`中的每个元素的**相似程度**

```python
import torch
import torch.nn.functional as F

x1 = torch.randn(2,3,4)
x2 = torch.randn(2,5,4)

raw_weights = torch.bmm(x1,x2.transpose(1,2)) #(2,3,5)
```

> `torch.bmm`表示批量矩阵乘法（Batch Matrix - Multiplication），它主要用于处理小批次（batch）的矩阵乘法运算场景，其输入需要为三维，而`torch.matmul`也是矩阵乘法，但是其更加灵活，可以处理2维，但是为了代码严谨性，在确定为批量矩阵乘法的情况下，使用`torch.bmm`可以提高代码可读性。

第3步，softmax进行归一化，意义不变，只是将相似程度变成了类似概率值形式，例如下面第一行是 *\[0.7248, 0.1541, 0.0420, 0.0030, 0.0761]*，表示`x1`的第一个元素对`x2`第一个元素关注度最高，有0.7248，对`x2`第二个元素关注度只有0.1541

```python
atten_weights = F.softmax(raw_weights,dim=2)
# tensor([[[0.7248, 0.1541, 0.0420, 0.0030, 0.0761],
#          [0.1541, 0.0578, 0.6464, 0.0147, 0.1270],
#          [0.4476, 0.0523, 0.0898, 0.3739, 0.0364]],
# 
#         [[0.6825, 0.2509, 0.0154, 0.0353, 0.0159],
#          [0.0075, 0.2422, 0.6167, 0.0582, 0.0755],
#          [0.1848, 0.0666, 0.5354, 0.1908, 0.0223]]])
```

第4步本质上是根据关注度，或者说权重，提取`x2`中的关键信息，因为**注意力机制的目的就是格局x1中各个位置的关注程度提取x2中的关键信息**，还是那个衣服的例子，假设x1的第一个元素对于x2三个元素的权重分布为0.8,0.1,0.1,而x2三个元素在“质量”这个特征上的值为100,1,1,那么x1第一个元素关于x2点积注意力中关于“质量”部分的值为80.2，**这是包含了x1和x2所有信息的结果。**

这样的意义在于，out中的词被编码之后的信息，就不再仅仅包含自身或只学习了周围几个词的信息，而是整合了整个序列的全部。

**其实本质上来说，注意力机制的目的是根据`x1`的各个位置的关注程度来提取`x2`中的关键信息**


<img src="https://gitee.com/leeMX111/images_for_markdown/raw/master/pic.png" alt="pic" style=" height: 300px !important;">


```python
atten_out = torch.bmm(atten_weights,x2) #(2,3,4)
# tensor([[[ 2.8867,  0.5762, -0.1491,  0.6604],
#          [ 0.1754,  0.7851,  0.5922,  1.0507],
#          [ 1.0783, -0.1699, -0.1212,  0.4201]],
# 
#         [[-0.0468,  0.1708,  1.3725,  0.0546],
#          [ 0.6665,  1.8111,  0.8774,  0.6866],
#          [ 0.7801,  1.3610,  0.9799,  0.5164]]]) torch.Size([2, 3, 4])
```

## 缩放点积注意力（Scaled Dot-Product Attention）

缩放点积注意力公式为：

$$
out = softmax(\frac{Q·K^T}{\sqrt{d}})·V
$$

**与点积注意力的最大差别是，在第2步之前和在第3步之后，将点积结果除以一个缩放因子，一般是输入特征维度的平方根。**

> 因为许多时候特征维度很大的时候，点积结果会很大，除以缩放因子$\sqrt{d}$（其中$d$是输入向量的维度）的目的是为了缓解上述问题。通过缩放注意力分数，使得 Softmax 函数的输入值不会因为维度过高而出现过大的差异，减轻了梯度消失的问题。

**softmax 反正会将结果归一化，为什么还需要除以缩放因子呢**

>1. **Softmax 函数的特点和潜在问题**
Softmax 函数的公式为$\text{Softmax}(x_{i}) = \frac{e^{x_{i}}}{\sum_{j}e^{x_{j}}}$，它将输入的数值向量转换为一个概率分布向量，其中每个元素都在 0 到 1 之间，且所有元素之和为 1。
当输入的数值向量中元素之间的差异较大时，例如在点积注意力机制中，如果查询向量和键向量的维度较高，点积的结果（未缩放的注意力分数）可能会出现较大的值。假设未缩放的注意力分数为$x = [x_{1}, x_{2}, \cdots, x_{n}]$，当其中某个$x_{i}$很大时，经过 Softmax 计算后，$e^{x_{i}}$会在分母$\sum_{j}e^{x_{j}}$中占主导地位。
这会导致 Softmax 函数的输出概率分布出现极端情况，大部分概率集中在最大值对应的位置，其他位置的概率接近于 0。在这种情况下，在反向传播过程中，梯度会变得非常小（梯度消失现象），使得模型难以有效地学习到不同位置之间的关系。
>2. **缩放因子的作用**
除以缩放因子$\sqrt{d_k}$（其中$d_{k}$是键向量的维度）的目的是为了缓解上述问题。通过缩放注意力分数，使得 Softmax 函数的输入值不会因为维度过高而出现过大的差异。
例如，在高维空间中，点积的结果可能会随着维度的增加而增大。假设未缩放的注意力分数与维度$d_{k}$成线性关系，当除以$\sqrt{d_{k}}$后，能够将注意力分数的大小控制在一个相对合理的范围内，避免 Softmax 函数的输出过于极端。
这样，在反向传播过程中，梯度能够更有效地传播，模型可以更好地学习到每个位置的信息对最终结果的贡献，尤其是在处理长序列和高维向量的场景下，这有助于提高模型的性能和训练效率。
```python
import torch
import torch.nn.functional as F

x1 = torch.randn(2,3,4)
x2 = torch.randn(2,5,4)

raw_weights = torch.bmm(x1,x2.transpose(1,2)) #(2,3,5)

scaling_factor = x1.size(-1)**0.5#2

atten_weights = F.softmax(raw_weights/scaling_factor,dim=2)

atten_out = torch.bmm(atten_weights,x2) #(2,3,4)
```

## 编码器-解码器注意力

将注意力机制运用到编码器-解码器架构中，上述文中的`x1`和`x2`分别对应**解码器**和**编码器**：

*   `x1`：对应**解码器**的各个时间步的隐藏状态。
*   `x2`：对应**编码器**的各个时间步的隐藏状态。

**大概步骤为：**

*   得到将编码器的输出`encoder_output`（这里可以表征为解码器每一个时间步的状态，大小为`(batch_size, seq_len, encoder_out_size)`）
*   将这个输出和解码器的每一个时间步的`rnn`的输出`decoder_rnn_output`进行**注意力**计算，得到`attention_output`
*   最后将`attention_output`和`decoder_rnn_output`拼接起来输入线性层，得到最终输出。

**下面是对上述方法的实现：**

首先是定义`Attenton`方法，实现了之前介绍的注意力机制

```python
class Attenion(nn.Module):
    def __init__(self):
        super(Attenion,self).__init__()

    def forward(self,encoder_context,decoder_context):
        raw_weights = torch.bmm(decoder_context, encoder_context.transpose(1, 2))  # 
        atten_weights = F.softmax(raw_weights, dim=2)

        atten_out = torch.bmm(atten_weights, encoder_context)  
        return atten_out,atten_weights
```

重构`Decoder`部分，主要的变化是

*   增加了`attention`部分，将`rnn`的输出和解码器的输出作为其输入
*   将`attention`的输出和`rnn`输出拼接起来输入线性层

```python
class DecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.attention = Attenion()
        self.linear = nn.Linear(2 * hidden_size, output_size)

    def forward(self, decoder_input, hidden, encoder_output):
        embedding = self.embedding(decoder_input)
        rnn_output, hidden = self.rnn(embedding, hidden)
        attention_output, attention_weights = self.attention(encoder_output,rnn_output)
        decoder_output = self.linear(torch.cat((rnn_output,attention_output),dim=-1))
        return decoder_output, hidden ,  attention_weights
```

其余实现部分与前文类似，此处不赘述

在大部分的关于注意力的文章中都会对于**Q，K，V**进行描述，以展开注意力的介绍，回忆缩放点积注意力的公式：

$$
out = softmax(\frac{Q·K^T}{\sqrt{d}})·V
$$

*   **Q：Query**，查询
*   **K：Key**，键
*   **V：Value**，值

这里通过一个通俗的例子说明这个过程的意义，假如我们需要去图书馆看书。**Q**表示我们需要的书的清单，**K**表示图书馆的书的编号，**V**表示书的具体内容。首先我们会根据我们的清单和编号去确定我需要的书的编号，然后根据这个结果去找书，最终拿到需要的书的结果。

而对于**编码器解码器**的过程，**Q，K，V**的对应如下所示：

*   编码器的隐藏状态：**K，V**
*   解码器的隐藏状态：**Q**

本质上，在Seq2Seq中运用Attention的意义就是，**可以得到在当前解码器的输入下，它对编码器的哪些信息更感兴趣，最后根据结果提取这个感兴趣的内容。** 这样的好处是，能使得解码器的输出再任意时刻不再单一依赖于编码器的最后隐藏层，且也可以过滤到许多无效信息。

例如对于一个翻译任务：`我爱你-->I Love You`，某一时刻解码器的输入是`I`,那么**Q，K，V**的通俗意义如下：

*   **Q**：`I`对应的一些表征
*   **K**：在解码器的输出中，对于`我爱你`的表征
*   **V**：同K，虽然跟K一样，但是其意义不一样

那么在这个过程中，Q和K点积会得到当前情况下（输入为`I`），解码器对`我爱你`的表征哪些比较感兴趣（也许是`我`和`爱`），这里得到的是一个感兴趣的概率，然后再乘以V，最终得到感兴趣的内容。

## 多头自注意力（Multi-head Attention）

**自注意力：**

在之前的做法中，Q，K，V向量可能是不同的来源，**而自注意力则是表示对同一个输入进行不同的线性变换，得到*Q，K，V*向量，然后再应用缩放点积注意力即可**

而多头自注意力是一种扩展形式，**可以帮助模型从不同的表示子空间捕获输入数据的特征**，主要做法是：

*   *Q，K，V*分别进行多次线性变化，从而获得不同的head
*   进行缩放点积注意力
*   将不同的head的注意力结果拼接起来输入线性层

![](https://gitee.com/leeMX111/images_for_markdown/raw/master/attention2.png)

**下面实现一个简单的多头自注意力：**

```python
import torch
import torch.nn.functional as F
from pyexpat import features

x = torch.rand(2,3,128)

num_heads = 4
head_dim = x.size(-1)//num_heads
# 计算每个头对应的维度大小，这里假设原始特征维度能被头的数量整除

linear_layers_q = [torch.nn.Linear(x.size(-1), head_dim) for _ in range(num_heads)]
linear_layers_k = [torch.nn.Linear(x.size(-1), head_dim) for _ in range(num_heads)]
linear_layers_v = [torch.nn.Linear(x.size(-1), head_dim) for _ in range(num_heads)]

# 生成Q、K、V，每个头都有独立的线性层进行转换
Qs = [linear_layer_q(x) for linear_layer_q in linear_layers_q]
Ks = [linear_layer_k(x) for linear_layer_k in linear_layers_k]
Vs = [linear_layer_v(x) for linear_layer_v in linear_layers_v]

# 将每个头的Q、K、V分别堆叠起来，形成新的维度 (batch_size, num_heads, seq_len, head_dim)
Q = torch.stack(Qs, dim=1)
K = torch.stack(Ks, dim=1)
V = torch.stack(Vs, dim=1)

# 完成缩放点积注意力运算
raw_weights = torch.matmul(Q,K.transpose(-2,-1))
scale_factor = K.size(-1) ** 0.5
scale_weights = raw_weights / scale_factor
print(scale_weights.size())
# torch.Size([2, 4, 3, 3]),batch_size, num_heads, seq_len, seq_len

attention_weights = F.softmax(scale_weights, dim=-1)
attention_output = torch.matmul(attention_weights, V)
print(attention_output.size())
# torch.Size([2, 4, 3, 32]) batch_size, num_heads, seq_len, head_dim

# 合并多头输出
def combine_heads(data):
    batch_size, num_heads,seq_len, head_dim  = data.size()
    feature_dim = num_heads*head_dim
    output = data.transpose(1,2).contiguous().view(batch_size,seq_len,feature_dim)
    return output


attention_output = combine_heads(attention_output)
linear_out = torch.nn.Linear(x.size(-1),64)
attention_output = linear_out(attention_output)

print(attention_output.size())
# torch.Size([2, 3, 64])
```

# Chapter6：Transformer
## Transformer结构分析
Transformer的主要结构为：



<img src="https://gitee.com/leeMX111/images_for_markdown/raw/master/transformer.png" alt="pic" style=" height: 800px !important;">

下面配合代码对Transformer的每个组成部分进行说明，在开始前需要对一些参数进行设置：
- `d_q，d_k，d_v`：**Q，K，V**张量的维度，其中`d_q=d_k`
- `batch_size`：每次训练的批数据大小
- `dim_embedding`：词的编码长度
- `num_heads`：多头注意力中“头”的数量
- `n_layer`：encoder或者decoder中的层数
```python
d_k = 64 
d_v = 64
d_q = 64 #必须跟d_k一样
batch_size = 3
dim_embedding = 512 
num_heads = 8
n_layer = 6
```
### 注意力掩码
在进行说明之前，还需要详细说明transformer中的**掩码机制**，具体来说分为两种：
- **填充注意力掩码**（Padding Attention Mask）：当处理的序列长度不一样时，需要对短的序列进行填充，使所有序列的长度一样，这样就可以批处理了。但是填充的部分往往是没有意义的，**这个时候需要将填充的部分进行掩码，具体做法是把这些位置的注意力权重设置为极小值，在softmax后这些权重趋近于0，** 就可以避免这部分内容的影响。
    - 填充注意力掩码在编码器部分和解码器部分都有运用
- **后续注意力掩码**（Subsequent Attention Mask）：在自回归任务时，模型需要逐步的输出序列，**为了避免在输出当前步的序列时看到未来的信息，这里需要将对应未来的信息掩码。具体的做法跟上面的类似，也是将当前位置之后的位置的注意力权重设置为极小值。**
    - 后续注意力掩码主要用在解码器的第一个多头注意力部分



<img src="https://gitee.com/leeMX111/images_for_markdown/raw/master/transformer2.png" alt="pic" style=" height: 400px !important;">

**现在分别实现两种掩码方式：**

- **填充注意力掩码**
    - 输入是最原始的**未编码**的序列，`seq_q`和`seq_k`大小都是`batch_size, len_q`,不同批次（句子）之间的序列长度不一样，所以可能需要填充无意义的内容
    - 掩码张量需要和注意力权重大小一样，大小为`batch_size,len_q,len_k`，直接复制即可

```python
def get_attention_pad_mask(seq_q,seq_k):
    print(seq_q)
    # tensor([[4, 5, 6, 7, 0],
    #         [14, 15, 16, 0, 0],
    #         [1, 2, 3, 0, 0]])
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_mask = seq_k.data.eq(0).unsqueeze(1) #batch_size, 1 ,len_k
    pad_mask = pad_mask.expand(batch_size,len_q,len_k) #batch_size,len_q,len_k
    print(pad_mask)
    # tensor([[[False, False, False, False, True],
    #          [False, False, False, False, True],
    #          [False, False, False, False, True],
    #          [False, False, False, False, True],
    #          [False, False, False, False, True]], ... ]
    return pad_mask
```

- **后续注意力掩码**
    - `np.triu`构建一个大小为`batch_size,len_q,len_k`的上三角矩阵，且往右平移一位
 
```python
def get_attention_subsequent_mask(seq):
    batch_size,seq_len = seq.size()
    subsequent_mask = np.triu(np.ones((batch_size,seq_len,seq_len)),k=1)
    print(subsequent_mask)
    # [[[0. 1. 1. 1. 1.]
    #   [0. 0. 1. 1. 1.]
    #  [0. 0. 0. 1. 1.]
    # [0. 0. 0. 0. 1.]
    # [0. 0. 0. 0. 0.]]...]
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask
```
### 多头自注意力&残差连接&归一化



<img src="https://gitee.com/leeMX111/images_for_markdown/raw/master/transformer3.png" alt="pic" style=" height: 300px !important;">

**首先实现多头自注意机制**

多头自注意力是实现tarnsformer的基础，我们首先需要实现**缩放点积注意力**，其原理在前文中已经阐述，这里不再赘述。
```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
     super(ScaledDotProductAttention,self).__init__()
    def forward(self,Q,K,V,attention_mask):
        raw_weights = torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(d_k)
        raw_weights.masked_fill(attention_mask,-1e9)
        weights = F.softmax(raw_weights,dim=-1)
        attention_output = torch.matmul(weights,V)
        return attention_output,weights
```
这里的重点是**掩码机制**，通过`masked_fill`函数完成，函数功能为：
> `tensor.masked_fill(mask, value)`
其中：
> -   `tensor` 是要进行操作的 `PyTorch` 张量。
> -   `mask` 是一个布尔类型（`torch.bool`）的张量，其形状需要和 `tensor` 的形状或者能够广播（broadcast）到与 `tensor` 相同的形状。这个掩码张量用于指定 `tensor` 中哪些元素需要被填充，在 `mask` 中对应位置为 `True` 的元素所在的 `tensor` 中的位置就是要被填充的位置。
> -   `value` 是用于填充的具体值，其数据类型需要和 `tensor` 中元素的数据类型相匹配（或者能够进行相应的类型转换）。

接下来以此为基础实现**多头自注意力机制**：
![](https://gitee.com/leeMX111/images_for_markdown/raw/master/attention2.png)


- 在实践中，上图指代的**Q，K，V**，都是一样的，都是**输入序列的编码**；
- 多头注意力中，**Q，K，V**应该分别由`num_heads`个线性层得到。在实践中没有定义那么多线性层，而是由一个输出大小为`d_q(or d_k,d_v) * num_heads`的线性层一样，通过后期的形状变化，等价于`num_heads`个线性层的效果；
- 经过线性层后的**Q，K，V**数据形状为`batch_size,num_heads,seq_len,d_v(or d_k,d_v)`
- `attention_mask`需要跟注意力权重大小一样，这里就是加了一个`num_head`维度，因为每个头的输入的序列其实是一样的，pad填充也是一样的，这里直接复制`num_head`即可
- 在完成点积注意力的计算之后，通过形状变化就可以实现上图中的`cancat`的效果

**残差连接和归一化**的做法是在多头自注意力的最终的线性层的下一层，与自注意的输入进行**加和**，一起通入归一化层，也就是`layer norm`
- **残差连接的意义主要是在网络层变大的情况下**，避免模型准确度不增反降的情况。在代码中的实现方式也很简单，直接`+`进行加和就行（注意这里不是拼接，所以这里`output`和`residual`大小一样）
- `layer norm`是针对一个序列的所有特征进行归一化，区别于`batch norm`，在这里直接用`nn.LayerNorm`即可
- 先进行残差加和再`layer norm`，和先`layer norm`再残差加和两种方式都可以，适应的场景不一样


```python
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        # 用一个线性层实现num_heads个线性层的效果
        self.linear_Q = nn.Linear(dim_embedding, d_q * num_heads)
        self.linear_K = nn.Linear(dim_embedding, d_k * num_heads)
        self.linear_V = nn.Linear(dim_embedding, d_v * num_heads)
        self.linear_out = nn.Linear(d_v * num_heads, dim_embedding)
        self.layer_norm = nn.LayerNorm(dim_embedding)

    def forward(self,Q,K,V,attention_mask):
        residual, batch_size = Q,Q.size(0)  #这里Q,K,V的内容是一样的，所以残差可以使用Q保存
        q_s = self.linear_Q(Q).view(batch_size,-1,num_heads,d_q).transpose(1,2)
        k_s = self.linear_K(K).view(batch_size,-1,num_heads,d_k).transpose(1,2)
        v_s = self.linear_V(V).view(batch_size,-1,num_heads,d_v).transpose(1,2)# batch_size,num_heads,seq_len,d_v
        attention_mask = attention_mask.unsqueeze(1).repeat(1,num_heads,1,1)# batch_size, num_heads, seq_len_q ,seq_len_k

        attention_output, weights = ScaledDotProductAttention()(q_s,k_s,v_s,attention_mask)# batch_size,num_heads,seq_len,d_v
        # cancat
        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size,-1,num_heads*d_v) # batch_size,seq_len,num_heads*d_v
        output = self.linear_out(attention_output)
        output = self.layer_norm(output + residual)# batch_size,seq_len,dim_embedding
        return output,weights
```

### 前馈神经网络
**Position-wise Feed Forward**主要分布在编码器和解码器的每一层注意力层之后，主要组成部分是一个简单的**两层线性层，线性层中间用一个激活函数（一般是ReLU）连接**

一般来说第一层线性层会将增加输入的维度，然后通过激活函数，在接入第二个线性层，然后把维度降到原始的大小，这样的好处是**有助于模型学习到更复杂的特征表示，让模型能够学习到输入和输出之间的非线性关系。**



<img src="https://gitee.com/leeMX111/images_for_markdown/raw/master/transformer4.png" alt="pic" style=" height: 200px !important;">

- 跟上面的方法类似，这里也实现了残差连接和归一化处理
- 这里的`Position-wise`的理解是，**神经网络是独立的处理输入序列的每个位置的**，对于一个序列的不同的词语都是应用相同的神经网络，代码中定义线性层的大小为`dim_embedding`可以说明这一点，这样的好处有两点：
    - **参数共享与效率提升**：其实本质是参数共享，减少了训练的时间
    - **位置无关性和泛化能力增强**：这种设计体现了位置无关性。因为每个位置使用相同的神经网络，模型不会对序列中的某个特定位置产生偏向。在处理不同长度的序列时，具有更好的泛化能力。有助于模型学习到序列的全局特征，而不是被局部位置信息所干扰。
```python
class PositionFeedForward(nn.Module):
    def __init__(self, d_ff=2048):
        super(PositionFeedForward, self).__init__()
        self.ffn1 = nn.Linear(dim_embedding, d_ff)
        self.ffn2 = nn.Linear(d_ff, dim_embedding)
        self.layer_norm = nn.LayerNorm(dim_embedding)

    def forward(self, inputs):
        residual = inputs
        output = nn.ReLU()(self.ffn1(inputs))
        output = self.ffn2(output)
        output = self.layer_norm(output + residual)
        return output
```
### 位置编码

$$
PE_{(pos,2i)} = sin(pos/10000^{2i/d_{model}})
\\PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_{model}})
$$
```python
def get_sin_enc_table(seq_len, dim_embedding):
    position_table = np.zeros((seq_len, dim_embedding))
    for pos in range(seq_len):
        for j in range(dim_embedding):
            i = j // 2
            angle = pos / (np.power(10000, (2 * i) / dim_embedding))
            position_table[pos, i] = angle

    position_table[:, 0::2] = np.sin(position_table[:, 0::2])
    position_table[:, 1::2] = np.cos(position_table[:, 0::2])
    return torch.FloatTensor(position_table)
```

### Encoder

```python
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.muti_head_attention = MultiHeadAttention()
        self.ffn = PositionFeedForward()

    def forward(self, embedding_output, encoder_mask):
        encoder_attention_output, weights = self.muti_head_attention(embedding_output, embedding_output,
                                                                     embedding_output, encoder_mask)
        encoder_output = self.ffn(encoder_attention_output)
        return encoder_output, weights
```

```python
class Encoder(nn.Module):
    def __init__(self, corpus):
        super(Encoder, self).__init__()
        self.embedding_layer = nn.Embedding(len(corpus.src_vocab), dim_embedding)
        self.position_embedding = nn.Embedding.from_pretrained(get_sin_enc_table(corpus.src_len + 1, dim_embedding),
                                                               freeze=True)
        self.encoder_layers = nn.ModuleList(EncoderLayer() for _ in range(n_layer))

    def forward(self, encoder_input):
        pos_indices = torch.arange(1, encoder_input.size(1) + 1).unsqueeze(0).to(encoder_input)  # 1,source_len
        encoder_output = self.embedding_layer(encoder_input) + self.position_embedding(pos_indices)
        encoder_mask = get_attention_pad_mask(encoder_input, encoder_input)
        encoder_weights = []
        for layer in self.encoder_layers:
            encoder_output, weights = layer(encoder_output, encoder_mask)
            encoder_weights.append(weights)

        return encoder_output, encoder_weights
```

### Decoder

```python
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.muti_self_attention = MultiHeadAttention()
        self.muti_seq2seq_attention = MultiHeadAttention()
        self.ffn = PositionFeedForward()

    def forward(self, encoder_output, embedding_output, self_mask, seq2seq_mask):
        decoder_output, self_weights = self.muti_self_attention(embedding_output, embedding_output, embedding_output,
                                                                self_mask)
        decoder_output, seq2seq_weights = self.muti_seq2seq_attention(decoder_output, encoder_output, encoder_output,
                                                                      seq2seq_mask)
        decoder_output = self.ffn(decoder_output)
        return decoder_output, self_weights, seq2seq_weights

```


```python
class Decoder(nn.Module):
    def __init__(self, corpus):
        super(Decoder, self).__init__()
        self.embedding_layer = nn.Embedding(len(corpus.tgt_vocab), dim_embedding)
        self.position_embedding = nn.Embedding.from_pretrained(get_sin_enc_table(corpus.src_len + 1, dim_embedding),
                                                               freeze=True)
        self.decoder_layers = nn.ModuleList(DecoderLayer() for _ in range(n_layer))

    def forward(self, decoder_input, encoder_input, encoder_output):
        pos_indices = torch.arange(1, decoder_input.size(1) + 1).unsqueeze(0).to(decoder_input)  # 1,source_len
        embedding_output = self.embedding_layer(decoder_input) + self.position_embedding(pos_indices)
        decoder_output = embedding_output

        decoder_self_pad_mask = get_attention_pad_mask(decoder_input, decoder_input)
        decoder_seq2seq_pad_mask = get_attention_pad_mask(decoder_input, encoder_input)
        decoder_self_subsequent_mask = get_attention_subsequent_mask(decoder_input)

        decoder_self_mask = torch.gt((decoder_self_pad_mask + decoder_self_subsequent_mask), 0)
        decoder_self_weights = []
        decoder_seq2seq_weights = []
        for layer in self.decoder_layers:
            decoder_output, self_weights, seq2seq_weights = layer(encoder_output, decoder_output, decoder_self_mask,
                                                                  decoder_seq2seq_pad_mask)
            decoder_self_weights.append(self_weights)
            decoder_seq2seq_weights.append(seq2seq_weights)

        return decoder_output, decoder_self_weights, decoder_seq2seq_weights
```

### Transformer

```python
class Transformer(nn.Module):
    def __init__(self, corpus):
        super(Transformer, self).__init__()
        self.encoder = Encoder(corpus)
        self.decoder = Decoder(corpus)
        self.projection = nn.Linear(dim_embedding, len(corpus.tgt_vocab), bias=False)

    def forward(self, encoder_input, decoder_input):
        encoder_output, encoder_weights = self.encoder(encoder_input)
        decoder_output, decoder_self_weights, decoder_seq2seq_weights = self.decoder(decoder_input, encoder_input,
                                                                                     encoder_output)
        decoder_logits = self.projection(decoder_output)
        return decoder_logits, encoder_weights, decoder_self_weights, decoder_seq2seq_weights
```
# Chapter7：GPT
语言模型从内部原理可以大致分为一支是 **[自编码语言模型]**（**Autoencoder Language Model**），**[自回归语言模型]（AutoregressiveLanguage Model）**
- **自编码语言模型**：通俗来说是从



<img src="https://gitee.com/leeMX111/images_for_markdown/raw/master/4-Figure1-2.png" alt="pic" style=" height: 400px !important;">


GPT作为自回归语言模型，与Transformer最大的差异在于，**其可以看成只采用了Decoder部分，且不包含编码器-解码器自注意力，任意时刻的输入都只能看到之前时间的信息**（Transformer也在Decoder部分增加了因果掩码，但是由于编码器的参与，其实也是可以看到所有时间的信息的）。GPT单层的组成主要由下面几个部分组成：
- **词语编码**：词语编码跟Transformer一样
- **位置编码**：GPT 主要采用了一种相对简单的位置编码方式，称为**绝对位置嵌入（Absolute Positional Embedding）**。其实实现方式就是一个相同的`embedding`层，输入是绝对位置
- **掩码多头注意力**：掩码和Transformer的`Decoder`层类似，包含**填充掩码**和**因果掩码**
- **残差连接&归一化&FFN**：跟Transformer一样

**总体来说，GPT的结构是Transformer的解码器删掉了Seq2Seq自注意力部分**

## GPT搭建
**下面实现GPT的单层：**
主要包含4层，注意此处与transfomer的解码器之间的区别：
- 掩码多头注意力层
- layer norm层
- 全连接层
- layer norm层
在本书的代码中，此处的网络层都是采用chapter 6中的基础网络层，但是这些基础网络层其实已经内嵌了残差连接和归一化等操作，此处直接调用可能与原本的GBT结构不一样（不过对结果影响不大）

```python
dim_embedding = 512
n_layer = 6
batch_size = 3
device = "cuda" if torch.cuda.is_available() else "cpu"
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer,self).__init__()
        self.multi_head_attention = MultiHeadAttention()
        self.ffn = PositionFeedForward()
        self.layer_norm_1 = nn.LayerNorm(dim_embedding)
        self.layer_norm_2 = nn.LayerNorm(dim_embedding)
    def forward(self,decoder_input,mask):
        attention_output,_ = self.multi_head_attention(decoder_input,decoder_input,decoder_input,mask)
        norm_1_output = self.layer_norm_1(attention_output + decoder_input)
        ffn_output = self.ffn(norm_1_output)
        norm_2_output = self.layer_norm_2(ffn_output + norm_1_output)
        return norm_2_output
```

**根据单层实现GPT的解码器结构**
主要添加了前缀的词嵌入层，并定义了`n_layer`个解码器单层

```python
class Decoder(nn.Module):
    def __init__(self,corpus):
        super(Decoder,self).__init__()
        self.embedding_layer = nn.Embedding(len(corpus.vocab), dim_embedding)
        self.position_embedding = nn.Embedding(len(corpus.vocab), dim_embedding)
        self.decoder_layers = nn.ModuleList(DecoderLayer() for _ in range(n_layer))

    def forward(self,decoder_input):
        pos_indices = torch.arange(1, decoder_input.size(1) + 1).unsqueeze(0).to(device)  # 1,source_len
        a = self.embedding_layer(decoder_input)
        b = self.position_embedding(pos_indices)
        embedding_output = a + b
        decoder_output = embedding_output

        decoder_self_subsequent_mask = get_attention_subsequent_mask(decoder_input).to(device)

        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output,decoder_self_subsequent_mask)

        return decoder_output
```
**最终定义完整的GPT结构**
添加输出的线性层，其输出大小为词表的大小，表示下一个词的概率
```python
class GPT(nn.Module):
    def __init__(self,corpus):
        super(GPT,self).__init__()
        self.decoder = Decoder(corpus)
        self.projection = nn.Linear(dim_embedding, len(corpus.vocab), bias=False)

    def forward(self,inputs):
        decoder_input = self.decoder(inputs)
        logits = self.projection(decoder_input)
        return logits
```
## 数据准备

## 集束搜索
# Chapter8：ChatGPT基于强化学习
# Chapter8：ChatGPT基于强化学习
