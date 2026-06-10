document.addEventListener('DOMContentLoaded', function() {
  const chatContainer = document.getElementById('chat-container');
  const userInput = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');
  
  // 示例回答数据
  const responses = {
    '什么是rag': 'RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的AI技术。它先从知识库中检索相关信息，然后将这些信息作为上下文输入到大语言模型中，生成基于知识库的精准回答。',
    '什么是 RAG': 'RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的AI技术。它先从知识库中检索相关信息，然后将这些信息作为上下文输入到大语言模型中，生成基于知识库的精准回答。',
    'rag是什么': 'RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的AI技术。它先从知识库中检索相关信息，然后将这些信息作为上下文输入到大语言模型中，生成基于知识库的精准回答。',
    '深度学习有哪些应用': '深度学习应用广泛，包括：\n\n1. **计算机视觉**: 图像识别、目标检测、人脸识别、图像生成\n2. **自然语言处理**: 机器翻译、文本生成、情感分析、问答系统\n3. **语音处理**: 语音识别、语音合成\n4. **推荐系统**: 个性化推荐、广告投放\n5. **医疗诊断**: 医学影像分析、疾病预测\n6. **自动驾驶**: 环境感知、路径规划',
    '介绍一下GPT': 'GPT (Generative Pre-trained Transformer) 是OpenAI开发的大型语言模型系列。它基于Transformer架构，通过预训练学习海量文本数据，能够生成高质量的自然语言文本。目前最新版本是GPT-4，支持多模态输入输出，在对话、写作、编程等多个领域表现出色。',
    '强化学习': '强化学习是机器学习的一个分支，智能体通过与环境交互，通过试错来学习最优策略。核心要素包括：\n\n- **状态(State)**: 环境的当前情况\n- **动作(Action)**: 智能体选择的行为\n- **奖励(Reward)**: 对动作的反馈\n- **策略(Policy)**: 从状态到动作的映射\n\n强化学习广泛应用于游戏AI、机器人控制、金融交易等领域。',
    '你好': '你好！我是 M-Seek，基于 RAG 技术的智能问答助手。有什么可以帮你的吗？',
    'hello': 'Hello! I am M-Seek, an intelligent Q&A assistant based on RAG technology. How can I help you?',
    '谢谢': '不客气！有任何问题随时问我。',
    '谢谢': '不客气！有任何问题随时问我。',
    '再见': '再见！祝你有美好的一天！'
  };

  // 默认回答
  const defaultResponse = '感谢你的提问！我正在查询知识库...\n\n由于当前是演示版本，完整的RAG问答功能需要后端服务支持。以下是一些相关信息：\n\n**RAG技术特点：**\n- 结合检索与生成\n- 支持私有知识库\n- 回答可追溯来源\n- 减少幻觉问题\n\n如需部署完整功能，请配置后端服务。';

  function addMessage(content, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = isUser ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // 将换行转换为 <p> 标签
    const paragraphs = content.split('\n').filter(p => p.trim());
    paragraphs.forEach(p => {
      if (p.startsWith('**') && p.endsWith('**')) {
        const text = p.slice(2, -2);
        contentDiv.innerHTML += `<strong>${text}</strong>`;
      } else if (p.startsWith('* ')) {
        const text = p.slice(2);
        contentDiv.innerHTML += `<p><span style="display: inline-block; margin-right: 8px;">•</span>${text}</p>`;
      } else {
        contentDiv.innerHTML += `<p>${p}</p>`;
      }
    });
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    chatContainer.appendChild(messageDiv);
    
    // 滚动到底部
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot';
    typingDiv.innerHTML = `
      <div class="message-avatar"><i class="fas fa-robot"></i></div>
      <div class="message-content">
        <div class="typing-indicator">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
      </div>
    `;
    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return typingDiv;
  }

  function removeTypingIndicator(element) {
    if (element && element.parentNode) {
      element.parentNode.removeChild(element);
    }
  }

  function getResponse(question) {
    const lowerQuestion = question.toLowerCase().trim();
    
    // 查找匹配的回答
    for (const key in responses) {
      if (lowerQuestion.includes(key.toLowerCase())) {
        return responses[key];
      }
    }
    
    return defaultResponse;
  }

  function sendMessage() {
    const question = userInput.value.trim();
    if (!question) return;
    
    // 清空输入
    userInput.value = '';
    
    // 添加用户消息
    addMessage(question, true);
    
    // 显示打字指示器
    const typingIndicator = showTypingIndicator();
    
    // 模拟延迟响应
    setTimeout(() => {
      removeTypingIndicator(typingIndicator);
      const response = getResponse(question);
      addMessage(response, false);
    }, 1500 + Math.random() * 1000);
  }

  function handleKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  // 发送按钮点击事件
  sendBtn.addEventListener('click', sendMessage);
  
  // 输入框回车键事件
  userInput.addEventListener('keydown', handleKeydown);
  
  // 示例标签点击事件
  const exampleTags = document.querySelectorAll('.example-tag');
  exampleTags.forEach(tag => {
    tag.addEventListener('click', function() {
      userInput.value = this.textContent.trim().replace('试试问我：', '');
      sendMessage();
    });
  });
  
  // 自动调整文本框高度
  userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
  });
});