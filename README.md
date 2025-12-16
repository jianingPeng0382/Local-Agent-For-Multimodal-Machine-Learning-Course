# 本地多模态 AI 助手 / Local Multimodal AI Assistant

本 README 用于演示"本地多模态 AI 助手"最小实现的运行效果、命令示例与截图指引。

## 核心功能

- **智能文档分类**：基于文档内容（标题、摘要）自动分类到指定主题（CV、NLP等）
- **语义搜索论文**：使用自然语言查询搜索相关论文内容
- **以文搜图**：通过文本描述搜索相关图像，结果自动保存到文件夹
- **批量整理**：自动处理文件夹中的 PDF 和图像文件，进行分类和索引

## 环境 & 启动

### 1. 创建并激活虚拟环境
```bash
python -m venv .venv && source .venv/bin/activate
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置 OpenAI API Key
```bash
# 方式1：使用 .env 文件
cp env.example .env
echo 'OPENAI_API_KEY=your_key_here' > .env

# 方式2：直接导出环境变量
export OPENAI_API_KEY='your_key_here'
```

**注意**：本项目使用 OpenAI API 进行文本嵌入和图像描述：
- 文本嵌入：`text-embedding-3-small`
- 图像描述：`gpt-4o-mini`（Vision API）

**功能说明**：
- **智能分类**：基于文档内容（标题、摘要）自动分类，而不是简单使用第一个主题
- **分类策略**：
  1. 首先使用关键词匹配（快速准确）
  2. 如果关键词匹配不明确，使用语义相似度（embedding）进行分类
- **自动提取**：从 PDF 中自动提取标题和摘要用于分类
- **文件移动**：分类后的文件会移动到 `{原文件夹}/organized/{主题}/` 目录

## 核心命令（CLI）

### 1. 添加/分类论文
将单个 PDF 文件添加到索引并标记主题：
```bash
python main.py add_paper samples/papers/1706.03762v7.pdf --topics "NLP"
```

### 2. 搜索论文
使用自然语言查询搜索相关论文：
```bash
python main.py search_paper "attention mechanism" --top_k 5
```

### 3. 以文搜图
通过文本描述搜索图像，结果会自动保存到 `search_results/{查询内容}/` 文件夹：
```bash
python main.py search_image "sunset"
```

**功能说明**：
- 搜索结果会根据查询内容自动推断相关主题（CV/NLP）
- 匹配的图像会自动复制到 `search_results/{查询内容}/` 文件夹
- 文件名添加序号前缀（01_, 02_, ...）便于排序

### 4. 批量整理文件夹
自动处理文件夹中的 PDF 和图像文件，进行智能分类：
```bash
python main.py organize_folder ./samples/papers --topics "CV,NLP"
```



## 技术栈

- **向量数据库**：ChromaDB（持久化目录：`data/index`）
- **文本嵌入**：OpenAI `text-embedding-3-small`
- **图像处理**：OpenAI Vision API (`gpt-4o-mini`) 生成图像描述，然后使用文本嵌入
- **PDF 处理**：pypdf
- **批处理优化**：支持批量 API 调用，提高效率
- **错误处理**：自动重试机制，支持网络波动

## 智能分类原理

### 分类流程
1. **提取文档信息**：从 PDF 第一页提取标题和摘要
2. **关键词匹配**：快速匹配文档中的关键词（如 "transformer", "GAN", "image recognition"）
3. **语义相似度**：如果关键词匹配不明确，计算文档 embedding 与主题描述 embedding 的余弦相似度
4. **主题扩展**：自动扩展主题描述（如 "CV" → "computer vision image recognition object detection..."）

### 支持的主题
- **CV**：计算机视觉相关（图像识别、目标检测、视觉处理等）
- **NLP**：自然语言处理相关（文本分析、语言模型、Transformer 等）
- **ML**：机器学习相关
- **AI**：人工智能相关

## 运行截图指引

将截图放入 `docs/screenshots/` 目录，并在下方路径替换为真实文件名。

1. **论文搜索结果示例**  
   - 命令：`python main.py search_paper "transformer encoder"`  
   - 截图：`docs/screenshots/search_paper.png`

2. **以文搜图结果示例**  
   - 命令：`python main.py search_image "sunset on the beach"`  
   - 截图：`docs/screenshots/search_image.png`

3. **批量整理后文件夹结构**  
   - 命令：`python main.py organize_folder ./samples/papers --topics "CV,NLP"`  
   - 截图：`docs/screenshots/organized_folder.png`

4. **搜索结果文件夹**  
   - 命令：`python main.py search_image "chart graph"`  
   - 截图：`docs/screenshots/search_results_folder.png`

> 如果需要更完整的报告，请将上述截图嵌入到 PDF 或直接在本 README 中引用，并附上命令输出文本（可通过终端复制粘贴）。

## 最简演示步骤

1. **准备数据**：准备若干 PDF 与图片至 `./samples/papers` 和 `./samples/images`
2. **批量整理**：
   ```bash
   python main.py organize_folder ./samples/papers --topics "CV,NLP"
   ```
3. **执行搜索**：
   ```bash
   python main.py search_paper "transformer"
   python main.py search_image "chart graph"
   ```
4. **查看结果**：
   - 论文搜索结果在终端输出
   - 图像搜索结果在 `search_results/` 文件夹中
5. **截图**：将截图放入 `docs/screenshots/` 并更新路径

## 注意事项

- **API Key 安全**：请勿提交真实 API Key；`.env` 已在 `.gitignore` 中
- **运行前准备**：运行搜索前需确保已完成对应内容的索引（`add_paper` 或 `organize_folder`）
- **网络配置**：如果使用代理，请设置环境变量：
  ```bash
  export https_proxy=http://127.0.0.1:7890
  export http_proxy=http://127.0.0.1:7890
  ```
- **API 限制**：项目包含自动重试机制，但请注意 OpenAI API 的速率限制
- **数据持久化**：ChromaDB 索引保存在 `data/index` 目录，删除该目录会清空所有索引

## 故障排除

### 问题：API 连接错误
- 检查网络连接和代理设置
- 验证 `OPENAI_API_KEY` 是否正确设置
- 项目包含自动重试机制，短暂网络波动会自动重试

### 问题：分类结果不准确
- 确保 PDF 文件包含可提取的标题和摘要
- 尝试使用更具体的主题（如 "Computer Vision" 而不是 "CV"）
- 检查文档内容是否与主题相关

### 问题：搜索无结果
- 确保已运行 `organize_folder` 或 `add_paper` 进行索引
- 检查 `data/index` 目录是否存在且包含数据
- 尝试不同的搜索查询

## 项目结构

```
.
├── main.py              # 主入口文件
├── src/
│   ├── store.py         # ChromaDB 存储管理
│   ├── embeddings.py    # 文本/图像嵌入处理
│   ├── pdf_utils.py     # PDF 处理工具
│   └── image_utils.py   # 图像处理工具
├── data/
│   └── index/           # ChromaDB 持久化目录
├── search_results/      # 图像搜索结果保存目录
├── requirements.txt     # Python 依赖
└── README.md           # 本文档
```
