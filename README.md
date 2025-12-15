# 演示文档 / Demo Report

本 README 用于演示“本地多模态 AI 助手”最小实现的运行效果、命令示例与截图指引。请按需替换占位符路径与截图文件。

## 环境 & 启动
1) 创建并激活虚拟环境
```bash
python -m venv .venv && source .venv/bin/activate
```
2) 安装依赖
```bash
pip install -r requirements.txt
```
3) 配置 OpenAI Key（使用低成本模型 `text-embedding-3-small` / `gpt-image-embedding-1`）
```bash
cp env.example .env
echo 'OPENAI_API_KEY=your_key_here' > .env  # 或 export OPENAI_API_KEY=...
```

## 核心命令（CLI）
- 添加/分类论文：`python main.py add_paper /path/to/paper.pdf --topics "CV,NLP"`
- 搜索论文：`python main.py search_paper "transformer encoder" --top_k 5`
- 以文搜图：`python main.py search_image "sunset on the beach" --top_k 5`
- 批量整理：`python main.py organize_folder ./my_folder --topics "CV"`
- 初始化 git：`python main.py init_git`

Chroma 持久化目录：`data/index`。

## 运行截图（请按下述指引替换占位符）
将截图放入 `docs/screenshots/` 目录，并在下方路径替换为真实文件名。

1. 论文搜索结果示例  
   - 命令：`python main.py search_paper "transformer encoder"`  
   - 截图：`docs/screenshots/search_paper.png`

2. 以文搜图结果示例  
   - 命令：`python main.py search_image "sunset on the beach"`  
   - 截图：`docs/screenshots/search_image.png`

3. 批量整理后文件夹结构  
   - 命令：`python main.py organize_folder ./samples --topics "CV"`  
   - 截图：`docs/screenshots/organized_folder.png`

> 如果需要更完整的报告，请将上述截图嵌入到 PDF 或直接在本 README 中引用，并附上命令输出文本（可通过终端复制粘贴）。

## 最简演示步骤建议
1. 准备若干 PDF 与图片至 `./samples`。  
2. 运行 `python main.py organize_folder ./samples --topics "CV"` 完成批量嵌入与可选移动。  
3. 执行搜索命令并截图：  
   - `python main.py search_paper "your query"`  
   - `python main.py search_image "your query"`  
4. 将截图放入 `docs/screenshots/` 并更新路径。  
5. 如需提交 PDF 报告，可将本 README 导出或复制到 PDF。  

## 注意
- 请勿提交真实 API Key；`.env` 已在 `.gitignore` 中。  
- 运行搜索前需确保已完成对应内容的 ingest（add_paper 或 organize_folder）。  

