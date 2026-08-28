import json
import re
from pathlib import Path

CORPUS_DIR = Path('AMC_mini')  
FILES = [
    'AMC_conversations_100K.txt',
    'AMC_fictions_100K.txt',
    'AMC_nonfictions_100K.txt',
    'AMC_scientific_articles_100K.txt',
]
OUT_TEXT = 'corpus_clean_zh.txt'   # 清洗合并后的语料
OUT_VOCAB = 'vocab_zh.json'        # 词表映射

# ===== 1. 读取并合并 =====
texts = []
for name in FILES:
    path = CORPUS_DIR / name
    content = path.read_text(encoding='utf-8')
    texts.append(content)
    print(f"{name}: {len(content)} 字符")

text = '\n'.join(texts)
print(f"合并后总计: {len(text)} 字符")

# ===== 2. 清洗 =====
def keep_char(c):
    o = ord(c)
    return (
        0x4E00 <= o <= 0x9FFF    # 基本汉字
        or 0x3400 <= o <= 0x4DBF # 扩展汉字（生僻字）
        or c in '\n，。！？；：、“”‘’（）《》〈〉【】「」……—·～'
        or c in '0123456789'
    )

cleaned = ''.join(c for c in text if keep_char(c))
cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)   # 压缩连续空行
print(f"清洗后: {len(cleaned)} 字符（删掉 {len(text) - len(cleaned)} 个）")

with open(OUT_TEXT, 'w', encoding='utf-8') as f:
    f.write(cleaned)

# ===== 3. 构建词表并保存 =====
chars = sorted(list(set(cleaned)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

with open(OUT_VOCAB, 'w', encoding='utf-8') as f:
    json.dump({'chars': chars}, f, ensure_ascii=False)

print(f"词表大小: {len(chars)}")
print("完成！生成文件:", OUT_TEXT, OUT_VOCAB)