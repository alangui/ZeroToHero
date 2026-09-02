import re

def clean_chars(text):
    cleaned_text = ''.join(c for c in text if keep_char(c))
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    return cleaned_text

def keep_char(c):
    o = ord(c)
    return (
        0x4E00 <= o <= 0x9FFF      # 基本汉字
        or 0x3400 <= o <= 0x4DBF   # 扩展A区汉字（生僻字）
        or c in '\n，。！？；：、""''（）《》〈〉【】「」……—·～'
        or c in '0123456789'       # 数字（可选）
    )

def save_txt(clean_chars):
    with open('clean_chars.txt', 'w',  encoding='utf-8') as f:
        f.write(clean_chars)
    

with open('../corpus_clean_zh.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(len(text))
text = clean_chars(text)
print(f"clean chars work done:{len(text)}")
save_txt(text)

