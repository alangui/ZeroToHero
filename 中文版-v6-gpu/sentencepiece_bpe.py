import sentencepiece as spm

"""
这是从字符级迁到 BPE 时最容易踩的坑。sentencepiece 训练时是按行读文件的——每行被视为一个独立句子，\n 本身永远不会成为 token。
而字符级模型词表里是含 \n 的，生成的小说有段落感；换成 BPE 后如果不做处理，生成文本会变成一坨从头到尾不分段的文字。
解决办法：注册一个用户自定义符号当"换行 token"：
"""

# 训练时：清洗后的文件保持原样（每行一段），加 user_defined_symbols
spm.SentencePieceTrainer.train(
    input='../wiki_corpus_0.1b_clean.txt',
    model_prefix='zh_bpe',
    model_type='bpe',
    vocab_size=16000,
    character_coverage=1.0,
    byte_fallback=True,
    split_digits=True,
    user_defined_symbols=['[BR]'],   # 自定义换行符号，必进词表
)

"""
# 编码训练数据时：逐行编码，行间插入 [BR] 的 id
sp = spm.SentencePieceProcessor(model_file='zh_bpe.model')
NL_ID = sp.piece_to_id('[BR]')

ids = []
for line in text.split('\n'):
    ids.extend(sp.encode(line))
    ids.append(NL_ID)
data = torch.tensor(ids, dtype=torch.long)

# 生成后解码时：把 [BR] 换回真换行
text_out = sp.decode(generated_ids).replace('[BR]', '\n')
"""