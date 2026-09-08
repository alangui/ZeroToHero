def stats(path):
    """流式读取，返回 (总行数, 去重后行数, 去重集合)。空行不计（避免段落间空行虚增重复率）"""
    total = 0
    uniq = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            total += 1
            uniq.add(line)
    return total, len(uniq), uniq

t1, u1, s1 = stats('wiki_corpus_0.1b_clean.txt')
print(f"0.1b: 有效行 {t1:,}, 去重后 {u1:,}, 内部重复率 {1 - u1 / t1:.2%}")

t3, u3, s3 = stats('wiki_corpus_0.3b_clean.txt')
print(f"0.3b: 有效行 {t3:,}, 去重后 {u3:,}, 内部重复率 {1 - u3 / t3:.2%}")

inter = len(s1 & s3)
print(f"\n0.1b 的去重行出现在 0.3b 中的比例: {inter / u1:.2%}   ← 接近 100% 说明 0.3b ⊇ 0.1b")
print(f"0.3b 的去重行来自 0.1b 的比例:   {inter / u3:.2%}   ← 正常应 ≈ 1/3")