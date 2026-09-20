# -*- coding: utf-8 -*-
# benchmark_speed.py —— 100 步训练基准：开训前/环境变更后跑一遍，干不干净见分晓
#
# 背景（v11 系列教训）：
#   - v11.1 单步 2.7 秒 vs v11.2 单步 0.76 秒，同数据同脚本，根因疑为电源模式/GPU 被占，
#     当时无基准可对照，只能靠事后模拟实验证伪猜测
#   - 纪律：训练前跑本脚本，把"单步耗时 + 显存峰值"记进 train.log 首行，异常先有数
#
# 运行：python benchmark_speed.py（本目录，需要 sft_data.pt 与 v10 基座权重）
import time
import torch
import importlib.util

spec = importlib.util.spec_from_file_location('sft', 'sft_zh_v11.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

WARMUP = 20       # 前 20 步预热（allocator 热身、功耗爬升），不计入
STEPS = 100       # 计时的步数


def main():
    train_data, _ = m.load_samples()
    model = m.load_pretrained_model().to(m.device)
    opt = torch.optim.AdamW(model.parameters(), lr=m.learning_rate, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for step in range(WARMUP + STEPS):
        if step == WARMUP:
            t0 = time.time()          # 预热结束，重新掐表
        xb, yb = m.get_batch(train_data)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    dt = time.time() - t0
    print(f'基准结果（{STEPS} 步, batch {m.batch_size}, 预热 {WARMUP} 步）:')
    print(f'  单步耗时: {dt / STEPS:.2f} 秒/步   (v11.2 参考值: 0.76 秒/步)')
    print(f'  显著变慢(>1.5×)请先查: 电源模式 / nvidia-smi 里谁在占卡 / 是否省电模式')
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f'  显存峰值: {peak:.2f} GB')

    # 显卡实时状态（SSH 变慢之谜的定位线索：看 SM 频率/功耗是否被压）
    import subprocess
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=clocks.sm,power.draw,pstate',
             '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        print(f'  显卡状态: {out.stdout.strip()}  (SM频率低/功耗低 = 被限频，查电源/会话)')
    except Exception:
        pass


if __name__ == '__main__':
    main()
