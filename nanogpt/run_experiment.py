"""
B 部分实验运行器 — 包装 train.py，将 stdout/stderr 同时输出到终端和日志文件。

用法：
  python run_experiment.py --name=B1 --learning_rate=6e-4 --min_lr=6e-5 --dropout=0.1 --block_size=512 --out_dir=out-wikitext2-14m-lr6e4-drop01 [--seed=1337]

每次实验会在 out_dir 下生成：
  - ckpt.pt          （train.py 保存的 checkpoint）
  - experiment.log   （完整 stdout 日志）
  - experiment_meta.json （实验超参记录）
"""

import subprocess
import sys
import os
import json
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="实验编号，如 B1")
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--min_lr", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max_iters", type=int, default=6000)
    parser.add_argument("--extra_args", default="", help="额外命令行参数")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 记录实验元信息
    meta = {
        "experiment": args.name,
        "learning_rate": args.learning_rate,
        "min_lr": args.min_lr,
        "dropout": args.dropout,
        "block_size": args.block_size,
        "out_dir": args.out_dir,
        "seed": args.seed,
        "max_iters": args.max_iters,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    meta_path = os.path.join(args.out_dir, "experiment_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[runner] 实验 {args.name} 开始，元信息已保存到 {meta_path}")

    log_path = os.path.join(args.out_dir, "experiment.log")

    cmd = [
        sys.executable, "-u", "train.py",
        "config/train_wikitext2_14m.py",
        "--compile=False",
        f"--learning_rate={args.learning_rate}",
        f"--min_lr={args.min_lr}",
        f"--dropout={args.dropout}",
        f"--block_size={args.block_size}",
        f"--out_dir={args.out_dir}",
        f"--seed={args.seed}",
        f"--max_iters={args.max_iters}",
    ]
    if args.extra_args:
        cmd.extend(args.extra_args.split())

    print(f"[runner] 命令: {' '.join(cmd)}")
    print(f"[runner] 日志文件: {log_path}")
    print(f"[runner] ====== 训练输出开始 ======")

    start = time.time()

    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        proc.wait()

    elapsed = time.time() - start

    meta["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["elapsed_seconds"] = round(elapsed, 1)
    meta["elapsed_minutes"] = round(elapsed / 60, 1)
    meta["return_code"] = proc.returncode
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[runner] ====== 训练输出结束 ======")
    print(f"[runner] 实验 {args.name} 完成，耗时 {elapsed/60:.1f} 分钟")
    print(f"[runner] 日志已保存到 {log_path}")
    print(f"[runner] 元信息已保存到 {meta_path}")

    return proc.returncode

if __name__ == "__main__":
    sys.exit(main())
