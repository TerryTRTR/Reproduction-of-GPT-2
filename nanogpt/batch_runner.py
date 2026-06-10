"""
B 部分批量实验运行器 — 按顺序运行 B1-B6 实验，然后对最优配置跑多种子。

每个实验的 stdout 会同时打印到终端和保存到对应 out_dir/experiment.log。
"""
import subprocess
import sys
import os
import json
import time

# 所有单次实验定义 (按推荐实验矩阵)
EXPERIMENTS = [
    {"name": "B1", "learning_rate": 6e-4, "min_lr": 6e-5, "dropout": 0.1, "block_size": 512, "out_dir": "out-wikitext2-14m-lr6e4-drop01", "seed": 1337, "desc": "默认配置基线"},
    {"name": "B2", "learning_rate": 3e-4, "min_lr": 3e-5, "dropout": 0.1, "block_size": 512, "out_dir": "out-wikitext2-14m-lr3e4-drop01", "seed": 1337, "desc": "降低学习率"},
    {"name": "B3", "learning_rate": 2e-4, "min_lr": 2e-5, "dropout": 0.1, "block_size": 512, "out_dir": "out-wikitext2-14m-lr2e4-drop01", "seed": 1337, "desc": "更保守学习率"},
    {"name": "B4", "learning_rate": 3e-4, "min_lr": 3e-5, "dropout": 0.2, "block_size": 512, "out_dir": "out-wikitext2-14m-lr3e4-drop02", "seed": 1337, "desc": "增强正则 dropout=0.2"},
    {"name": "B5", "learning_rate": 3e-4, "min_lr": 3e-5, "dropout": 0.3, "block_size": 512, "out_dir": "out-wikitext2-14m-lr3e4-drop03", "seed": 1337, "desc": "更强正则 dropout=0.3"},
    {"name": "B6", "learning_rate": 3e-4, "min_lr": 3e-5, "dropout": 0.2, "block_size": 256, "out_dir": "out-wikitext2-14m-lr3e4-drop02-bs256", "seed": 1337, "desc": "短上下文 block_size=256"},
]

# 多种子实验 — 将在阶段1完成后确定最优配置
MULTI_SEED_SEEDS = [42, 123, 456]

def run_one(exp, python_exe, script_dir):
    """运行单个实验，返回 (success, best_val_loss, best_iter, elapsed_minutes)"""
    name = exp["name"]
    out_dir = os.path.join(script_dir, exp["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "experiment": name,
        "description": exp["desc"],
        "learning_rate": exp["learning_rate"],
        "min_lr": exp["min_lr"],
        "dropout": exp["dropout"],
        "block_size": exp["block_size"],
        "seed": exp["seed"],
        "out_dir": exp["out_dir"],
        "max_iters": 6000,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "experiment_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log_path = os.path.join(out_dir, "experiment.log")

    cmd = [
        python_exe, "-u", "train.py",
        "config/train_wikitext2_14m.py",
        "--compile=False",
        f"--learning_rate={exp['learning_rate']}",
        f"--min_lr={exp['min_lr']}",
        f"--dropout={exp['dropout']}",
        f"--block_size={exp['block_size']}",
        f"--out_dir={exp['out_dir']}",
        f"--seed={exp['seed']}",
        "--max_iters=6000",
    ]

    print(f"\n{'='*70}")
    print(f"[BatchRunner] 开始实验 {name}: {exp['desc']}")
    print(f"[BatchRunner] 命令: {' '.join(cmd)}")
    print(f"[BatchRunner] 输出目录: {out_dir}")
    print(f"{'='*70}\n")
    sys.stdout.flush()

    start = time.time()
    best_val_loss = None
    best_iter = None

    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=script_dir,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
            # 提取 best_val_loss 和 best_iter
            if "val loss" in line and "step" in line:
                # 格式: step N: train loss X.XXXX, val loss X.XXXX, val ppl XXX.XX
                try:
                    parts = line.strip().split()
                    step_idx = parts.index("step") + 1
                    iter_num = int(parts[step_idx].rstrip(":"))
                    val_loss_idx = parts.index("val") + 1
                    val_loss_str = parts[val_loss_idx + 1].rstrip(",")
                    val_loss = float(val_loss_str)
                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_iter = iter_num
                except (ValueError, IndexError):
                    pass
        proc.wait()

    elapsed = time.time() - start
    elapsed_min = round(elapsed / 60, 1)

    meta["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["elapsed_seconds"] = round(elapsed, 1)
    meta["elapsed_minutes"] = elapsed_min
    meta["return_code"] = proc.returncode
    meta["best_val_loss"] = best_val_loss
    meta["best_iter"] = best_iter
    with open(os.path.join(out_dir, "experiment_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[BatchRunner] 实验 {name} 完成: best_val_loss={best_val_loss}, best_iter={best_iter}, 耗时 {elapsed_min} min\n")
    sys.stdout.flush()

    return proc.returncode == 0, best_val_loss, best_iter, elapsed_min


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable
    batch_log = os.path.join(script_dir, "batch_experiments.log")

    print(f"[BatchRunner] Python: {python_exe}")
    print(f"[BatchRunner] 工作目录: {script_dir}")
    print(f"[BatchRunner] 批量日志: {batch_log}")
    print(f"[BatchRunner] 阶段1: 运行 B1-B6 单次实验")
    print(f"[BatchRunner] 实验数量: {len(EXPERIMENTS)}")
    sys.stdout.flush()

    # ---- 阶段1: 运行 B1-B6 ----
    results = {}
    for exp in EXPERIMENTS:
        ok, best_val, best_iter, elapsed = run_one(exp, python_exe, script_dir)
        results[exp["name"]] = {
            **exp,
            "best_val_loss": best_val,
            "best_iter": best_iter,
            "elapsed_minutes": elapsed,
            "success": ok,
        }

    # 保存阶段1结果
    with open(os.path.join(script_dir, "phase1_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- 阶段2: 找出最佳配置 ----
    best_name = None
    best_loss = float("inf")
    for name, r in results.items():
        if r["best_val_loss"] is not None and r["best_val_loss"] < best_loss:
            best_loss = r["best_val_loss"]
            best_name = name

    print(f"\n[BatchRunner] ====== 阶段1完成 ======")
    print(f"[BatchRunner] 最优实验: {best_name}, best_val_loss={best_loss}")
    sys.stdout.flush()

    if best_name is None:
        print("[BatchRunner] 无法确定最优配置，跳过阶段2")
        return

    best_exp = results[best_name]
    print(f"[BatchRunner] 阶段2: 对最优配置 {best_name} 跑 {len(MULTI_SEED_SEEDS)} 个种子")
    sys.stdout.flush()

    # ---- 阶段2: 多种子 ----
    multi_results = {}
    for i, s in enumerate(MULTI_SEED_SEEDS):
        name = f"{best_name}-s{s}"
        out_dir_name = f"{best_exp['out_dir']}-s{s}"
        multi_exp = {
            "name": name,
            "learning_rate": best_exp["learning_rate"],
            "min_lr": best_exp["min_lr"],
            "dropout": best_exp["dropout"],
            "block_size": best_exp["block_size"],
            "out_dir": out_dir_name,
            "seed": s,
            "desc": f"多种子验证 (seed={s})",
        }
        ok, best_val, best_iter, elapsed = run_one(multi_exp, python_exe, script_dir)
        multi_results[name] = {
            **multi_exp,
            "best_val_loss": best_val,
            "best_iter": best_iter,
            "elapsed_minutes": elapsed,
            "success": ok,
        }

    # 保存阶段2结果
    with open(os.path.join(script_dir, "phase2_results.json"), "w") as f:
        json.dump(multi_results, f, indent=2)

    # ---- 汇总 ----
    print(f"\n[BatchRunner] ====== 全部实验完成 ======")
    print(f"[BatchRunner] 阶段1结果: {json.dumps(results, indent=2)}")
    print(f"[BatchRunner] 阶段2结果: {json.dumps(multi_results, indent=2)}")
    print(f"[BatchRunner] 汇总文件: phase1_results.json, phase2_results.json")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
