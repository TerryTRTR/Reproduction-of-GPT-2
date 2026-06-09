"""
nanoGPT 风格的配置覆盖工具（自写复现版）。

用法：在主脚本（train.py / sample.py）末尾用
    exec(open('configurator.py').read())
来执行本文件，它会：
  1. 把命令行里形如 `path/to/config.py` 的参数当作配置文件执行（其中的赋值会覆盖全局变量）；
  2. 把形如 `--key=value` 的参数解析后覆盖同名全局变量（类型按已有默认值推断）。

之所以采用这种 “在调用方作用域里 exec” 的写法，是为了让配置文件与命令行可以直接读写
train.py 顶部定义的那一批全局超参，而无需把所有超参打包成对象传来传去——这与
Karpathy 的 nanoGPT 行为一致，便于复现与对照。
"""

import sys
from ast import literal_eval


def _override():
    for arg in sys.argv[1:]:
        if "=" not in arg:
            # 形如 config/train_xxx.py 的配置文件：直接执行
            assert not arg.startswith("--"), f"未知参数（缺少值）：{arg}"
            config_file = arg
            print(f"Overriding config with {config_file}:")
            with open(config_file, "r", encoding="utf-8") as f:
                code = f.read()
            print(code)
            exec(code, globals())
        else:
            # 形如 --key=value 的命令行覆盖
            assert arg.startswith("--"), f"未知参数：{arg}"
            key, val = arg[2:].split("=", 1)
            if key not in globals():
                raise ValueError(f"未知配置项：{key}")
            default = globals()[key]
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val  # 解析失败就当字符串
            if default is not None:
                assert type(attempt) == type(default), (
                    f"配置项 {key} 类型不匹配：期望 {type(default)}，得到 {type(attempt)}"
                )
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt


# 直接在调用方（exec 进来的）全局作用域执行
_override()
