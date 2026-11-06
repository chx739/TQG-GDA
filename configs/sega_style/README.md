# SEGA式 TQG 主配置

这里使用新的 typed Config，不使用旧配置的 extends 继承。两份配置共享源初始化、骨干和适配预算，分别执行 v1/v2 文档。

从仓库根目录运行，替换 cache_path 或用 CLI 的 --cache-path；数据 SHA 必须与 source_split 一致。目标索引默认 0→1，0→2/3 用 --target-index 指定。运行和公式对照见 [实现说明](../../docs/code/sega_style_tqg.md)。尚无本配置的新真实数据成绩。
