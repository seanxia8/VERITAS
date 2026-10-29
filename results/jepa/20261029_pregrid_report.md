# JEPA 预网格检查报告 — 2026-10-29

**结果均为 `registered: false`；网格保持暂停，等待作者。**

- **A：复现通过。** 线性与 population 两项的最大角度差均小于 1e-9°，限值为 0.1°。
- **B：β 重选完成。** 固定 plain 顶点分辨率从旧参考 0.264400 m 改为 0.219259 m；两个 anchored 臂仍选 β=1000，anchor-only 仍为 none-qualified。
- **C：全部通过。** 六个 checkpoint 在一次新调用完成，`complete:true`、anc100 β=100，五项独立期望误差均 ≤0.001 m；原失败尝试保留。[七个项目、full/half 和全部指标的表](20261029_part_c/METRICS.md)。固定种子 half/full 为 1.484 和 1.436，evaluation mismatch **不支持**。
- **D：十项完成。** [五个变体 × 两种子及三个 floor 的表](20261029_d2_partial/METRICS.md)。task too local、drift too weak、label shaping、SIGReg overridden 和 EMA 五个预先规定判读均**不支持**；它们是描述性结果，不是网格决策。Strong-drift gain R² 为 0.0975/0.2172，低于其 0.8416 初始化 floor。
- **E：九项完成并完整聚合。** 下面每项为 plain / anchored 的三种子均值；延迟固定为 0.01。幅度 probe 为 post-time / token R²。详细来源和逐臂证据见 [对照表](20261029_bridge_delay/COMPARISON.md)。

| 条件 | nuisance R² | 振幅 MSE/CRLB | t0 MSE/CRLB | τ MSE/CRLB | 幅度 probe post-time R² | 幅度 probe token R² |
|---|---:|---:|---:|---:|---:|---:|
| 注册 B1 | .5251 / .5157 | 1.1646 / 1.1465 | 9.3740 / 3.1591 | 1.1003 / 1.0795 | .8096 / .8140 | .9074 / .9018 |
| 延迟扩展 | .4320 / .6221 | 1.2807 / 1.0790 | 49.2160 / 98.7191 | 1.1794 / 1.0150 | .8153 / .8169 | .8879 / .8955 |

延迟结果的描述性 B1/R7 均 fail，不替代注册判定。原 WP4a β 决策文件在记录路径缺失；本次 anchored 用由三条已提交注册 manifest 和开发报告佐证的 **β=10 派生非注册描述文件**，记录原文件哈希，但不是原文件。

**资源：**本次 pre-grid Titan 10.18295、L40S 7.36112 GPU-h，合计 17.54407；包含以往工作在内的账本 Titan 16.70575、L40S 91.77106，合计 108.47681 GPU-h。延迟 bridge 含 smoke 为 3.31876 / 8 Titan GPU-h；无活动预留。

**执行偏差：**一次远端 CPU 测试遗漏 TMPDIR，写入 `/tmp/pytest-of-dwong/pytest-99`，超出授权根目录；发现后中断并保留现场。随后在 `/ceph/dwong/jepa/tmp/` 内复测 58/58 通过；本地完整套件 497/497 通过。原失败 Part C 证据未改动。
