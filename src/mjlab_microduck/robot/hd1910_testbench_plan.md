# HD-1910-C001 BAM 台架标定方案

目标：用 BAM（better-actuator-models）的标准识别流程，标定 HD-1910-C001 的
真实参数，替换当前 `hd1910_m6.json` 的首性估算值，消除控制律/摩擦参数的
sim2real 不确定项。

产出：`hd1910/m6.json`（标定版 BAM 参数）→ 替换 `src/mjlab_microduck/robot/hd1910_m6.json`。

---

## 1. 原理：BAM 怎么识别参数

BAM 的标定是「物理台架 + 轨迹激励 + 仿真拟合」闭环：

1. **台架**：舵机输出轴挂一个已知质量/长度的摆锤（单摆）。摆锤的动力学
   完全已知：`τ_ext = (m + m_arm/2)·g·L·sin(q)`、`J = m·L² + (m_arm/3)·L²`。
2. **激励**：给舵机发设计好的目标轨迹（sin(t²)、起落、多频正弦等），同时
   记录每一拍的 位置/速度/电流/电压/目标角/力矩使能。真实舵机对轨迹的
   **跟随误差**（滞后、过冲、爬行）就是电机+摩擦参数的指纹。
3. **拟合**（`bam.fit`）：用参数化 BAM 模型（m6）**仿真同一轨迹**，用
   optuna CMA-ES 搜索参数，最小化 `|sim_position(t) - log_position(t)|` 的均值。

流程已全部由 BAM 提供（`bam.fit` / `bam.mae` / GPU 并行 `bam.mjlab.Simulator`），
我们只需：**台架机械 + 记录脚本 + 注册 HD-1910 执行器类 + 跑采集与拟合**。

## 2. 机械台架设计

### 2.1 结构（镜像 xl330_test_bench）

现有 `src/mjlab_microduck/robot/xl330_test_bench/` 就是标准方案：固定座 +
舵机 + 摆臂 + 可换配重。HD-1910 版已按此设计了 3D 打印件，生成脚本：
`scripts/hd1910_bench_parts.py`（cadquery，`uv run --with cadquery --no-project
python scripts/hd1910_bench_parts.py`），输出到 `assets/hd1910_bench/`：

| 部件 | 文件 | 说明 |
|---|---|---|
| 固定座 | `hd1910_bench_mount.stl` | 后板 + 桌脚，舵机背面贴板，4×M2.5 沉头螺丝贯穿舵机拧到板上（螺母锁背面），输出轴侧完全无遮挡 |
| 摆臂 | `hd1910_bench_arm.stl` | 120 mm 摆长（轴心→配重孔 CoM），25T 毂（Ø4.95 轴孔 + M3 紧定），末端 M8 配重孔 |
| 配重盘 | `hd1910_bench_weight.stl` | Ø40 M8 参考盘（PLA 太轻，实际配重用钢垫圈） |

**安装孔位来源**：从 `cad/HD-1910-C001-20260902.stp`（STEP 内部为
HL-2915-C002 装配，与 HD-1910 机械接口一致）用圆柱面特征分析提取：

- 外形 34×20×23 mm，输出轴心在安装面中心 (0,0)，摇臂 HORN Ø17.2（25T）
- **安装孔 4 个（M3，Ø3.0 贯穿整个 23mm 壳厚，顶部沉头）**：
  `(-8.0, 7.5) (8.0, 7.5) (-8.0, -22.5) (8.0, -22.5)`（完美对称）

> ⚠️ **2026-09-09 修正**：最初从 STEP 螺丝实体质心提取的
> `(4.90,-22.90) (-8.40,-19.40) (-8.40,4.50) (8.40,4.50)` 经圆柱面审计确认为
> **PHS 舵机外壳组装螺丝孔**（Ø0.84–Ø3.16 多级台阶、只贯穿到中壳、4 颗
> 螺丝实体占据），**不是安装孔**。真正的安装孔是对称的 M3 贯穿孔（组 B）。
> 真机到手仍需核对孔距（`servo_compatibility_check.md` §6-#4 同标注）。

**配重**：用 M8 螺杆 + 钢垫圈/大垫片堆叠凑 0.1 / 0.3 / 0.6 kg（精确称量），
打印盘作间隔垫。钢垫圈密度 ~7.8 g/cm³：Ø40×5 mm ≈ 47 g/片。

### 2.2 摆臂/配重参数（关键计算）

HD-1910 堵转 **1.176 N·m @ 6V**（12 kg·cm）。重力偏置 `τ_bias = m·g·L·sin(q)`，
需要覆盖轻/中/重三档负载，激励不同的摩擦区间：

| 档位 | 配重 m | 摆长 L | max 偏置扭矩 | 占堵转 | 主要识别目标 |
|---|---|---|---|---|---|
| 轻载 | 0.10 kg | 0.12 m | 0.118 N·m | 10% | armature、空载电流、Stribeck |
| 中载 | 0.30 kg | 0.12 m | 0.353 N·m | 30% | 负载相关摩擦、粘滞 |
| 重载 | 0.60 kg | 0.12 m | 0.706 N·m | 60% | 大负载摩擦、电流-扭矩斜率 |

- 惯量：0.10kg→1.44e-3、0.30kg→4.32e-3、0.60kg→8.64e-3 kg·m²
  （远大于估计 armature 0.002，动态响应足以识别反演惯量）
- 摆臂自身质量计入 `arm_mass`（Pendulum 模型用 `m_arm/3` 分布质量）
- **角零点对齐**：摆锤垂直向下 = 0 rad；记录时 `q_offset` 参数吸收安装偏差
- 配重做成可换圆柱/圆盘，用 3D 打印 + 螺丝固定，保证质量可精确称量

> 注意 HD-1910 空载转速 92 rpm = 9.63 rad/s。`sin(t²)` 在 t=6s 时速度峰值
> ~12 rad/s 会超速（伺服滞后于目标）——这本身可用于识别，但若饱和段过多，
> 把轨迹时长缩到 4-5s 或减小幅值，让速度峰值落在 6-9 rad/s。

## 3. 电子与通信

| 部件 | 要求 | 说明 |
|---|---|---|
| USB-TTL 半双工适配器 | Feetech 兼容（如 FTDI + 换向电路，或成品 USB 舵机调试器） | 与真机 Zero HAT 同架构 |
| 可调稳压电源 | 0-10 V / ≥3 A | 6 V / 7.4 V / 8.4 V 三档 |
| 电流测量 | 舵机反馈寄存器优先，电流钳交叉验证 | HD-1910 反馈含 Current |
| 电压测量 | log 里记录 input_volts（每拍） | BAM 电压控制律需要 |

- 波特率：**1 Mbps**（出厂默认，规格书 7-4）
- ID：单舵机测试用默认 ID 1
- 协议：TTL 半双工 8N1（与 STS3215/HL 同平台）

## 4. 数据采集

### 4.1 记录脚本

BAM 自带 `bam/feetech/record.py`（基于 pypot `FeetechSTS3215IO`）。需要：

1. **验证寄存器映射兼容性**：HD-1910 的 位置/速度/电流/电压/温度 寄存器地址
   是否与 STS3215 一致（拿到真机后第一个任务，用 Feetech 上位机对照）。
   - 兼容 → 直接用；不兼容 → 写 `bam/feetech/hd1910_record.py`
2. **补读电流**：现有脚本 `load = 0 # TMP`，HD-1910 有 Current 反馈，必须读
   真实电流（识别 R 与负载摩擦的关键）
3. 采样率尽量高（≥100 Hz，位置更新率 1 kHz）

### 4.2 采集矩阵

```
电压 {6 V, 7.4 V, 8.4 V} × 负载 {轻, 中, 重} × 轨迹 {sin_time_square, lift_and_drop,
up_and_down, sin_sin, nothing} × 重复 3 次
```
- 每次 6 s → 原始数据 ≈ 3×3×5×3 = 135 条 × 6s ≈ 13.5 分钟，加装夹/等待
  ~1 小时可采完
- **kp 档**：至少采两组（默认 kp 与一组高 kp），控制律参数需要多 kp 数据
- 每条 log 记录：`mass, arm_mass, length, kp, vin, motor, trajectory, entries[]`
  （entries 每拍含 position/speed/load/input_volts/temp/goal_position/torque_enable/timestamp）
- 环境：台架固定刚性桌面，避免振动；每档电压稳压后再采

## 5. 标定（bam.fit）

### 5.1 注册 HD-1910 执行器

`bam/actuators.py` 加：

```python
"hd1910": lambda: HD1910Actuator(Pendulum),
```

`bam/feetech/actuator.py`（或新文件）定义 `HD1910Actuator(VoltageControlledActuator)`：
- vin=7.4（默认；记录时从 log 覆盖）
- kp=200（初值，待标定）、error_gain=XL330 骨架值（待标定）、max_pwm=1.0
- max_current=1.75（初值；HD-1910 堵转 1.6-2.0 A）
- 初值 bounds 参考当前 `hd1910_m6.json` + 6V 图推导值

> ⚠️ 该注册改的是 bam 的 git 依赖。建议 fork `Rhoban/bam` 或提 PR 前先本地
> `[tool.uv.sources]` 指向本地路径（`better-actuator-models = { path = "...", editable = true }`）。

### 5.2 拟合命令

```bash
# 先固定 kt/R（已有实测依据），只优化摩擦项：
python -m bam.fit --logdir logs/hd1910/ \
    --actuator hd1910 --model m6 \
    --set '{"kt": 0.735, "R": 3.75, "armature": 0.002}' \
    --trials 20000 --workers 4 --output hd1910

# 全部放开（含 kt/R/armature）：
python -m bam.fit --logdir logs/hd1910/ \
    --actuator hd1910 --model m6 \
    --trials 50000 --workers 8 --validation_kp 1 --output hd1910_full

# GPU 并行（bam.mjlab.Simulator 一次 rollout 一批 log）：
# 需要把 fit.py 的 simulate.Simulator 换成 mjlab 版，或按 bam/mae.py 的 GPU 路径改造
```

- `--validation_kp`：分一部分日志做验证，防过拟合
- 产出 `hd1910/m6.json` → 替换 `hd1910_m6.json`

### 5.3 拟合顺序建议（避免局部最优）

1. **纯电压响应**（nothing 轨迹，力矩关断）：标 `armature` + 背驱摩擦
2. **中低速**（up_and_down / lift_and_drop）：标 `friction_base/stribeck/viscous`
3. **宽速**（sin_time_square）：标 kt/R + 全部负载摩擦
4. **多电压合并**：验证电压缩放一致性
5. 控制律参数（error_gain/kp_fw/max_current）单独标：对比「目标角 vs 实际角」
   的静态误差（kp/error_gain）与高速跟随（max_current/速度饱和）

## 6. 验证

- `bam.mae --logdir logs/hd1910/ --model m6 --actuator hd1910`：整体 MAE
- **交叉验证**：留出部分电压/负载组合，看预测误差（不能只拟合好训练集）
- 用标定参数重放 6V 运动特性图数据：模拟 vs 图曲线对比（kt/R/friction 应能
  复现图的电流-扭矩线与转速-扭矩线）
- 替换 `hd1910_m6.json` 后跑 microduck_rl 冒烟测试，确认训练不炸

## 7. 实施清单与预算

| 项 | 内容 | 预算参考 |
|---|---|---|
| 机械 | 3D 打印固定座 + 摆臂 + 配重（3 档） | 材料 ~50 元 |
| 电子 | USB-TTL 半双工适配器 / 舵机调试器 | 30-150 元 |
| 电源 | 可调稳压源（或现成 2S 电池 + 稳压） | 100-300 元 |
| 测量 | 电流钳（可选，交叉验证） | 100 元 |
| 软件 | 复用 BAM + microduck_rl，改 record 脚本 | 0 |

总预算 ≈ 300-600 元，无 GPU 也能跑（CPU 拟合慢些；GPU 并行可加速）。

## 7.5 公开数据调研（2026-09，全部 URL 已实际抓取验证）

### 结论：HD-1910-C001 本身无任何公开实测/台架数据

HD-1910 是飞特为开源小鸭做的**预售件**，唯一公开资料是官方规格书数据表
（`OpenMicroDuck/hardware_spec/servo/HD-1910-C001串型规格书-20260907.pdf`），
**零 BAM 参数、零 testbench 记录**。台架自测是必要路径。

### 可借用的同平台 Feetech 实测/标定数据（4 个来源）

1. **BAM main 分支 `bam/params/feetech_sts3215_7_4V/` m1–m6 全套**（最佳参考）
   —— 7.4V 正好落在 HD-1910 的 5–8.4V 供电范围内，同为 Feetech 电压控制平台。
   m6 拟合值（commit 31e5b97 后新增，比本仓库锁定的 mjlab_frictionloss 分支新）：
   `kt=1.275, R=2.753Ω, armature=0.0216, friction_base=0.0533, friction_viscous=0.0282,
   max_velocity=5.096 rad/s, command_delay=0.00498s, error_gain_ratio=1.162`
   （STS3215 是 20 kg·cm 级、扭矩约为 HD-1910 的 1.6–2×，绝对值按扭矩比缩放后作初值）
2. **`i1Cps/duck_mini_pro_headless/bam_bench/`** —— Waveshare ST3025（12V，
   **复用 STS3215 位置控制律**）的完整 BAM M1–M6 标定：
   - `fits/full_v2/params_m6.json`（机器人训练用）：`kt=1.586, R=2.825Ω,
     armature=0.00465, friction_base=0.0453, max_velocity=25.06 rad/s`
   - README 实测值：base friction≈0.045 N·m、armature≈0.0046、电流限制 2.47 A
   - **135 条原始记录已发布**：release `st3025-bam-data-v1` →
     `waveshare_st3025_raw.zip`（3.98 MB）
     https://github.com/i1Cps/duck_mini_pro_headless/releases/tag/st3025-bam-data-v1
   - 对应 BAM PR #12 已合并（2026-08-26），M6 训练/留出 MAE 23.0/19.8 mrad
3. **K-Scale `kscalelabs/kscale-assets/actuators/`** —— 正式 sys-id 的 Feetech 模型：
   `feetech_sts3215_12v.json`（max_torque=5.47 N·m、kt=1.0、R=2.21Ω、armature=0.04）、
   `feetech_sts3250.json`（8.72 N·m）。**均为 12V**，借用到 HD-1910 需按电压换算。
4. **`lukas/hexapod/sysid/`** —— **唯一带原始 CSV 轨迹的公开 STS3215 系统辨识数据集**：
   17+ 个真实测量数据集（air/loaded 双状态、25 Hz、t_send/t_recv 时间戳），
   含 `fit.py`（留出验证）和 Reality Gap 报告。协议可直接照搬。

### 方法论参考

- BAM issue #14（XL330 电流模式台架实测）：bus 电流模型 R=2.98Ω 恢复、
  error_gain 推导方法 —— 证明「硬停堵转 + 逐表回归」可精确标定 R 与增益
- `i1Cps` 的 `record.py`（15 KB）是**经过实战的记录脚本模板**，关键实践：
  零点用 HOME 中心（2047）而非重力垂（齿隙摩擦会卡住摆臂）；torque-on 前先
  goal-sync 防突跳；纯 P（D=0）自激看门狗；轨迹间温度门控；loop-rate 自检
  （≥150 Hz）；BENCH 遥测交叉校验；多 kp（4/8/16/32）扫描

### 对参数初值的影响

- `armature`：ST3025 实测 0.00465 与 STS3215 拟合 0.0216 之间差异大（同平台
  不同型号+台架差异），HD-1910 取 0.002–0.004 区间做 fit 初始范围
- `friction_base`：参考值 0.045（ST3025）– 0.053（STS3215）按扭矩比 0.5–0.6
  缩放 → 0.023–0.032，**高于**当前估算的 0.0095；fit 时把范围放宽到 0.005–0.06
- `max_velocity`：HD-1910 空载 92 rpm = 9.63 rad/s，可作固定约束
- `command_delay`：STS3215 拟合出 ~5 ms，HD-1910 训练端 delay_min_lag=3（60 ms
  仿真步级）是控制环延迟，两者不同源，fit 时单独标定

## 8. 风险与备注

1. **寄存器兼容性**：HD-1910 是预售件，Feetech 寄存器表可能与 STS3215 不同，
   真机到手先核对（这是最大的不确定项）
2. **error_gain / kp_fw 标定最难**：需要示波器看 PWM/电压占空比，或接受
   静态误差法近似；实在拿不到就保持估计值并用域随机化覆盖
3. **轨迹速度超限**：HD-1910 空载 9.63 rad/s，轨迹幅值/时长需按此调整
4. **台架与真机负载差异**：单摆是「已知动力学的简单负载」，真机关节的
   惯性/重力耦合复杂得多。标定参数保证电机层准确，机器人层仍需
   `infer_policy.py` 真机彩排 + 现场调 kp/DR
5. **数据质量**：采样率、电压稳定性、装配刚度直接影响拟合；每档配置
   先目视检查跟随曲线再批量采集
6. **bam 分支差异**：本仓库 pyproject 锁定 `mjlab_frictionloss` 分支（含
   bam.mjlab frictionloss 支持），而 STS3215 m1–m6 / ST3025 参数在 `main`
   分支——参考参数用手动拷贝，不要整支切换，除非先验证 main 也有 mjlab 接口

## 9. 已验证的 dry-run（2026-09，真机到手前）

真机到货前，已用公开的 ST3025 数据在**本环境端到端验证了标定全流程**：

1. 下载 ST3025 135 条原始记录：
   `https://github.com/i1Cps/duck_mini_pro_headless/releases/download/st3025-bam-data-v1/waveshare_st3025_raw.zip`
2. 数据格式与本地 bam 完全兼容（mass/arm_mass/length/kp/vin/motor/entries[]）
3. `uv run python -m bam.process --raw <raw> --logdir <out>` 处理 ✅
4. `uv run --with optuna --with cmaes python -m bam.fit \
     --logdir <processed> --output fit.json --actuator sts3215 --model m6 \
     --trials 60` 跑通 ✅（kp4 子集 60 trials，MAE 0.070——收敛不充分但
   流程全通；全量标定需 trials ≥ 2 万 + 多 worker）
5. **注意**：bam.fit 依赖 `optuna` + `cmaes`（未列入本项目 pyproject，是
   bam 的可选依赖），跑标定时用 `uv run --with optuna --with cmaes` 即可，
   不用污染项目依赖

真机数据到手后的标定命令（HD-1910）：

```bash
# 先固定 kt/R/armature（已有实测依据），只优化摩擦：
uv run --with optuna --with cmaes python -m bam.fit \
    --logdir logs/hd1910/processed --output hd1910 \
    --actuator sts3215 --model m6 \
    --set '{"kt": 0.735, "R": 3.75, "armature": 0.003}' \
    --trials 20000 --workers 4
# 或注册 HD1910Actuator（fork bam）后 --actuator hd1910
```
