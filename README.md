# SGMSE+ 复数 STFT 域条件扩散降噪框架

这是原 DCCRN 仓库内的独立工作区。原有 `../train.py`、`../evaluate.py`、
`../infer.py`、Dataset 和 DCCRN 配置均未改变。本工作区实现可训练、可恢复、
torchrun/DDP、EMA、确定性验证、PC 采样、评估和长音频推理的 SGMSE+ 风格框架。

参考：

- Julius Richter et al., *Speech Enhancement and Dereverberation With
  Diffusion-Based Generative Models*, IEEE/ACM TASLP, 2023。
- 官方实现：<https://github.com/sp-uhh/sgmse>。

迁移的具体上游文件、blob SHA、MIT/Apache-2.0 来源和本地修改记录见
`THIRD_PARTY_NOTICES.md`。官方仓库当前推荐 Python 3.11；这里不安装官方包，而用
原生 PyTorch 局部迁移，以兼容目标 Python 3.8.5、PyTorch 1.11、torchaudio 0.11。

## 数据

配对模式继续使用旧格式：

```text
/path/noisy.wav<TAB>/path/clean.wav
```

在线模式分别提供 clean 和 noise 路径列表。配置：

```yaml
data:
  data_mode: on_the_fly
  train_manifest: data/train_clean.txt
  noise_manifest: data/train_noise.txt
  valid_manifest: data/valid_pairs.txt
  test_manifest: data/test_pairs.txt
```

paired 数据在 Dataset 构造时检查原始采样率和帧数严格相等。训练 clean/noisy 使用
同一裁剪起点；短音频右侧补零。在线混合使用：

```text
noise_scale = sqrt(clean_power /
                   (noise_power * 10 ** (snr_db / 10)))
noisy = clean + noise_scale * noise
```

不会分别归一化 clean/noisy。若启用 `shared_peak_limit`，只使用一个公共系数同步缩放
两者。静音 clean 明确报错，不静默作为正常样本。

## 复数 STFT 与数据流

small 配置沿用水声 DCCRN 已验证参数：`512/400/100`，当前配置约
298,838 个可训练参数。full 配置提供论文复现预设（当前配置约
50,337,806 个可训练参数）：
16 kHz、`win_length=n_fft=510`、`hop_length=128`、`crop_frames=256`。

```text
waveform [B,T]
 -> compressed STFT X0,Y [B,F,N] complex64
 -> Xt = OUVE_mean(X0,Y,t) + std(t) * CN(0,I)
 -> [Re(Xt),Im(Xt),Re(Y),Im(Y)] [B,4,F,N]
 -> NCSN++ score [B,2,F,N]
 -> complex score [B,F,N]
```

压缩与逆压缩：

```text
X_tilde = beta * abs(X)**alpha * exp(j*angle(X))
X = (abs(X_tilde)/beta)**(1/alpha) * exp(j*angle(X_tilde))
```

网络包含多分辨率 U-Net、BigGAN residual block、GroupNorm、SiLU、Gaussian Fourier
time embedding、低分辨率 attention、skip rescale、progressive input/output 和
`[1,3,3,1]` FIR 重采样。普通 Conv2d 只处理实值通道。

## OUVE 与损失

前向 SDE：

```text
dx_t = gamma * (y - x_t) dt + g(t) dw_t
g(t) = sigma_min * (sigma_max/sigma_min)**t
       * sqrt(2*log(sigma_max/sigma_min))
```

每个样本独立采样 `t` 和
`z=(randn_real+j*randn_imag)/sqrt(2)`，因此 `E|z|²=1`。主损失严格是：

```text
target_score = -z / std(t)
loss = mean((pred.real-target.real)^2 +
            (pred.imag-target.imag)^2)
```

SI-SNR/SDR 不混入默认训练损失。STFT、SDE 与复数运算显式 float32，NCSN++ 卷积可
AMP。

### 已知 OUVE 公式差异

任务指定的 variance 是：

```text
sigma_min² * (r**(2t) - exp(-2*gamma*t)) /
(gamma + log(r))
```

官方 `sgmse/sdes.py` 的解析式还含一个 `log(r)` 因子。当前代码按任务明确给出的公式
实现，因此不是该处的逐位官方复现；若用于论文严格对比，应先确认采用哪一约定并同步
训练与采样。

## 训练

在本目录执行：

```bash
python train.py --config configs/sgmse_water_small.yaml
```

full：

```bash
python train.py --config configs/sgmse_water_full.yaml
```

DDP：

```bash
torchrun --nproc_per_node=4 train.py \
  --config configs/sgmse_water_full.yaml
```

恢复：

```bash
python train.py --config configs/sgmse_water_full.yaml \
  --resume runs/sgmse_water_full/checkpoints/last.pt
```

checkpoint 包含 raw model、EMA、optimizer、scheduler、AMP scaler、epoch、
global_step、best metric、完整配置和 Python/NumPy/PyTorch/CUDA RNG state。
每轮计算确定性 validation score loss；按 `eval_interval` 在固定验证子集、固定 seed
上完整采样，默认按 `val/SI-SNRi` 保存 best，同时记录 `val/SDRi`。测试集不参与选择。

## 评估

```bash
python evaluate.py \
  --config configs/sgmse_water_full.yaml \
  --checkpoint runs/sgmse_water_full/checkpoints/best.pt \
  --split test \
  --output_dir results/sgmse_test
```

输出：

```text
per_file_metrics.csv
summary_metrics.json
enhanced_wavs/
evaluation.log
```

SDR 是尺度相关 signal-to-error SDR，不是 mir_eval/BSS-Eval 滤波 SDR：

```text
SDR(est,ref)=10log10((sum(ref²)+eps)/(sum((est-ref)²)+eps))
```

SI-SNR 对 est/ref 分别去均值后投影。SDRi/SI-SNRi 是增强输出减输入。指标前不 clip、
不分别归一化；静音或非有限 reference 返回 `valid=false` 和原因，汇总包含
`valid_count/invalid_count`、mean/std/median/p25/p75。

## 推理

```bash
python infer.py \
  --config configs/sgmse_water_full.yaml \
  --checkpoint runs/sgmse_water_full/checkpoints/best.pt \
  --input input.wav \
  --output enhanced.wav
```

默认优先整段频谱。可显式设置 `--chunk-seconds` 和 `--overlap-seconds`；显存 OOM 时
可回退到分块。块间用 Hann cross-fade overlap-add，不硬拼接，输出长度与输入一致。
默认 PC sampler 为 30 步、每步 1 次 corrector，因此通常 NFE=60。
`corrector: none` 可用于速度对比。probability-flow ODE 保留接口，列为 P1。

## 测试

```bash
pytest -q
```

测试覆盖 STFT/压缩 round trip、OUVE 广播与 Monte Carlo、复高斯能量、score loss、
NCSN++ shape、dummy sampler、全部指标边界、完整 checkpoint、CPU 优化 step 和
单文件推理。

## 风险与已知差异

- NCSN++ full 配置较大，水声数据上的最佳参数尚需实验校准；OOM 时优先减 batch、
  crop_frames、base_channels。
- 本地 FIR 使用普通 PyTorch depthwise filter，不是官方 StyleGAN2 自定义
  `upfirdn2d` 的逐位实现。
- 本地 NCSN++ 保留论文关键结构，但为矩形/奇数频率尺寸重新组织了模块拓扑，不能直接
  加载官方 checkpoint。
- 30 步 PC 推理远慢于 DCCRN；RTF 与 NFE 已输出，后续可增加 probability-flow ODE、
  更高阶 solver 或蒸馏。
- DDP 逻辑已实现；多机 NCCL 性能与目标 Linux 集群文件系统仍需现场验证。
