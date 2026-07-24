# Manifest 说明

配对 manifest 继续兼容旧项目格式，每行：

```text
/absolute/path/noisy.wav<TAB>/absolute/path/clean.wav
```

`on_the_fly` 模式下，`train_manifest` 是每行一个 clean WAV 的列表，
`noise_manifest` 是每行一个 noise WAV 的列表。验证与测试仍使用固定配对 manifest。

paired clean/noisy 的原始采样率和帧数必须一致，否则数据集构造时直接报错。训练随机
裁剪使用同一个起点；不足长度时右侧补零。任何防溢出缩放同时作用于 clean 和 noisy，
不会分别峰值归一化。

