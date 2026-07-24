# Third-party notices and migration record

## sp-uhh/sgmse

Source: <https://github.com/sp-uhh/sgmse>, `main` branch inspected on
2026-07-24.

Files used as mathematical/structural references:

- `LICENSE`, blob `44f44000468b7389f9230f3f5aff564466c68b05`
- `sgmse/sdes.py`, blob `14600fbc2f72a87dd30919da048481674c559ab3`
- `sgmse/backbones/ncsnpp.py`, blob
  `f5c810e7ec1a20ccbc20a61600d867ab8b1e7b7f`
- `sgmse/backbones/ncsnpp_utils/layerspp.py`, blob
  `948b06884f72ffc4ec2bae0fa91fae9f5c863412`
- `sgmse/backbones/ncsnpp_utils/up_or_down_sampling.py`, blob
  `cf7cd443fe7b6cf4f5d99afce4ed7a5f5a0fccf5`

The upstream repository is MIT licensed:

> MIT License
>
> Copyright (c) 2022 Signal Processing (SP), Universität Hamburg
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## NCSN++ ancestry

The upstream `ncsnpp.py` and `layerspp.py` state that they are adapted from
Google Research score-SDE code under Apache License 2.0:
<https://www.apache.org/licenses/LICENSE-2.0>.

The FIR ideas in upstream `up_or_down_sampling.py` are derived from NVIDIA
StyleGAN2. This workspace does not copy its custom CUDA `upfirdn2d` operator;
it reimplements the required `[1,3,3,1]` low-pass behavior with ordinary
PyTorch depthwise convolution, interpolation, and pooling.

## Local modifications

- Removed Lightning, W&B, registries, NumPy SDE math, and custom CUDA operators.
- Reorganized NCSN++ as explicit encoder/middle/decoder module lists.
- Added rectangular/odd-size alignment for 257-bin water-project STFTs.
- The network accepts real `[B,4,F,N]` and emits real `[B,2,F,N]`; ordinary
  Conv2d never receives native complex tensors.
- Added Python 3.8.5 / PyTorch 1.11 compatible typing and APIs.
- Added strict manifests, native DDP, EMA/RNG checkpoints, explicit metrics,
  deterministic validation, and standalone PC sampling.
- The OUVE variance follows the task specification; see README for the known
  formula difference from upstream.

