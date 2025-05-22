<div align="center">

# ALTo: Adaptive-Length Tokenizer for Autoregressive Mask Generation

![perform](imgs/cover.png)

</div>

## News
- [2025.5.21] We released the ALToLLM-8B, available [here](https://huggingface.co/yayafengzi/ALToLLM-8B). 
<!-- - [2025.5.22] We released the [paper](https://arxiv.org/abs/). -->

## Abstract
While humans effortlessly draw visual objects and shapes by adaptively allocating attention based on their complexity, existing multimodal large language models (MLLMs) remain constrained by rigid token representations. Bridging this gap, we propose ALTo, an adaptive length tokenizer for autoregressive mask generation. To achieve this, a novel token length predictor is designed, along with a length regularization term and a differentiable token chunking strategy. We further build ALToLLM that seamlessly integrates ALTo into MLLM. Preferences on the trade-offs between mask quality and efficiency is implemented by group relative policy optimization (GRPO). Experiments demonstrate that ALToLLM achieves state-of-the-art performance with adaptive token cost on popular segmentation benchmarks.

## Installation
```
conda env create -f environment.yml
```

## Demo
Run [inference_altollm.py](inference_altollm.py) to generate a segmentation mask for an object in an image.

## Training
You can train your own models based on our ALTo and ALToLLM Hugging Face models.

First, run:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
mkdir -p "runs"
```

To train ALTo, download the SA1B dataset from [here](https://ai.meta.com/datasets/segment-anything/) and prepare the data in the same format as [example/sa1b.jsonl](example/sa1b.jsonl). You can either download our pretrained ALTo model and continue training from it, or start training from scratch using the [Titok](https://github.com/bytedance/1d-tokenizer) and [SAM](https://github.com/facebookresearch/segment-anything]) models.

For stage 1 training, run:
```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29501 \
    trainers/main_multi_nodes.py \
    config=config/config_alto_stage1.py
```

For stage 1.5 training, run:
```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29501 \
    trainers/main_multi_nodes.py \
    config=config/config_alto_stage1_5.py
```

To train ALToLLM, prepare your data in the same format as [example/anns/seg_data_with_mask.jsonl](example/anns/seg_data_with_mask.jsonl).

Important keys contained in the JSONL files:
```
- "image": Source image.
- "mask": Mask image.
- "conversations": Conversations between human and GPT. The mask placeholder is <ALTo_Start><TOK_0>...<ALTo_End> for full-length mask generation and <ALTo_Start><TOK_1>...<ALTo_End> for adaptive-length mask generation.
```

For stage 2 training, run `bash scripts/train_altollm_stage2_sft.sh` to train ALToLLM.

For stage 3 training, run `bash scripts/train_altollm_stage3_grpo.sh` to train ALToLLM using GRPO.

## Evaluation

Follow the evaluation pipeline in [EVALUATE.md](EVALUATE.md).

## Citation
If you find this project useful in your research, please consider citing:

```BibTeX
@article{wang2025alto,
  title={ALTo: Adaptive-Length Tokenizer for Autoregressive Mask Generation},
  author={Wang, Lingfeng and Lin, Hualing and Chen, Senda and Wang, Tao and Cheng, Changxu and Zhong, Yangyang and Zheng, Dong and Zhao, Wuyue},
  journal={arXiv preprint arXiv:},
  year={2025}
}
```

## Acknowledgement
This project is built with reference to [InternVL](https://github.com/OpenGVLab/InternVL), [Titok](https://github.com/bytedance/1d-tokenizer) and [HiMTok](https://github.com/yayafengzi/LMM-HiMTok).

## License
```
Copyright 2025-UniUbi.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
