# README for Evaluation

## Data Preparation

Before starting to download the data, please create the `data` folder.

### COCO Images

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/coco && cd data/coco

# Step 2: Download and unzip image files
wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip && unzip val2014.zip
```

After preparation is complete, the directory structure is:

```shell
data/coco
├── train2014
├── val2014
```

### (Generalized) Referring Expression Segmentation

Follow the instructions below to prepare the data:
```shell
# Step 1: Create the data directory
mkdir -p data/res && cd data/res
```

Download converted files (by PSALM) ([Google Drive](https://drive.google.com/file/d/1EcC1tl1OQRgIqqy7KFG7JZz2KHujAQB3/view?usp=sharing) | [Baidu Cloud](https://pan.baidu.com/s/1NRGJGkJDUGn8CU-sU5ScOg) (code: hust)).

After preparation is complete, the directory structure is:
```shell
refcoco/
    refcoco_val.json
    refcoco_testA.json
    refcoco_testB.json
refcoco+/
    refcoco+_val.json
    refcoco+_testA.json
    refcoco+_testB.json
refcocog/
    refcocog_val.json
    refcocog_test.json
grefcoco/
    refcocog_val.json
    refcocog_testA.json
    refcocog_testB.json
```

### Multi-Granularity Referring Expression Segmentation

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/RefCOCOm && cd data/RefCOCOm

# Step 2: Download RefCOCOm_Benchmark_release_version
pip install gdown
gdown --id 1Qzt4cxyexe0xspsH_2oexylOKBI4Jim_ -O Refcocom_Benchmark.zip
unzip Refcocom_Benchmark.zip

cd ../..
```

After preparation is complete, the directory structure is:
```shell
data/RefCOCOm
 ├── annotations
 ├── images
 └── masks
 
```

### Multi-Class Open Vocabulary Segmentation

Download and extract the data from [here](https://huggingface.co/datasets/yayafengzi/Multi-Class-OV), then follow [here](https://github.com/bytedance/fc-clip/blob/main/datasets/README.md) to get images of ADE20k, Pascal VOC, and Pascal Context.

After preparation is complete, the directory structure is:
```shell
data/Multi-Class-OV
├── ADEChallengeData2016
│   ├── images
│   ├── annotations
│   │   └── validation_merged # extracted from ade.zip
├── pascal_ctx_d2
│   ├── images
│   ├── annotations_ctx59
│   │   └── validation_merged # extracted from pascal_ctx.zip
├── pascal_voc_d2
│   ├── images
│   ├── annotations_pascal20
│   │   └── validation_merged # extracted from pascal_voc.zip
├── ade.jsonl
├── pascal_ctx.jsonl
└── pascal_voc.jsonl
 
```

### MME

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/mme && cd data/mme

# Step 2: Download MME_Benchmark_release_version
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/MME_Benchmark_release_version.zip
unzip MME_Benchmark_release_version.zip

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/mme
 └── MME_Benchmark_release_version
```

### VQAv2

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/vqav2 && cd data/vqav2

# Step 2: Make sure you have downloaded COCO images
ln -s ../coco/train2014 ./
ln -s ../coco/val2014 ./
ln -s ../coco/test2015 ./

# Step 3: Download questions and annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip && unzip v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip && unzip v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip && unzip v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip && unzip v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip && unzip v2_Questions_Test_mscoco.zip

# Step 4: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_testdev.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/vqav2
├── train2014 -> ../coco/train2014
├── val2014 -> ../coco/val2014
├── test2015 -> ../coco/test2015
├── v2_mscoco_train2014_annotations.json
├── v2_mscoco_train2014_complementary_pairs.json
├── v2_mscoco_val2014_annotations.json
├── v2_OpenEnded_mscoco_test2015_questions.json
├── v2_OpenEnded_mscoco_test-dev2015_questions.json
├── v2_OpenEnded_mscoco_train2014_questions.json
├── v2_OpenEnded_mscoco_val2014_questions.json
├── vqav2_testdev.jsonl
├── vqav2_train.jsonl
└── vqav2_val.jsonl
```

### POPE

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/pope && cd data/pope

# Step 2: Make sure you have downloaded COCO images
ln -s ../coco/val2014 ./
wget https://github.com/OpenGVLab/InternVL/releases/download/data/llava_pope_test.jsonl

# Step 3: Download `coco` from POPE
mkdir -p coco && cd coco
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_adversarial.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_popular.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_random.json
cd ../../..
```

After preparation is complete, the directory structure is:

```shell
data/pope
├── coco
│   ├── coco_pope_adversarial.json
│   ├── coco_pope_popular.json
│   └── coco_pope_random.json
├── llava_pope_test.jsonl
└── val2014
```

## Evaluation

First run
```shell
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
CHECKPOINT=yayafengzi/ALToLLM-8B # Download the checkpoint from huggingface
```

For Generalized Referring Expression Segmentation, run

```shell
python eval/evaluate_referseg.py \
    --datasets 'grefcoco_val' \
    --checkpoint ${CHECKPOINT}
```

For Referring Expression Segmentation, run

```shell
python eval/evaluate_referseg.py \
    --datasets 'refcoco_val,refcoco+_val,refcocog_val' \
    --checkpoint ${CHECKPOINT}
```

For Multi-Granularity Referring Expression Segmentation, run

```shell
python eval/evaluate_referseg.py \
    --datasets 'refcocom_val_part_only,refcocom_val_object_part' \
    --checkpoint ${CHECKPOINT} \
    --data-dir ./data/RefCOCOm
```

For Multi-Class Open Vocabulary Segmentation, run

```shell
python eval/evaluate_mov.py \
    --datasets 'ade,pascal_voc,pascal_ctx' \
    --checkpoint ${CHECKPOINT} \
    --data-dir /mnt/wlf/datas/Multi-Class-OV \
    --image-dir /mnt/wlf/datas/Multi-Class-OV
```

If you want to evaluate the performance on above segmentation benchmarks in the adaptive length variant, you need to set `--text-mode adaptive`.

The following benchmarks are for general VQA. For MME, run

```shell
DIRNAME=`basename ${CHECKPOINT}`
cd eval/mme
python eval.py --checkpoint ${CHECKPOINT} --dynamic --max-num 4
python calculation.py --results_dir ${DIRNAME}
cd ../../
```

For VQAv2, run

```shell
python eval/evaluate_vqa.py \
    --datasets 'vqav2_val'  \
    --dynamic --max-num 4 \
    --checkpoint ${CHECKPOINT}
```

For POPE, run

```shell
python eval/pope/evaluate_pope.py \
    --dynamic --max-num 4 \
    --checkpoint ${CHECKPOINT}
```
