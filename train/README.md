# Training VisCoder

This directory contains training scripts and instructions for reproducing the VisCoder-3B and VisCoder-7B models using [ms-swift](https://github.com/modelscope/ms-swift).

## 1. Setup ms-swift

```bash
conda create -n swift python=3.10 -y
conda activate swift

git clone https://github.com/modelscope/ms-swift.git
cd ms-swift

pip install -e .
sh requirements/install_all.sh
pip install flash-attn -U --no-build-isolation

# Optional: for logging
pip install wandb
```

## 2. Prepare Data
Download the VisCode-200K dataset:

```bash
huggingface-cli download TIGER-Lab/VisCode-200K\
 --repo-type=dataset --resume-download --local-dir data
```

## 3. Run Training Scripts

```bash
sh train_viscoder_3b.sh
sh train_viscoder_7b.sh
```
Each script launches full fine-tuning with DeepSpeed and FlashAttention, using Qwen2.5-Coder as the base model.

## ⚠️ Notes
To start training quickly using ms-swift, you may need to remove the default dataset config:

```bash
rm ms-swift/swift/llm/dataset/data/dataset_info.json
echo "[]" > ms-swift/swift/llm/dataset/data/dataset_info.json
```
For detailed training options, refer to the [ms-swift CLI documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

