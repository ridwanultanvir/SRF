# [AAAI 2026 Oral] See, Rank and Filter: Important Word-Aware Clip Filtering via Scene Understanding for Moment Retrieval and Highlight Detection

[![arXiv](https://img.shields.io/badge/arXiv-2511.22906-B31B1B?style=flat&logo=arXiv&logoColor=white)](https://arxiv.org/abs/2511.22906)

Official Repository for "See, Rank and Filter: Important Word-Aware Clip Filtering via Scene Understanding for Moment Retrieval and Highlight Detection".

Accepted at **AAAI 2026 Oral🔥**

by [YuEun Lee](https://www.linkedin.com/in/yueun-lee-86537a350), [Jung Uk Kim*](https://scholar.google.co.kr/citations?user=JMZ80R8AAAAJ&hl=en)

(* indicate corresponding author)

---
## 🛠️ Installation
### 0. Clone this repository
```
git clone https://github.com/VisualAIKHU/SRF.git
cd SRF
```
### 1. Prepare datasets
#### QVHighlights
- Download the official feature files for the QVHighlights dataset from Moment-DETR.

- Download [moment_detr_features.tar.gz](https://drive.google.com/file/d/1LXsZZBsv6Xbg_MmNQOezw0QYKccjcOkP/view)(8GB) and extract it under the `../features` directory.

- Additionally, You can download the [caption_features_internvl.tar.gz]().

#### TVSum
- Download the feature files from [UMT](https://github.com/tencentarc/umt?tab=readme-ov-file).

- Additionally, You can download the [TVSum_caption_features.tar.gz]().

#### Charades-STA
- Download the feature files from [UMT](https://github.com/tencentarc/umt?tab=readme-ov-file).

- Additionally, You can download the [Charades-STA_caption_features.tar.gz]().

### 2. Install requirements
```
conda create -n srf python=3.11.8
conda activate srf

pip install -r requirements.txt
```

## 🚀 Training
### QVHighlights
You can train the model using only video features or both video and audio features by running the shell below.
```
bash srf/scripts/train.sh
bash srf/scripts/train_audio.sh
```
You need to modify `results_root`, `exp_id` and `feat_root` before running the shell and make sure each feature directory(`v_feat_dirs`, `t_feat_dir` and `c_feat_dir`) is set correctly.

## 📊 Evaluation
### QVHighlights
You can generate `hl_val_submission.jsonl` and `hl_test_submission.jsonl` after training by running the shell below.
```
bash srf/scripts/inference.sh {results_path}/model_best.ckpt 'val'
bash srf/scripts/inference.sh {results_path}/model_best.ckpt 'test'
```
where `results_path` is the path to the saved checkpoint.

For more details for submission, [check standalone_eval/README.md](https://github.com/VisualAIKHU/SRF/blob/main/standalone_eval/README.md)

## 🔖 Citation
```
@article{lee2025see,
  title={See, Rank, and Filter: Important Word-Aware Clip Filtering via Scene Understanding for Moment Retrieval and Highlight Detection},
  author={Lee, YuEun and Kim, Jung Uk},
  journal={arXiv preprint arXiv:2511.22906},
  year={2025}
}
```

## 💛 Acknowledgement
Our codes benefits from the excellent [TR-DETR](https://github.com/mingyao1120/TR-DETR#).
