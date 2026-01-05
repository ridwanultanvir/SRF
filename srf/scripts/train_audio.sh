dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip
a_feat_type=pann
results_root=result_audio
exp_id=exp_id

######## data paths
train_path=data/qvhighlights_train.jsonl
eval_path=data/qvhighlights_val.jsonl
eval_split_name=val

######## setup video+text features
feat_root=...feat_root.../qvhighlights

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

# audio features
if [[ ${a_feat_type} == "pann" ]]; then
  a_feat_dir=${feat_root}/pann_features/
  a_feat_dim=2050
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

# caption features
if [[ ${t_feat_type} == "clip" ]]; then
  c_feat_dir=${feat_root}/qvhighlights_captions # caption_features_qwen2-VL # caption_features_llava
  c_feat_dim=512
else
  echo "Wrong arg for c_feat_type."
  exit 1
fi


#### training
gpu=1
bsz=32
lr_drop=80
lr=0.0001
n_epoch=200
lw_saliency=1.0
seed=2017
VTC_loss_coef=0.3
CTC_loss_coef=0.5
caption_loss_coef=1.5
label_loss_coef=4
iter_num=5


CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PYTHONPATH:. python srf/train.py \
--dset_name ${dset_name} \
--seed $seed \
--VTC_loss_coef $VTC_loss_coef \
--CTC_loss_coef $CTC_loss_coef \
--caption_loss_coef $caption_loss_coef \
--label_loss_coef $label_loss_coef \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--a_feat_dir ${a_feat_dir} \
--a_feat_dim ${a_feat_dim} \
--c_feat_dir ${c_feat_dir} \
--c_feat_dim ${c_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--iter_num ${iter_num} \
--lr ${lr} \
--n_epoch ${n_epoch} \
--lw_saliency ${lw_saliency} \
--lr_drop ${lr_drop} \
${@:1}