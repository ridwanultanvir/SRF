ckpt_path=$1
eval_split_name=$2
a_feat_type=pann
a_feat_dim=2050
feat_root=...feat_root.../qvhighlights
a_feat_dir=${feat_root}/pann_features/
eval_path=data/qvhighlights_${eval_split_name}.jsonl
gpu=1

CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PYTHONPATH:. python srf/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
--a_feat_dir ${a_feat_dir} \
--a_feat_dim ${a_feat_dim} \
${@:3}
