# move to where 'SwiFT is located'
 
TRAINER_ARGS='--accelerator gpu --max_epochs 5 --precision 16 --num_nodes 1 --devices 1' # specify the number of gpus as '--devices'
MAIN_ARGS='--loggername tensorboard --classifier_module v6 --dataset_name DS003745 --image_path /content/SRP_MNI_to_TRs_minmax/img' 
DATA_ARGS='--batch_size 4 --num_workers 1  --input_type rest'
DEFAULT_ARGS='--project_name test_swift_ds003745'
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA True --clf_head_version v1 --downstream_task sex' #--use_scheduler --gamma 0.5 --cycle 0.5' 
RESUME_ARGS='--resume_ckpt_path pretrained_models/hcp_sex_classification.ckpt'

export CUDA_VISIBLE_DEVICES=0

python project/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20

