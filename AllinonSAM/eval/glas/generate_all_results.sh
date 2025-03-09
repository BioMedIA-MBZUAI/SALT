
echo "Testing Accuracy: "

python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/GLAS/archive/test" --data_config config_glas.yml  --model_config model_svdtuning.yml --save_path "./lorasam" --gt_path "/media/ubuntu/New Volume/jay/PolypDataset/TestDataset/TestDataset/test/masks" --labels_of_interest "Glands" --pretrained_path "lora_glas_1024.pth"

echo "......................."

# echo "Training Accuracy: "
# python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/GLAS/archive/train" --data_config config_glas.yml  --model_config model_svdtuning.yml --save_path "./tmp" --gt_path "/media/ubuntu/New Volume/jay/GLAS/archive" --labels_of_interest "Glands" --pretrained_path "svdtuning_shiftscale_glas_tal_1repeat_CE_dice_1e-3_sz_1024.pth"
