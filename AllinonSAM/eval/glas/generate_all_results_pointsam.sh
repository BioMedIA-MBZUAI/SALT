
echo "Testing Accuracy: "

python generate_predictions_pointsam.py --data_folder "/media/ubuntu/New Volume/jay/GLAS/archive/test" --data_config config_glas.yml  --model_config model_svdtuning.yml --save_path "./sam_point_glas" --gt_path "/media/ubuntu/New Volume/jay/PolypDataset/TestDataset/TestDataset/test/masks" --labels_of_interest "Glands"
