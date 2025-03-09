declare -a StringArray=("CVC-300" "CVC-ClinicDB" "CVC-ColonDB" "ETIS-LaribPolypDB" "Kvasir")
# declare -a StringArray=("Kvasir")
# echo "Training Accuracy: "
# python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/PolypDataset/kvasirsegTRimage" --data_config config_polyp.yml  --model_config model_svdtuning.yml --save_path "./svdsam_polyp_1024/${dataset}" --gt_path "/media/ubuntu/New Volume/jay/PolypDataset/kvasirsegTRmask" --labels_of_interest "Polyp" --pretrained_path "svdsam_polyp_1024.pth"

echo "Testing Accuracy: "
for dataset in "${StringArray[@]}"; do
    echo "${dataset}"

    python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/Polyp2/TestDataset/${dataset}/images" --data_config config_polyp.yml  --model_config model_svdtuning.yml --save_path "./svdsam_polyp_1024/${dataset}" --gt_path "/media/ubuntu/New Volume/jay/Polyp2/TestDataset/${dataset}/masks" --labels_of_interest "Polyp" --pretrained_path "svdsam_polyp_1024.pth"

    echo "......................."
done