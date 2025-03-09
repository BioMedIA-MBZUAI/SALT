
echo "Testing Accuracy 1000: "
python generate_predictions.py --csv_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr/test_csvs/test_1000.csv" --data_config config_atr1.yml --model_config model_svdtuning.yml --pretrained_path "biastuning_atr1_thermal_512_bs8.pth" --save_path biastuning_atr1_thermal_512_bs8/1000 --root_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr" --img_folder_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset_old/range/thermal" --device "cuda:0" --labels_of_interest "Military Vehicle"
echo "......................."

echo "Testing Accuracy 2000: "
python generate_predictions.py --csv_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr/test_csvs/test_2000.csv" --data_config config_atr1.yml --model_config model_svdtuning.yml --pretrained_path "biastuning_atr1_thermal_512_bs8.pth" --save_path biastuning_atr1_thermal_512_bs8/2000 --root_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr" --img_folder_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset_old/range/thermal" --device "cuda:0" --labels_of_interest "Military Vehicle"
echo "......................."

echo "Testing Accuracy 3000: "
python generate_predictions.py --csv_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr/test_csvs/test_3000.csv" --data_config config_atr1.yml --model_config model_svdtuning.yml --pretrained_path "biastuning_atr1_thermal_512_bs8.pth" --save_path biastuning_atr1_thermal_512_bs8/3000 --root_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr" --img_folder_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset_old/range/thermal" --device "cuda:0" --labels_of_interest "Military Vehicle"
echo "......................."

echo "Testing Accuracy 4000: "
python generate_predictions.py --csv_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr/test_csvs/test_4000.csv" --data_config config_atr1.yml --model_config model_svdtuning.yml --pretrained_path "biastuning_atr1_thermal_512_bs8.pth" --save_path biastuning_atr1_thermal_512_bs8/4000 --root_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr" --img_folder_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset_old/range/thermal" --device "cuda:0" --labels_of_interest "Military Vehicle"
echo "......................."

echo "Testing Accuracy 5000: "
python generate_predictions.py --csv_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr/test_csvs/test_5000.csv" --data_config config_atr1.yml --model_config model_svdtuning.yml --pretrained_path "biastuning_atr1_thermal_512_bs8.pth" --save_path biastuning_atr1_thermal_512_bs8/5000 --root_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr" --img_folder_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset_old/range/thermal" --device "cuda:0" --labels_of_interest "Military Vehicle"
echo "......................."

echo "Testing Accuracy day: "
python generate_predictions.py --csv_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr/test_csvs/test_day.csv" --data_config config_atr1.yml --model_config model_svdtuning.yml --pretrained_path "biastuning_atr1_thermal_512_bs8.pth" --save_path biastuning_atr1_thermal_512_bs8/day --root_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr" --img_folder_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr/test_imgs" --device "cuda:0" --labels_of_interest "Military Vehicle"
echo "......................."

echo "Testing Accuracy night: "
python generate_predictions.py --csv_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr/test_csvs/test_night.csv" --data_config config_atr1.yml --model_config model_svdtuning.yml --pretrained_path "biastuning_atr1_thermal_512_bs8.pth" --save_path biastuning_atr1_thermal_512_bs8/night --root_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr" --img_folder_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/cegr/test_imgs" --device "cuda:0" --labels_of_interest "Military Vehicle"
echo "......................."