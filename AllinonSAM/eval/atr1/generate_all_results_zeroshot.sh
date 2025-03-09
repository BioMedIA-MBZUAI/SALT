
echo "Testing Accuracy 1000: "
python generate_predictions.py --csv_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co/test_csvs/test_1000.csv" --data_config config_atr1.yml --model_config model_svdtuning.yml  --save_path sam-zs_i1co/1000 --root_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co" --img_folder_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset_old/range/visible" --device "cuda:0" --labels_of_interest "Military Vehicle"
echo "......................."

echo "Testing Accuracy 2000: "
python generate_predictions.py --csv_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co/test_csvs/test_2000.csv" --data_config config_atr1.yml --model_config model_svdtuning.yml  --save_path sam-zs_i1co/2000 --root_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co" --img_folder_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset_old/range/visible" --device "cuda:0" --labels_of_interest "Military Vehicle"
echo "......................."

echo "Testing Accuracy 3000: "
python generate_predictions.py --csv_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co/test_csvs/test_3000.csv" --data_config config_atr1.yml --model_config model_svdtuning.yml  --save_path sam-zs_i1co/3000 --root_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co" --img_folder_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset_old/range/visible" --device "cuda:0" --labels_of_interest "Military Vehicle"
echo "......................."

echo "Testing Accuracy 4000: "
python generate_predictions.py --csv_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co/test_csvs/test_4000.csv" --data_config config_atr1.yml --model_config model_svdtuning.yml  --save_path sam-zs_i1co/4000 --root_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co" --img_folder_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset_old/range/visible" --device "cuda:0" --labels_of_interest "Military Vehicle"
echo "......................."

echo "Testing Accuracy 5000: "
python generate_predictions.py --csv_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co/test_csvs/test_5000.csv" --data_config config_atr1.yml --model_config model_svdtuning.yml  --save_path sam-zs_i1co/5000 --root_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co" --img_folder_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset_old/range/visible" --device "cuda:0" --labels_of_interest "Military Vehicle"
echo "......................."

echo "Testing Accuracy day: "
python generate_predictions.py --csv_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co/test_csvs/test_day.csv" --data_config config_atr1.yml --model_config model_svdtuning.yml  --save_path sam-zs_i1co/day --root_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co" --img_folder_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co/test_imgs" --device "cuda:0" --labels_of_interest "Military Vehicle"
echo "......................."

echo "Testing Accuracy night: "
python generate_predictions.py --csv_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co/test_csvs/test_night.csv" --data_config config_atr1.yml --model_config model_svdtuning.yml  --save_path sam-zs_i1co/night --root_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co" --img_folder_path "/media/ubuntu/New Volume/jay/ATR/atr_dataset/i1co/test_imgs" --device "cuda:0" --labels_of_interest "Military Vehicle"
echo "......................."