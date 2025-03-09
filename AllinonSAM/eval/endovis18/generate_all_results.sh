declare -a StringArray=("background tissue" "surgical instrument" "kidney parenchyma" "covered kidney" "thread" "clamps" "suturing needle" "suction instrument" "small intestine" "ultrasound probe")

for label in "${StringArray[@]}"; do
    echo "${label}"

    python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_1-20230624T000458Z-001/seq_1/left_frames" --data_config config_endovis18_test.yml  --model_config model_svdtuning.yml --save_path "./svdshiftscale_ev18_bs16_tal_focal075alpha2_1e-4_epoch150/seq1/${label}" --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_1-20230624T000458Z-001/seq_1/labels" --labels_of_interest "${label}" --pretrained_path "svdshiftscale_ev18_bs16_tal_focal075alpha2_1e-4_epoch150.pth"

    echo "......................."

    python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_2-20230624T000507Z-001/seq_2/left_frames" --data_config config_endovis18_test.yml  --model_config model_svdtuning.yml --save_path "./svdshiftscale_ev18_bs16_tal_focal075alpha2_1e-4_epoch150/seq2/${label}" --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_2-20230624T000507Z-001/seq_2/labels" --labels_of_interest "${label}" --pretrained_path "svdshiftscale_ev18_bs16_tal_focal075alpha2_1e-4_epoch150.pth"


    echo "......................."

    python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_3-20230624T000508Z-001/seq_3/left_frames" --data_config config_endovis18_test.yml  --model_config model_svdtuning.yml --save_path "./svdshiftscale_ev18_bs16_tal_focal075alpha2_1e-4_epoch150/seq3/${label}" --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_3-20230624T000508Z-001/seq_3/labels" --labels_of_interest "${label}" --pretrained_path "svdshiftscale_ev18_bs16_tal_focal075alpha2_1e-4_epoch150.pth"


    echo "......................."

    python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_4-20230624T000509Z-001/seq_4/left_frames" --data_config config_endovis18_test.yml  --model_config model_svdtuning.yml --save_path "./svdshiftscale_ev18_bs16_tal_focal075alpha2_1e-4_epoch150/seq4/${label}" --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_4-20230624T000509Z-001/seq_4/labels" --labels_of_interest "${label}" --pretrained_path "svdshiftscale_ev18_bs16_tal_focal075alpha2_1e-4_epoch150.pth"


    echo "......................."
done