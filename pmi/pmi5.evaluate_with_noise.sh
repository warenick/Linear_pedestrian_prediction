# conda init pytorch_env

# val_arr=('eth_hotel' 'biwi_eth' 'zara01' 'zara02' 'students' 'SDD')
val_arr=('SDD')
# pose_noise=('0.01 0.05 0.1 0.2 0.4 0.8 1.5')
id_noise=('1 2 3 4')
for dataset in ${val_arr[*]}
do
    # python3 pmi5.evaluate_with_noise.py --validate_with=${dataset}

    # for item in ${pose_noise[*]}
    # do
    #     python3 pmi5.evaluate_with_noise.py --pose_noise=${item} --validate_with=${dataset}
    # done


    for item in ${id_noise[*]}
    do
        python3 pmi5.evaluate_with_noise.py --id_noise=${item} --validate_with=${dataset}
    done
done