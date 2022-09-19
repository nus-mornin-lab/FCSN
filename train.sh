task=$1
model=$2
modelName=$3
encoderName=$4
learnSize=$5
gpuN=$6

folds=("0" "1" "2" "3" "4")

if [[ $task =~ ^(ISIC)$  ]]; then

    for i in ${folds[@]}; do

        echo "==================================================="
        echo "fold        : $i/4"
        echo "task        : $task"
        echo "model       : $model"
        echo "modelName   : $modelName"
        echo "encoderName : $encoderName"
        echo "==================================================="

        python3 ./train.py \
            --task $task \
            --model $model \
            --modelName $modelName \
            --encoderName $encoderName \
            --learnSize $learnSize \
            --fold $i \
            --gpuN $gpuN
            
    done
    
else

        python3 ./train.py \
            --task $task \
            --model $model \
            --modelName $modelName\
            --encoderName $encoderName \
            --learnSize $learnSize \
            --fold 0 \
            --gpuN $gpuN

fi;

