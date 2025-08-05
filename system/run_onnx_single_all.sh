#!/bin/bash
# for scope in "single" "full"
# do 
#     for i in {1..7}
#     do
#     echo "Running EENetB0_$i""_FC for $scope"
#     python3 system/run_onnx_single.py -m "EENetB0_$i""_FC" -s $scope
#     done
# done

# for scope in "single" "full"
# do 
#     for i in {1..4}
#     do
#     echo "Running ERNet50_$i""_FC for $scope"
#     python3 system/run_onnx_single.py -m "ERNet50_$i""_FC" -s $scope
#     done
# done

for scope in "single" "full"
do 
    for i in 12 #{1..6}
    do
    echo "Running EViT_$i""_FC for $scope"
    python3 system/run_onnx_single.py -m "EViT_$i""_FC" -s $scope
    done
done


# for scope in "single" "full"
# do 
#     for i in {1..6}
#     do
#     echo "Running EDeepSp_$i""_FC for $scope"
#     python3 system/run_onnx_single.py -m "EDeepSp_$i""_FC" -s $scope
#     done
# done
