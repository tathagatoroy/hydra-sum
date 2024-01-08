#!/bin/bash


#myscript > ~/summarization/hydra-sum/modellingmyscript.log 2>&1 &


# echo "starting baseline experiment in the background"
# ./test_model_gpu_0.sh > ~/summarization/hydra-sum/modellinggpu_0.log 2>&1 &

# echo "starting 1 experiment in the background"
# ./test_model_gpu_1.sh > ~/summarization/hydra-sum/modellinggpu_1.log 2>&1 &

# echo "starting 2 experiment in the background"
# ./test_model_gpu_2.sh > ~/summarization/hydra-sum/modellinggpu_2.log 2>&1 &

# echo "starting 3 experiment in the background"
# ./test_model_gpu_3.sh > ~/summarization/hydra-sum/modelling/gpu_3.log 2>&1 &

# wait 

#!/bin/bash


#myscript > ~/summarization/hydra-sum/modelling/myscript.log 2>&1 &


echo "starting  gpu 0 in the background"
./test_model_gpu_0.sh > ~/summarization/hydra-sum/modelling/gpu_0.log 2>&1 &
#get pid 
pid_1=$!

echo "starting  gpu 1 in the background"
./test_model_gpu_1.sh > ~/summarization/hydra-sum/modelling/gpu_1.log 2>&1 &
#get pid
pid_2=$!

echo "starting  starting 2 in the background"
./test_model_gpu_2.sh > ~/summarization/hydra-sum/modelling/gpu_2.log 2>&1 &
#get pid
pid_3=$!

#echo "starting  gpu 3 in the background"
#./test_model_gpu_3.sh > ~/summarization/hydra-sum/modelling/gpu_3.log 2>&1 &
#get pid
#pid_4=$!

#echo all the pid
echo "pid_1: $pid_1"
echo "pid_2: $pid_2"
echo "pid_3: $pid_3"
#echo "pid_4: $pid_4"

#wait for all pid
#wait $pid_1 $pid_2 $pid_3 $pid_4
wait $pid_1 $pid_2 $pid_3


echo "all done"

