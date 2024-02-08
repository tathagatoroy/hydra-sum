#!/bin/bash


#myscript > ~/myscript.log 2>&1 &


echo "starting  experiment 1 in the background"
./exp_1.sh > ~/gpu_0.log 2>&1 &
pid_1=$!

echo "starting  experiment 0 in the background"
./exp_2.sh > ~/gpu_1.log 2>&1 &
pid_2=$!

#

#echo "starting  experiment 0 in the background"
#./exp_4.sh > ~/gpu_3.log 2>&1 &
#get pid
#pid_4=$!

#echo all the pid
echo "pid_1: $pid_1"
echo "pid_2: $pid_2"
#echo "pid_3: $pid_3"
#echo "pid_4: $pid_4"

#wait for all pid
#wait $pid_1 $pid_2 $pid_3 $pid_4
#wait $pid_1 $pid_3 $pid_2
wait $pid_1 $pid_2
#os.environ["CUDA_VISIBLE_DEVICES"]= "2"


echo "all done"


