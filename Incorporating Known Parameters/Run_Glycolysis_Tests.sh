#!/bin/bash

# Define the number of replicates
N=30

# Arrays of hyper-parameters
learning_rates=(0.1)
batch_times=(32)
network_sizes=(5 15 25)

script=Glycolysis_KnownParams_Tests.py

start_time=$(date +%s)

# Loop over each combination of hyper-parameters
for lr in "${learning_rates[@]}"; do
    for batch in "${batch_times[@]}"; do
        for width in "${network_sizes[@]}"; do
            for (( i=11; i<=N; i++ )); do
                small_start_time=$(date +%s)

                echo "Running replicate $i for learning rate $lr and batch time $batch and network width $width"
                # Run the Python script with the current set of hyper-parameters
                python $script --lr $lr --bt $batch --w $width --rep $i

                small_end_time=$(date +%s)

                small_duration=$((small_end_time-small_start_time))
                echo "Program took $small_duration seconds."
            done
        done
    done
done

end_time=$(date +%s)
duration=$((end_time-start_time))
echo "Program took $duration seconds."