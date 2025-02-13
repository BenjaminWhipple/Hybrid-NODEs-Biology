#!/bin/bash

# Define the number of replicates
N=3

# Arrays of hyper-parameters
#learning_rates=(0.1)
#batch_times=(32 64)
#batch_times=(32)
#batch_times=(64)
#network_sizes=(5 15 25)
regularizations=(0.0 0.1 1.0 10.0)

script=Glycolysis_LikelihoodProfile_Sampling.py

start_time=$(date +%s)

# Loop over each parameter
# seq 0 10
for param in $(seq 0 10); do
    for regularization in "${regularizations[@]}"; do
        for (( i=1; i<=N; i++ )); do
            small_start_time=$(date +%s)

            echo "Running replicate $i for parameter $param and regularization $regularization"
            # Run the Python script with the current set of hyper-parameters
            python $script --reg $regularization --pi $param --rep $i

            small_end_time=$(date +%s)

            small_duration=$((small_end_time-small_start_time))
            echo "Program took $small_duration seconds."
        done
    done
done

end_time=$(date +%s)
duration=$((end_time-start_time))
echo "Program took $duration seconds."