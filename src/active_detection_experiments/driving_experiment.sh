#!/bin/bash

# Path to your Python script
python_script="driving_experiments\cli.py"

# Define nuumber of iterations for every setup
num_iters=10

# Define the maximum number of retries
max_retries=2

# Sleep before stating the experiment
sleep 30


# Loop from start to stop with the specified jump
for ((iter=0; iter<num_iters; iter+=1)); do
    for num_widths in 2 4 8; do

        # With attack
        attack_type="FullFrameInjection"
        # Initialize the retry count
        retry_count=0
        while [ $retry_count -lt $max_retries ]; do
            echo "Calling Python script with arguments: $num_widths $attack_type"
            python "$python_script" "--num_widths" "$num_widths" "--attack_type" "$attack_type"
            # Capture the exit code in a variable
            python_exit_code=$?

            if [ $python_exit_code -eq 0 ]; then
                echo "Python script exited with code 0. Success!"
                break  # Exit the loop if successful
            else
                echo "Python script exited with code $python_exit_code. Retrying..."
                retry_count=$((retry_count + 1))  # Increment the retry count
            fi

             sleep 1
        done
        # Check if the maximum number of retries was reached
        if [ $retry_count -eq $max_retries ]; then
            echo "Maximum number of retries ($max_retries) reached. Exiting."
        fi
    
    
        # Without attack
        # Initialize the retry count
        retry_count=0
        while [ $retry_count -lt $max_retries ]; do
        
            echo "Calling Python script with arguments: $num_width"
            python "$python_script" "--num_widths" "$num_widths"
            # Capture the exit code in a variable
            python_exit_code=$?

            if [ $python_exit_code -eq 0 ]; then
                echo "Python script exited with code 0. Success!"
                break  # Exit the loop if successful
            else
                echo "Python script exited with code $python_exit_code. Retrying..."
                retry_count=$((retry_count + 1))  # Increment the retry count
            fi

             sleep 1
        done
        # Check if the maximum number of retries was reached
        if [ $retry_count -eq $max_retries ]; then
            echo "Maximum number of retries ($max_retries) reached. Exiting."
        fi
    
    done
done
