#!/bin/bash

# Path to your Python script
python_script="C:\Users\user\Desktop\Dror\video-manipulation-detection\src\GUI\cli.py"

# Define start, stop, and jump values
exposure_start=2300
exposure_stop=4000
expoure_jump=300

diff_start=-200
diff_stop=200
diff_jump=80


# Define the maximum number of retries
max_retries=2

sleep 10

# Loop from start to stop with the specified jump
for ((exposure=exposure_start; exposure<=exposure_stop; exposure+=expoure_jump)); do
    for ((diff=diff_start; diff<=diff_stop; diff+=diff_jump)); do
        if [ $diff -eq 0 ]; then
            echo "Skipping iteration because value equals 0"
        continue
    fi
        
        # Initialize the retry count
        retry_count=0

        while [ $retry_count -lt $max_retries ]; do
        
            echo "Calling Python script with arguments: $exposure $diff"
            python "$python_script" "--exposure" "$exposure" "--diff" "$diff"
            # Capture the exit code in a variable
            python_exit_code=$?

            if [ $python_exit_code -eq 0 ]; then
                echo "Python script exited with code 0. Success!"
                break  # Exit the loop if successful
            else
                echo "Python script exited with code $python_exit_code. Retrying..."
                retry_count=$((retry_count + 1))  # Increment the retry count
            fi

             sleep 5  # Sleep for 1 second (adjust as needed)
        done
        # Check if the maximum number of retries was reached
        if [ $retry_count -eq $max_retries ]; then
            echo "Maximum number of retries ($max_retries) reached. Exiting."
        fi
    done
done




