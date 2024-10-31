#!/bin/bash

# chmod +x nvidia_smi_logger.sh
# ./nvidia_smi_logger.sh <frequency> <output_directory>


# Default values for frequency and output directory
DEFAULT_FREQUENCY=3.0
DEFAULT_OUTPUT_DIR="."

# Check for command-line arguments
if [ "$#" -ge 1 ]; then
    FREQUENCY=$1
else
    FREQUENCY=$DEFAULT_FREQUENCY
fi

if [ "$#" -ge 2 ]; then
    OUTPUT_DIR=$2
else
    OUTPUT_DIR=$DEFAULT_OUTPUT_DIR
fi

# Ensure the output directory exists
mkdir -p $OUTPUT_DIR

# Set the output file path including date and time
OUTPUT_FILE="$OUTPUT_DIR/gpu_usage_log_$(date '+%Y-%m-%d_%H-%M-%S').csv"

# Clear the output file if it exists
> $OUTPUT_FILE

echo "Logging GPU memory usage and utilization every $FREQUENCY seconds..."
echo "Saving output to $OUTPUT_FILE"
echo "Press [CTRL+C] to stop logging."

# Flag to track if header has been written
HEADER_WRITTEN=false

# Infinite loop to record GPU usage
while true; do
    # Append timestamp to the output file
    #echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" >> $OUTPUT_FILE
    
    # Write header only if it hasn't been written yet
    if [ "$HEADER_WRITTEN" = false ]; then
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.total,memory.used,memory.free --format=csv,nounits | head -n 1 >> $OUTPUT_FILE
        HEADER_WRITTEN=true
    fi
    
    # Append GPU usage data to the file
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.total,memory.used,memory.free --format=csv,noheader,nounits >> $OUTPUT_FILE
    
    # Wait for the specified frequency
    sleep $FREQUENCY
done
