#!/bin/bash

# Define these paths
PYTHON_BIN="~/miniconda3/envs/pytorch/bin/python3"
NCU_BIN="/usr/local/cuda/bin/ncu"

# Define these options
CUDA_DEVICE=1

CLOCK_SWEEP=1
CLOCK_MIN=900
CLOCK_MAX=900

NVML=true
NCU=true
TRACE=true
COMPILE=false

MODEL_TYPE="VisionModel"
MODEL="MobileViTModel"
CONFIG="workload_config/mobilevit/"
PREC="bf16 fp32"
BATCH="8 16 32 64"
SEQLEN="1"
MODE="prefill"

# Check if correct number of arguments provided
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <USER_ID> <PASSWORD> [SUBFOLDER]"
    echo "Example: $0 john_doe mypassword123"
    echo "Example: $0 john_doe mypassword123 custom_folder"
    exit 1
fi

# Get arguments
USER_ID="$1"
PASSWORD="$2"

# Generate timestamp in yy-mm-dd-hh-mm format
TIMESTAMP=$(date +%y-%m-%d-%H-%M)

# Define directory variables (modify as needed)
SAVE_DIR="save"        # Change this to your desired main directory name

# Set SUBFOLDER based on number of arguments
if [ $# -eq 3 ]; then
    SUBFOLDER="$3"     # Use the third argument as subfolder name
    PARSE=false 
else
    SUBFOLDER="${MODEL}_${TIMESTAMP}"  # Use default naming convention
    PARSE=true
fi

# Create SAVE directory if it doesn't exist
if [ ! -d "$SAVE_DIR" ]; then
    echo "Creating $SAVE_DIR directory..."
    mkdir -p "$SAVE_DIR"
    echo "$SAVE_DIR directory created."
else
    echo "$SAVE_DIR directory already exists."
fi

SUBFOLDER_PATH="${SAVE_DIR}/${SUBFOLDER}"

# Run code
CMD="${PYTHON_BIN} run.py --sudo_pwd ${PASSWORD} --python_bin_path ${PYTHON_BIN} --cuda_device ${CUDA_DEVICE}"

# If GPU_SWEEP
if [ "$CLOCK_SWEEP" -gt 0 ]; then
    CMD="${CMD} --gpu_clock_freq_sweep --gpu_min_freq ${CLOCK_MIN} --gpu_max_freq ${CLOCK_MAX}"
fi

# If NVML
if $NVML; then
    CMD="${CMD} --run_nvml --nvml_save_to ${SUBFOLDER_PATH} --nvml_poll_clock"
fi

# If NCU
if $NCU; then
    CMD="${CMD} --run_ncu --ncu_save_to ${SUBFOLDER_PATH} --ncu_bin_path ${NCU_BIN}"
fi

# If Trace
if $TRACE; then
    TRACE_SUBFOLDER_PATH="${SAVE_DIR}/${SUBFOLDER}_trace"
    CMD="${CMD} --run_trace --trace_save_to ${TRACE_SUBFOLDER_PATH}"
fi

# If Compile
if $COMPILE; then
    CMD="${CMD} --compile"
fi

CMD="${CMD} --model_type ${MODEL_TYPE} --model ${MODEL} --config_folder ${CONFIG} --precision ${PREC} --batch ${BATCH} --seqlen ${SEQLEN} --mode ${MODE}"

echo $CMD
eval $CMD

if [ "$PARSE" = true ]; then
    # Change ownership of the subfolder to the specified user
    if [ "$NCU" = true ]; then
        echo "Changing ownership of $SUBFOLDER_PATH to $USER_ID..."
        echo $PASSWORD | sudo -S chown -R "$USER_ID" "$SUBFOLDER_PATH"

        if [ $? -eq 0 ]; then
            echo "Successfully changed ownership of $SUBFOLDER_PATH to $USER_ID"
        else
            echo "Error: Failed to change ownership. Please check if user $USER_ID exists."
            exit 1
        fi
    fi

    # parse result
    if [[ "$NVML" = true || "$NCU" = true ]]; then
        PARSE_SUBFOLDER_PATH="${SAVE_DIR}/${SUBFOLDER}_parsed"
        PARSE_CMD="${PYTHON_BIN} parse_result.py --path_to_folder ${SUBFOLDER_PATH} --save_to ${PARSE_SUBFOLDER_PATH}"
        eval $PARSE_CMD

        # Create tar.gz archive of the subfolder
        echo "Creating tar.gz archive of $PARSE_SUBFOLDER_PATH..."
        tar -cvzf "${SUBFOLDER}.tar.gz" -C "$SAVE_DIR" "${SUBFOLDER}_parsed"

        if [ $? -eq 0 ]; then
            echo "Successfully created archive: ${SUBFOLDER}.tar.gz"
        else
            echo "Error: Failed to create tar.gz archive"
            exit 1
        fi
    fi

    # parse trace
    if [ "$TRACE" = true ]; then
        TRACE_PARSE_SUBFOLDER_PATH="${SAVE_DIR}/${SUBFOLDER}_trace_parsed"
        PARSE_CMD="${PYTHON_BIN} parse_trace.py --trace_path ${SAVE_DIR}/${SUBFOLDER}_trace --parsed_save_to ${SAVE_DIR}/${SUBFOLDER}_trace_parsed"
        eval $PARSE_CMD

        # Create tar.gz archive of the subfolder
        echo "Creating tar.gz archive of $TRACE_PARSE_SUBFOLDER_PATH..."
        tar -cvzf "workloads_${SUBFOLDER}.tar.gz" -C "$SAVE_DIR" "${SUBFOLDER}_trace_parsed"

        if [ $? -eq 0 ]; then
            echo "Successfully created archive: workloads_${SUBFOLDER}.tar.gz"
        else
            echo "Error: Failed to create tar.gz archive"
            exit 1
        fi
    fi
fi

echo "Script completed successfully!"
