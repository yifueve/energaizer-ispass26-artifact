# Define these paths
NVML=true
NCU=false
TRACE=false

# Check if correct number of arguments provided
if [ $# -lt 5 ]; then
    echo "Usage: $0 <USER_ID> <PASSWORD> <SAVE_DIR> <SUBFOLDER> <PYTHON_BIN>"
    echo "Example: $0 john_doe mypassword123 save_folder sub_folder_to_parse python_bin"
    exit 1
fi

# Get arguments
USER_ID="$1"
PASSWORD="$2"
SAVE_DIR="$3"
SUBFOLDER="$4"
PYTHON_BIN="$5"

SUBFOLDER_PATH="${SAVE_DIR}/${SUBFOLDER}"

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