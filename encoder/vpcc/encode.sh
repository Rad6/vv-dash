#!/bin/bash

LOGFILE="encoding_output.log"  # Define the common log file
> "$LOGFILE"  # Clear previous log content before starting

for second in {0..9}
do
CHUNK_NUM=$(printf "%05d" $((second + 1)))  # Format as 5-digit number (00001, 00002, ..., 00010)
PIDS=()  # Array to store process IDs
	for rate in 1 2 3 4 5
	do
        START_FRAME=$(( (30 * $second) + 1 ))  # Compute start frame number dynamically
        RATE_MINUS_ONE=$((rate - 1))


        ./bin/PccAppEncoder \
            --configurationFolder=cfg/ \
            --config=cfg/common/ctc-common.cfg \
            --config=cfg/condition/ctc-all-intra.cfg \
            --config=cfg/sequence/longdress_vox10.cfg \
            --config="cfg/rate/ctc-r${rate}.cfg" \
            --uncompressedDataPath="../longdress/frame_%d.ply" \
            --nbThread=4 \
            --frameCount=30 \
            --startFrameNumber=${START_FRAME} \
            --reconstructedDataPath="./decoded-stream/frame_%d.ply" \
            --compressedStreamPath="./encoded-stream/rate${RATE_MINUS_ONE}_chunk${CHUNK_NUM}.bin" \
            >> "$LOGFILE" 2>&1 &  # Append output and errors to the log file

        PIDS+=($!)  # Store process ID
	done
    wait "${PIDS[@]}"  # Wait for all 16 processes before moving to the next rate
done

# Final wait to ensure all encoding tasks are done
wait
echo "All encoding tasks completed!" | tee -a "$LOGFILE"
