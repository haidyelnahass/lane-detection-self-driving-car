# number of arguments after the script name
num_of_arguments=$#

if [ $num_of_arguments -lt 2 ] || [ $num_of_arguments -gt 3 ]; then
    echo "Usage: ./run.sh <input_file> <output_file> -d (for debug mode 'optional')"
    exit 1

elif [ $num_of_arguments -eq 2 ]; then
    # echo "your args: ${1} , ${2}"
    echo 'Processing video, please wait...'
    start=$SECONDS
    python lane-detection.py $1 $2
    duration=$((SECONDS - start))
    echo "Done!"
    echo "Time taken: "$duration" seconds"

    exit 0

elif [ $num_of_arguments -eq 3 ]; then
    # echo "your args: ${1}, ${2}, ${3}"
    if [ "$3" == "-d" ]; then
        echo "Running in debug mode..."
        echo 'Processing video, please wait...'
        start=$SECONDS

        python lane-detection.py $1 $2 $3

        duration=$((SECONDS - start))
        echo "Done!"
        echo "Time taken: "$duration" seconds"
        exit 0
    else
        echo "Usage: ./run.sh <input_file> <output_file> -d (for debug mode 'optional')"
        exit 1

    fi
fi
