#!/bin/bash
# Put your command below to execute your program.
# Replace "./my-program" with the command that can execute your program.
# Remember to preserve " $@" at the end, which will be the program options we give you.
while getopts ":i:o:m:d:r" opt; do
  case $opt in
    i) query_file="$OPTARG"
    ;;
    o) out_file="$OPTARG"
    ;;
    m) model_dir="$OPTARG"
    ;;
    d) corpus_dir="$OPTARG"
    ;;
    r) feedback="feedback"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

if [ "$feedback" = "feedback" ]
then
    python3 cvsm.py --query_file $query_file --out_file $out_file --model_dir $model_dir --corpus_dir $corpus_dir -feedback
else
    echo "false"
    python3 cvsm.py --query_file $query_file --out_file $out_file --model_dir $model_dir --corpus_dir $corpus_dir
fi