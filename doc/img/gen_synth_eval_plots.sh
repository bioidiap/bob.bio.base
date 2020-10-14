OUTPUT_FOLDER="./generated_scores"

while getopts "hko:s" opt; do
    case ${opt} in
        h)  echo "Generates scores for multiple systems and evaluate them with different plots."
            echo
            echo "Usage: ./gen_roc_det_cmc.sh [OPTIONS]"
            echo
            echo "Options:"
            echo "    -h:"
            echo "      Prints this message and exit."
            echo
            echo "    -k:"
            echo "      Keeps generated temporary files instead of cleaning them at exit."
            echo
            echo "    -o output_folder:"
            echo "      Chooses the temporary scores folder [default:'./generated_scores']."
            echo
            echo "    -s:"
            echo "      skips the scores generation."
            echo
            exit 1
        ;;
        k) KEEP_SCORES=YES
        ;;
        o) OUTPUT_FOLDER=$OPTARG
        ;;
        s) SKIP_GEN=YES
        ;;
        *)
        ;;
    esac
done
echo

if [ "${SKIP_GEN}" = "YES" ]; then
    echo "Scores generation skipped."
else
    echo "Generating scores for 4 different systems in ${OUTPUT_FOLDER}..."
    echo "  >> Random"
    ../../bin/bob bio gen -mm  0 -mnm   0 -sp  5 -sn  5 -s 50 -p 5 "${OUTPUT_FOLDER}/random_system/"
    echo "  >> Bad"
    ../../bin/bob bio gen -mm 10 -mnm -10 -sp 10 -sn 10 -s 50 -p 5 "${OUTPUT_FOLDER}/bad_system/"
    echo "  >> Good"
    ../../bin/bob bio gen -mm 10 -mnm -10 -sp  5 -sn  5 -s 50 -p 5 "${OUTPUT_FOLDER}/good_system/"
    echo "  >> Perfect"
    ../../bin/bob bio gen -mm 10 -mnm -10 -sp  2 -sn  2 -s 50 -p 5 "${OUTPUT_FOLDER}/perfect_system/"
    echo "Done."
    echo

    echo "Generating scores for 4 different systems with unknown probes in ${OUTPUT_FOLDER}/dir..."
    echo "  >> Random"
    ../../bin/bob bio gen -mm  0 -mnm   0 -sp  5 -sn  5 -s 50 -p 5 -u 50 "${OUTPUT_FOLDER}/dir/random_system/"
    echo "  >> Bad"
    ../../bin/bob bio gen -mm 10 -mnm -10 -sp 10 -sn 10 -s 50 -p 5 -u 50 "${OUTPUT_FOLDER}/dir/bad_system/"
    echo "  >> Good"
    ../../bin/bob bio gen -mm 10 -mnm -10 -sp  5 -sn  5 -s 50 -p 5 -u 50 "${OUTPUT_FOLDER}/dir/good_system/"
    echo "  >> Perfect"
    ../../bin/bob bio gen -mm 10 -mnm -10 -sp  2 -sn  2 -s 50 -p 5 -u 50 "${OUTPUT_FOLDER}/dir/perfect_system/"
    echo "Done."
fi
echo

echo "Drawing the ROC, DET, CMC and DIR for all the systems..."
echo "  >> ROC"
bob bio roc "${OUTPUT_FOLDER}/perfect_system/scores-dev" "${OUTPUT_FOLDER}/good_system/scores-dev" "${OUTPUT_FOLDER}/bad_system/scores-dev" "${OUTPUT_FOLDER}/random_system/scores-dev" -o "roc_synthetic_comparison.pdf" -la " " --legends "System A,System B,System C,Random" -ts "ROC of synthetic systems"
echo "  >> DET"
bob bio det "${OUTPUT_FOLDER}/perfect_system/scores-dev" "${OUTPUT_FOLDER}/good_system/scores-dev" "${OUTPUT_FOLDER}/bad_system/scores-dev" "${OUTPUT_FOLDER}/random_system/scores-dev" -o "det_synthetic_comparison.pdf" -la " " --legends "System A,System B,System C,Random" -ts "DET of synthetic systems" -L 0.01,99,0.01,99
echo "  >> CMC"
bob bio cmc "${OUTPUT_FOLDER}/perfect_system/scores-dev" "${OUTPUT_FOLDER}/good_system/scores-dev" "${OUTPUT_FOLDER}/bad_system/scores-dev" "${OUTPUT_FOLDER}/random_system/scores-dev" -o "cmc_synthetic_comparison.pdf" --legends "System A,System B,System C,Random" -ts "CMC of synthetic systems"
echo "  >> DIR"
bob bio dir "${OUTPUT_FOLDER}/dir/perfect_system/scores-dev" "${OUTPUT_FOLDER}/dir/good_system/scores-dev" "${OUTPUT_FOLDER}/dir/bad_system/scores-dev" "${OUTPUT_FOLDER}/dir/random_system/scores-dev" -o "dir_synthetic_comparison.pdf" --legends "System A,System B,System C,Random" -ts "DIR of synthetic systems"
echo "Done."
echo

echo "Converting the pdf files to png files..."
echo "  >> ROC"
convert -density 200 roc_synthetic_comparison.pdf roc_synthetic_comparison.png
echo "  >> DET"
convert -density 200 det_synthetic_comparison.pdf det_synthetic_comparison.png
echo "  >> CMC"
convert -density 200 cmc_synthetic_comparison.pdf cmc_synthetic_comparison.png
echo "  >> DIR"
convert -density 200 dir_synthetic_comparison.pdf dir_synthetic_comparison.png
echo "Done."
echo

if [ "${KEEP_SCORES}" != "YES" ]; then
    echo "Cleaning."
    rm -r ${OUTPUT_FOLDER} roc_synthetic_comparison.pdf det_synthetic_comparison.pdf cmc_synthetic_comparison.pdf dir_synthetic_comparison.pdf
    echo "Done."
    echo
fi
