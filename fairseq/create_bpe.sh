# Run with sh create_bpe.sh directory
dir=$1

for SPLIT in train val
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$dir/$SPLIT.$LANG" \
    --outputs "$dir/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
