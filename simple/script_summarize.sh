for data in AAPL AMZN FB GOOG MSFT NFLX
do
  for m in gbfry gamma ns vgamma3 vgamma4 nig ghd student
  do
    python3 summarize.py --filename ../data/data_minute_tech/${data}_min_train.pkl --model $m --no_states
  done
done
