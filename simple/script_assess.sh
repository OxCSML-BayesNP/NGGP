
for m in gbfry gamma ns vgamma3 vgamma4 nig student ghd
do
  python3 assess_ordered.py --filename ../data/data_minute_tech/$1_min_train.pkl --thin 25 --model $m
done
