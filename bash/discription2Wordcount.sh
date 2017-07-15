# Get word count from descriptions. Useful for setting dictionary size in Word2vec
cat $1 | (tr ' ' '\n' | sort | uniq -c | awk '{print $2"|"$1}') > out_freq #to store word with its count
cat out_freq | sort -r -n --field-separator='|' -k2 > sorted_freq #sorting descending order



