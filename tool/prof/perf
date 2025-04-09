
#!/bin/bash
 
# top_20250409135102
filename="top_$(date "+%Y%m%d%H%M%S").log"
 
while true; do
    timestamp=$(date "+%Y%m%d%H%M%S")

    echo "Current time: $timestamp"

    echo $timestamp >> "$filename"

    top -b -n 1 >> "$filename"

    # 1 s
    sleep 1
done


#  bash  ./perf.sh  
#
#  grep havp_nni_node ~/top_20250409135102.log
#  
#  # 0 means full line 
#  # 9 CPU; 10 mem
# grep havp_nni_node  ~/top_20250409135646.log | awk '{print $9}' > mainborad.csv
#
#  
