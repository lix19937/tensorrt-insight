## 系统资源监测  

```sh   
bash ./perf  

grep 3497  ~/top_20220101001338.log | awk '{print $9}' > mainborad.csv    
grep chrome  ~/top_20220101001338.log | awk '{print $9}' > mainborad.csv

python3 ./draw.py




grep 42745  ./top_20220101001338.log | awk '{print $9 "," $10}' > mainborad.csv
python3 ./draw_multi_plots.py


```
