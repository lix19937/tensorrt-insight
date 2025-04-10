import pandas as pd
import matplotlib.pyplot as plt
 

df = pd.read_csv('mainborad.csv')
fig = plt.figure()     

assert 2 == len(df.columns), "bad collect by grep"
key_str = ["CPU", "MEM"]
i = 1

for column in df.columns:
    fig.add_subplot(int(f"12{i}"))
    
    df[column].plot(kind='line', color='deepskyblue')   

    max_v = df[column].max()
    min_v = df[column].min()
    avg_v = df[column].mean()
    p90_v = df[column].quantile(0.9)

    # draw max value as a horizon line
    plt.axhline(y=max_v, color='r', linestyle='--')    
    plt.text(0, max_v, 'max:%.1f' % max_v, fontsize=12, color='r', ha='right')   

    # draw min  
    plt.axhline(y=min_v, color='y', linestyle='--')  
    plt.text(0, min_v, 'min:%.1f' % min_v, fontsize=12, color='y', ha='right')   

    # draw mean  
    print(f"{avg_v} {max_v}")
    if abs(avg_v - max_v) > 0.2:
        # plt.axhline(y=avg_v, color='lawngreen', linestyle='--')   
        plt.text(0, avg_v, 'mean:%.1f' % avg_v, fontsize=12, color='lawngreen', ha='right')     

    # draw p90  
    if abs(p90_v - max_v) > 0.2:
        # plt.axhline(y=p90_v, color='m', linestyle='--')   
        plt.text(0, p90_v, 'p90:%.1f' % p90_v, fontsize=12, color='gray', ha='right')   

    plt.ylabel('usage %')
    plt.xlabel('samping idx')

    plt.title(f'{key_str[i-1]} usage')
    i += 1
plt.show()   

# grep 42745  ./top_20220101001338.log | awk '{print $9 "," $10}' > mainborad.csv
# https://zhuanlan.zhihu.com/p/65220518
