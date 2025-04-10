import pandas as pd
import matplotlib.pyplot as plt
 
# split single figure to show each resource   

df = pd.read_csv('mainborad.csv')
 
for column in df.columns:
    plt.figure()     
    df[column].plot(kind='line')   

    # draw max value as a horizon line
    plt.axhline(y=df[column].max(), color='r', linestyle='--')  #  
    # draw a text label nemed max:value for max value besize the line
    plt.text(0, df[column].max(), 'max:%d' % df[column].max(), fontsize=12, color='r', ha='right')  # 

    # draw min value as a horizon line
    plt.axhline(y=df[column].min(), color='y', linestyle='--')  #  
    # draw a text label nemed mean:value for mean value besize the line
    plt.text(0, df[column].min(), 'min:%d' % df[column].min(), fontsize=12, color='y', ha='right')  # 

    # draw mean value as a horizon line
    plt.axhline(y=df[column].mean(), color='g', linestyle='--')  #  
    # draw a text label nemed mean:value for mean value besize the line
    plt.text(0, df[column].mean(), 'mean:%d' % df[column].mean(), fontsize=12, color='g', ha='right')  #  

    # draw p90 value as a horizon line
    plt.axhline(y=df[column].quantile(0.9), color='b', linestyle='--')  # 
    # draw a text label nemed p90:value for p90 value besize the line
    plt.text(0, df[column].quantile(0.9), 'p90:%d' % df[column].quantile(0.9), fontsize=12, color='b', ha='right')  #  

    plt.ylabel('cpu usage %')
    plt.xlabel('samping idx')

    plt.title("CPU usage")     
    plt.show()   
