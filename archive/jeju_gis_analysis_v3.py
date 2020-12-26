
import pandas as pd
import os 
import folium
from folium import plugins

import plotly.express as px



os.getcwd()
df_path = './jeju/'
os.chdir(df_path)

df_5 = pd.read_csv('KRI-DAC_Jeju_data5.txt')
df_6 = pd.read_csv('KRI-DAC_Jeju_data6.txt')
df_7 = pd.read_csv('KRI-DAC_Jeju_data7.txt')
df_8 = pd.read_csv('KRI-DAC_Jeju_data8.txt')

df_all = pd.concat([df_5, df_6, df_7, df_8], axis=0)



hours = df_all['Time'].unique()
hours_dict = dict()
han_dict = dict()
foreign_dict = dict()
pyun_dict = dict()
super_dict = dict()
for i in range(len(hours)): 
    hours_dict[hours[i]] = 0
    han_dict[hours[i]] = 0
    pyun_dict[hours[i]] = 0
    super_dict[hours[i]] = 0
    foreign_dict[hours[i]] = 0

for i in range(len(df_all)):
    cur_hour = df_all.iloc[i]['Time']
    cur_disspent = df_all.iloc[i]['DisSpent']   
    hours_dict[cur_hour] += cur_disspent
    
    if df_all.iloc[i]['Type'] == '일반한식':
        han_dict[cur_hour] += cur_disspent
    if df_all.iloc[i]['Type'] == '편의점':
        pyun_dict[cur_hour] += cur_disspent
    if df_all.iloc[i]['Type'] == '슈퍼마켓':
        super_dict[cur_hour] += cur_disspent
    if df_all.iloc[i]['Type'] == '서양음식':
        foreign_dict[cur_hour] += cur_disspent



hours_dict.pop('x시', None)
han_dict.pop('x시', None)
pyun_dict.pop('x시', None)
super_dict.pop('x시', None)
foreign_dict.pop('x시', None)


def dict2df(dictionary, c1, c2): 
    df = pd.DataFrame.from_dict(dictionary, orient='index')
    df = df.reset_index()
    df.columns = [c1, c2]
    
    return df

hours_df = dict2df(hours_dict, 'hours', 'DisSpent')    
han_df = dict2df(han_dict, 'hours', 'DisSpent')
pyun_df = dict2df(pyun_dict, 'hours', 'DisSpent')
super_df = dict2df(super_dict, 'hours', 'DisSpent')
foreign_df = dict2df(foreign_dict, 'hours', 'DisSpent')


fig1 = px.pie(hours_df, values='DisSpent', names='hours', title='한시간별 재난지원금 사용금액비율')
fig1.write_image('한시간별_재난지원금_사용금액_비율.png')


fig2 = px.bar(hours_df, x='hours', y='DisSpent', title='한시간별 재난지원금 사용금액', color='DisSpent',
             labels={'DisSpent':'재난지원금 사용금액'}, height=400) 
fig2.write_image('한시간별_재난지원금_사용금액.png')



fig3 = px.bar(han_df, x='hours', y='DisSpent', title='일반한식 한시간별 재난지원금 사용금액', color='DisSpent',
             labels={'DisSpent':'재난지원금 사용금액'}, height=400) 
fig3.write_image('일반한식_한시간별_재난지원금_사용금액.png')

fig4 = px.bar(pyun_df, x='hours', y='DisSpent', title='편의점 한시간별 재난지원금 사용금액', color='DisSpent',
             labels={'DisSpent':'재난지원금 사용금액'}, height=400) 
fig4.write_image('편의점_한시간별_재난지원금_사용금액.png')

fig5 = px.bar(super_df, x='hours', y='DisSpent', title='슈퍼마켓 한시간별 재난지원금 사용금액', color='DisSpent',
             labels={'DisSpent':'재난지원금 사용금액'}, height=400) 
fig5.write_image('슈퍼마켓_한시간별_재난지원금_사용금액.png')

fig6 = px.bar(foreign_df, x='hours', y='DisSpent', title='외국음식 한시간별 재난지원금 사용금액', color='DisSpent',
             labels={'DisSpent':'재난지원금 사용금액'}, height=400) 
fig6.write_image('외국음식_한시간별_재난지원금_사용금액.png')
    
    
    

    
    
    
    
    
    
    






