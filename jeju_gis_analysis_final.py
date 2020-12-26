
import pandas as pd
import os 
import folium
from pyproj import Proj, transform
import matplotlib as mpl
import pickle 
import numpy as np
from folium.plugins import HeatMap
import time
import plotly.express as px
import plotly.graph_objects as go
from tqdm.notebook import tqdm
import geopandas as gpd
import json
from tqdm import tqdm
from plotly.subplots import make_subplots
from shapely.geometry import Point

#=============================================================================================================================================
code_table = pd.read_csv('extra_data/code_table.csv', encoding = 'utf-8-sig', sep = ',', index_col = False)
color_dict = []
for i in range(len(code_table)):
    color_dict.append('#' + format(np.random.randint(0, 255), '02x').upper() + format(np.random.randint(0, 255), '02x').upper() + format(np.random.randint(0, 255), '02x').upper())
code_table['color'] = color_dict
code_table.to_csv('extra_data/code_table.csv', encoding = 'utf-8-sig', index = False)
type_dict = {'a' : '영세', 'b' : '일반', 'c' : '중소' , 'd' : '중소1', 'e' : '중소2'}
time_dict = {'AM1' : '0시 ~ 6시', 'AM2' : '6시 ~ 12시', 'PM1' : '12시 ~ 18시' , 'PM2' : '18시 ~ 24시'}
code_dict = {0: '음식', 1: '소매', 2 : '생활 서비스', 3: '학문, 교육', 4: '숙박', 5: '관광, 여가 ,오락', 6:'문화, 예술, 종교', 7:'도매, 유통, 무역',\
             8: '부동산', 9: '스포츠' ,10: '의료', 11:'교통, 운송', 12:'전자, 정보통신', 13: '기타'}
#=============================================================================================================================================


def write_data(data, name):
    with open(name + '.bin', 'wb') as f:
        pickle.dump(data, f)
        
def load_data(name):
    with open(name + '.bin', 'rb') as f:
        data = pickle.load(f)
    return data        

def isnull(df): 
    df_5 = df 
    
    df_5.isnull().sum()
    info = df_5.info()
    
    return info


#==============================================================PART 0 (MACRO ANALYSIS)============================================================


def macro_analysis(macro_path):
    df_path = 'jeju'
    os.chdir(os.path.join(os.getcwd(), df_path))
    dfs = dict()
    for i in [5, 6, 7, 8]:
        dfs[i] = pd.read_csv('KRI-DAC_Jeju_data' + str(i) + '.txt')
    os.chdir('../')
    df_all = pd.concat(dfs.values(), axis=0)
    
    # 전체 기간 모든 TYPE 별 값들
    print("Macro 1. 업종별 분석")
    types, types_cnt = np.unique(df_all['Type'], return_counts = True)
    cols = ['TotalSpent', 'DisSpent', 'NumofSpent', 'NumofDisSpent']
    type_df = pd.DataFrame(columns = ['Type'] + cols + ['ratio (%)', 'perDisSpent'])
    
    for idx, t in enumerate(types):
        cur_type = pd.Series({'Type' : t})
        cur_type = cur_type.append(df_all.loc[df_all['Type'] == t][cols].sum())
        cur_type['ratio (%)'] = round(cur_type['DisSpent']/cur_type['TotalSpent']*100, 2)
        if cur_type['NumofDisSpent'] != 0:
            cur_type['perDisSpent'] = round(cur_type['DisSpent']/cur_type['NumofDisSpent'])
        else:
            cur_type['perDisSpent'] = int(0)
        type_df = type_df.append(cur_type, ignore_index=True)
    for c in cols:
        type_df = type_df.astype({c : 'int64'})
    os.chdir('extra_data')
    code_table = pd.read_csv('code_table.csv', encoding = 'utf-8-sig', sep = ',')
    os.chdir('../')
    type_df['code'] = code_table['code']
    code_df = type_df.groupby(by = 'code').sum()
    code_df['ratio (%)'] = round(code_df['DisSpent']/code_df['TotalSpent']*100, 2)
    code_df['class'] = code_dict.values()
                
    # 월별 매출액 and 재난지원금 사용률
    print("Macro 2. 월별 지출 분석")
    monthly_spent = pd.DataFrame(columns = ['month'] + cols + ['ratio (%)', 'perDisSpent'])
    for m in [5, 6, 7, 8]:
        cur_df = dfs[m]
        cur_df = cur_df[cols].sum()
        monthly_spent = monthly_spent.append({'month': m, 'TotalSpent' : cur_df['TotalSpent'], 'DisSpent' : cur_df['DisSpent'],\
                                              'NumofSpent' : cur_df['NumofSpent'], 'NumofDisSpent' : cur_df['NumofDisSpent'],'ratio (%)': \
                                              round(cur_df['DisSpent']/cur_df['TotalSpent']*100, 2), 'perDisSpent': round(cur_df['DisSpent']/cur_df['NumofDisSpent'])}, ignore_index = True)
    
    # 시각화
    os.chdir(macro_path)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=monthly_spent['month'], y=monthly_spent['ratio (%)'], name='월별 재난지원금 사용 비율'), secondary_y=False)
    fig.update_traces(marker_color= 'deepskyblue', secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly_spent['month'], y=monthly_spent['perDisSpent'],name='월별 평균 재난지원금 사용 액'), secondary_y=True)
    fig.update_traces(marker_color= 'deeppink', secondary_y=True)
    fig.update_layout(title = dict(text = '월별 재난지원금 사용금액 비율/평균 사용 액', font = dict(size = 25)), legend = dict())
    fig.update_xaxes(
            nticks = 4,
            title_text = "Month",
            title_font = {"size": 20},
            title_standoff = 25)
    fig.update_yaxes(
            title_text = "비율 (%)",
            title_font = {"size": 20},
            secondary_y = False
            )
    fig.update_yaxes(
            title_text = "사용 금액 (원)",
            title_font = {"size": 20},
            secondary_y = True
            )
    
    fig.write_image('월별 총 사용금액 대비 재난지원금 사용금액.png')
    
    
    
    data = []
    data.append(go.Bar(x=monthly_spent['month'], y=monthly_spent['TotalSpent'] - monthly_spent['DisSpent'], name='사용 금액', marker=dict(
                        color = '#1D3557')))
    data.append(go.Bar(x=monthly_spent['month'], y=monthly_spent['DisSpent'], name='재난지원금 사용 금액', marker=dict(
                        color = '#A8DADC')))
    layout = go.Layout(barmode='stack', title='월별 사용금액')
    fig = go.Figure(data=data, layout=layout)
    
    fig.update_xaxes(
            title_text = "Month",
            title_font = {"size": 20},
            title_standoff = 25)
    fig.update_yaxes(
            title_text = "금액 (원)",
            title_font = {"size": 20},
            )
    fig.write_image('월별 사용 금액.png')
    
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x = code_df['TotalSpent'] - code_df['DisSpent'], y = code_df['class'], name = '총 사용 금액', orientation= 'h', marker=dict(color='#FA9579')))
    fig.add_trace(go.Bar(x = code_df['DisSpent'], y = code_df['class'], name = '재난지원금액', orientation= 'h', marker = dict(color = '#65D6CE')))
    fig.update_layout(title = '업종 대분류 별 사용 금액 (%: 재난지원금의 비율)', barmode = 'stack', paper_bgcolor='rgb(255, 255, 255)',\
                              plot_bgcolor='rgb(255, 255, 255)',margin=dict(l=30, r=40, t=30, b=30), showlegend=True) 
    
    annotations = []
    for yd, xd in zip(code_df['class'], code_df['TotalSpent']):
        annotations.append(dict(xref='x', yref='y', xanchor = 'left', x=xd, y=yd, text= str(round(code_df.loc[code_df['class'] == yd]['DisSpent'].values[0]/code_df.loc[code_df['class'] == yd]['TotalSpent'].values[0] * 100, 1)) + '%',
                                        font=dict(family='Arial', size=14, color='rgb(67, 67, 67)'),
                                        showarrow=False))
    fig.update_layout(annotations=annotations)                 
    fig.write_image('업종 대분류 별 사용 금액.png')


    print("Macro 3. 시간별 지출 분석")
    hours = df_all['Time'].unique()
    hours_dict = dict()
    han_dict = dict()
    foreign_dict = dict()
    pyun_dict = dict()
    super_dict = dict()
    princ_types = ['일반한식', '편의점', '슈퍼마켓', '서양음식']
    for i in tqdm(range(len(hours))): 
        hours_dict[hours[i]] = 0
        han_dict[hours[i]] = 0
        pyun_dict[hours[i]] = 0
        super_dict[hours[i]] = 0
        foreign_dict[hours[i]] = 0
    
    for h in hours:
        cur_h = df_all.loc[df_all['Time'] == h]
        hours_dict[h] = cur_h['DisSpent'].sum()
        han_dict[h] = cur_h.loc[cur_h['Type'] == '일반한식']['DisSpent'].sum()
        pyun_dict[h] = cur_h.loc[cur_h['Type'] == '편의점']['DisSpent'].sum()
        super_dict[h] = cur_h.loc[cur_h['Type'] == '슈퍼마켓']['DisSpent'].sum()
        foreign_dict[h] = cur_h.loc[cur_h['Type'] == '서양음식']['DisSpent'].sum()
    
    
    # X시 지우기
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
    fig6.write_image('서양음식_한시간별_재난지원금_사용금액.png')


def change_crs(df, pkl_path, month): 
    
    #df_5 = df
    df_5 = df.sample(frac=0.01, random_state=77).reset_index()
    
    if month == '7':
        df_5 = df_5.drop(['X', 'Y'], axis = 1)
    proj_UTMK = Proj(init='epsg:5178') #기존 좌표계: ITRF2000
    proj_WGS84 = Proj(init='epsg:4326') #변환할 좌표계
    column_names = df_5.columns.tolist()
    
    
    df_55d = dict() 
    for i in range(len(df_5)):
        ITRF2000_x = df_5.loc[i, ['POINT_X']].astype(int)[0]
        ITRF2000_y = df_5.loc[i, ['POINT_Y']].astype(int)[0]
        WGS84_x, WGS84_y = transform(proj_UTMK, proj_WGS84, ITRF2000_x, ITRF2000_y)
        
        df_55d[i] = {'OBJECTID': df_5.loc[i, ['OBJECTID']][0],
                          'Field1': df_5.loc[i, ['Field1']][0],
                          'YM': df_5.loc[i, ['YM']][0],
                          'SIDO': df_5.loc[i, ['SIDO']][0],
                          'SIGUNGU': df_5.loc[i, ['SIGUNGU']][0],
                          'FranClass': df_5.loc[i, ['FranClass']][0],
                          'Type': df_5.loc[i, ['Type']][0],
                          'Time': df_5.loc[i, ['Time']][0],
                          'TotalSpent': df_5.loc[i, ['TotalSpent']][0],
                          'DisSpent': df_5.loc[i, ['DisSpent']][0],
                          'NumofSpent': df_5.loc[i, ['NumofSpent']][0],
                          'NumofDisSpent': df_5.loc[i, ['NumofDisSpent']][0],
                          'POINT_X': WGS84_x,
                          'POINT_Y': WGS84_y}
       
        if i%100 == 0: 
            total = len(df_5)/100
            print('{} out of {}...'.format(int(i/100), int(total)))
        
  
    df_55 = pd.DataFrame(df_55d).T
    # os.chdir(pkl_path)
    write_data(df_55, 'df_{}월'.format(month))
    # df_55.to_pickle('df_{}월.pkl'.format(month))    

    
    
def time_split(new_df): 
    df_55 = new_df

    # 시간대별 분할 
    df_55_si = df_55['Time'].tolist()
    df_55_ssi = [int(i.replace('시', '')) for i in df_55_si if i != 'x시'] #x시 drop
    df_55.drop(df_55.loc[df_55['Time']=='x시'].index, inplace=True) # OO시로 되어 있던 string 변수를 OO int 변수로 대체 
    df_55['Time'] = df_55_ssi
    
    
    df_55_AM1 = df_55[df_55['Time'] < 6] #00시 - 06시
    df_55_AM2 = df_55[(df_55['Time'] >= 6) & (df_55['Time'] < 11)] #06시 - 12시
    df_55_PM1 = df_55[(df_55['Time'] >= 12) & (df_55['Time'] < 18)] #12시 - 18시
    df_55_PM2 = df_55[(df_55['Time'] >= 18) & (df_55['Time'] < 24)] #18시 - 24시

    return df_55_AM1, df_55_AM2, df_55_PM1, df_55_PM2




'''
2. 소상공인구분(FranClass)별 업종(Type) 분포
'''

def franClass_split(time_df, time): 
    
    df_55 = time_df
    df_55 = df_55.reset_index()
    fc = df_55['FranClass'].value_counts()
    column_names = df_55.columns.tolist()
    
    # 영세 - a, 일반 - b, 중소 - c, 중소1 - d, 중소2 - e >> make empty dfs
    df_5a = pd.DataFrame(data = [], columns=column_names)
    df_5b = pd.DataFrame(data = [], columns=column_names)
    df_5c = pd.DataFrame(data = [], columns=column_names)
    df_5d = pd.DataFrame(data = [], columns=column_names)
    df_5e = pd.DataFrame(data = [], columns=column_names)
    
    # 소상공인구분 별로 df 분리 
    for i in range(len(df_55)):     
        
        if i%100 == 0: #tracker
            total = len(df_55)/100
            print('{} out of {}...'.format(int(i/100), int(total)))
        
        if df_55.loc[i, ['FranClass']][0] == '영세': 
            ex_df = pd.DataFrame({'OBJECTID': df_55.loc[i, ['OBJECTID']][0],
                          'Field1': df_55.loc[i, ['Field1']][0],
                          'YM': df_55.loc[i, ['YM']][0],
                          'SIDO': df_55.loc[i, ['SIDO']][0],
                          'SIGUNGU': df_55.loc[i, ['SIGUNGU']][0],
                          'FranClass': df_55.loc[i, ['FranClass']][0],
                          'Type': df_55.loc[i, ['Type']][0],
                          'Time': df_55.loc[i, ['Time']][0],
                          'TotalSpent': df_55.loc[i, ['TotalSpent']][0],
                          'DisSpent': df_55.loc[i, ['DisSpent']][0],
                          'NumofSpent': df_55.loc[i, ['NumofSpent']][0],
                          'NumofDisSpent': df_55.loc[i, ['NumofDisSpent']][0],
                          'POINT_X': df_55.loc[i, ['POINT_X']][0],
                          'POINT_Y': df_55.loc[i, ['POINT_Y']][0]}, index = [0])
            df_5a = df_5a.append(ex_df, ignore_index=True)   
            
            
        if df_55.loc[i, ['FranClass']][0] == '일반':
            ex_df = pd.DataFrame({'OBJECTID': df_55.loc[i, ['OBJECTID']][0],
                          'Field1': df_55.loc[i, ['Field1']][0],
                          'YM': df_55.loc[i, ['YM']][0],
                          'SIDO': df_55.loc[i, ['SIDO']][0],
                          'SIGUNGU': df_55.loc[i, ['SIGUNGU']][0],
                          'FranClass': df_55.loc[i, ['FranClass']][0],
                          'Type': df_55.loc[i, ['Type']][0],
                          'Time': df_55.loc[i, ['Time']][0],
                          'TotalSpent': df_55.loc[i, ['TotalSpent']][0],
                          'DisSpent': df_55.loc[i, ['DisSpent']][0],
                          'NumofSpent': df_55.loc[i, ['NumofSpent']][0],
                          'NumofDisSpent': df_55.loc[i, ['NumofDisSpent']][0],
                          'POINT_X': df_55.loc[i, ['POINT_X']][0],
                          'POINT_Y': df_55.loc[i, ['POINT_Y']][0]}, index = [0])
            df_5b = df_5b.append(ex_df, ignore_index=True)   
            
        if df_55.loc[i, ['FranClass']][0] == '중소':
            ex_df = pd.DataFrame({'OBJECTID': df_55.loc[i, ['OBJECTID']][0],
                          'Field1': df_55.loc[i, ['Field1']][0],
                          'YM': df_55.loc[i, ['YM']][0],
                          'SIDO': df_55.loc[i, ['SIDO']][0],
                          'SIGUNGU': df_55.loc[i, ['SIGUNGU']][0],
                          'FranClass': df_55.loc[i, ['FranClass']][0],
                          'Type': df_55.loc[i, ['Type']][0],
                          'Time': df_55.loc[i, ['Time']][0],
                          'TotalSpent': df_55.loc[i, ['TotalSpent']][0],
                          'DisSpent': df_55.loc[i, ['DisSpent']][0],
                          'NumofSpent': df_55.loc[i, ['NumofSpent']][0],
                          'NumofDisSpent': df_55.loc[i, ['NumofDisSpent']][0],
                          'POINT_X': df_55.loc[i, ['POINT_X']][0],
                          'POINT_Y': df_55.loc[i, ['POINT_Y']][0]}, index = [0])
            df_5c = df_5c.append(ex_df, ignore_index=True)    
            
        if df_55.loc[i, ['FranClass']][0] == '중소1':
            ex_df = pd.DataFrame({'OBJECTID': df_55.loc[i, ['OBJECTID']][0],
                          'Field1': df_55.loc[i, ['Field1']][0],
                          'YM': df_55.loc[i, ['YM']][0],
                          'SIDO': df_55.loc[i, ['SIDO']][0],
                          'SIGUNGU': df_55.loc[i, ['SIGUNGU']][0],
                          'FranClass': df_55.loc[i, ['FranClass']][0],
                          'Type': df_55.loc[i, ['Type']][0],
                          'Time': df_55.loc[i, ['Time']][0],
                          'TotalSpent': df_55.loc[i, ['TotalSpent']][0],
                          'DisSpent': df_55.loc[i, ['DisSpent']][0],
                          'NumofSpent': df_55.loc[i, ['NumofSpent']][0],
                          'NumofDisSpent': df_55.loc[i, ['NumofDisSpent']][0],
                          'POINT_X': df_55.loc[i, ['POINT_X']][0],
                          'POINT_Y': df_55.loc[i, ['POINT_Y']][0]}, index = [0])
            df_5d = df_5d.append(ex_df, ignore_index=True)          
            
        if df_55.loc[i, ['FranClass']][0] == '중소2':
            ex_df = pd.DataFrame({'OBJECTID': df_55.loc[i, ['OBJECTID']][0],
                          'Field1': df_55.loc[i, ['Field1']][0],
                          'YM': df_55.loc[i, ['YM']][0],
                          'SIDO': df_55.loc[i, ['SIDO']][0],
                          'SIGUNGU': df_55.loc[i, ['SIGUNGU']][0],
                          'FranClass': df_55.loc[i, ['FranClass']][0],
                          'Type': df_55.loc[i, ['Type']][0],
                          'Time': df_55.loc[i, ['Time']][0],
                          'TotalSpent': df_55.loc[i, ['TotalSpent']][0],
                          'DisSpent': df_55.loc[i, ['DisSpent']][0],
                          'NumofSpent': df_55.loc[i, ['NumofSpent']][0],
                          'NumofDisSpent': df_55.loc[i, ['NumofDisSpent']][0],
                          'POINT_X': df_55.loc[i, ['POINT_X']][0],
                          'POINT_Y': df_55.loc[i, ['POINT_Y']][0]}, index = [0])
            df_5e = df_5e.append(ex_df, ignore_index=True)   
    
    # # save dfs to pickles
    write_data(df_5a, 'df_{}_a'.format(time))
    write_data(df_5b, 'df_{}_b'.format(time))
    write_data(df_5c, 'df_{}_c'.format(time))
    write_data(df_5d, 'df_{}_d'.format(time))
    write_data(df_5e, 'df_{}_e'.format(time))
    # df_5a.to_pickle('df_{}_a.pkl'.format(time))    
    # df_5b.to_pickle('df_{}_b.pkl'.format(time))    
    # df_5c.to_pickle('df_{}_c.pkl'.format(time))    
    # df_5d.to_pickle('df_{}_d.pkl'.format(time))    
    # df_5e.to_pickle('df_{}_e.pkl'.format(time))    
    
    return df_5a, df_5b, df_5c, df_5d, df_5e
 
    
def franClass_type_analysis_plotly(df_dict, month, map_path, time_s):
    types_u = np.unique(df_dict[time_s]['a']['Type'].to_numpy())
    def pie_chart(df_orig, target_prop, value_prop, cut = 0.01, top = 10, title = None):
        '''
        df_orig: 입력 데이터 프레임
        target_prop: 표현하고자 하는 대상 (범례로 들어감)
        value_prop: 표현되는 값
        cut: 얼만 큼의 부분 이하는 자르기 (%)
        top: 상위 n개 이외의 자료는 기타로 부여
        titla: 차트 제목
        '''
        
        df = pd.DataFrame(columns = ['names', 'values'])
        if target_prop == value_prop:
            df['names'] = df_orig[target_prop].value_counts().keys()
            df['values'] = df_orig[target_prop].value_counts().values
        else:
            df['names'] = df_orig[target_prop]
            df['values'] = df_orig[value_prop]
        
        df.sort_values(by=['values'])
        
        if cut:
            df.loc[df['values']/df['values'].cumsum() < cut, 'names'] = '기타'
        if top:
            critria = df.iloc[top - 1]['values']
            df.loc[df['values'] < critria, 'names'] = '기타'
        
        
        fig = px.pie(df, values='values', names='names', title = title)
        # fig2 = go.Figure(data = [go.Pie(labels = df.names, values = df.values)])
        return df, fig
    
    def stacked_bar(df_orig, x, cols, sum_col = None, name_used = False, title = False):
        '''
        df_orig: 입력 데이터 프레임
        x: x 축에 들어갈 카테고리 정보(업종 등)
        cols: 누적될 value 열 이름 (리스트)
        sul_col: cols에 이미 누적된 데이터가 있을 경우 (예: TotalSpent)
        name_used: x의 원소 중 사용되어질 원소가 미리 정해져 있는 경우 (리스트)
        title: 차트 제목
        '''
        
        if sum_col:
            y = sum_col
            columns = [x]
            for c in cols:
                if c == sum_col:
                    columns.append(sum_col + ' - others')
                else:
                    columns.append(c)
            columns.append(y)
        else:
            y = 'sum'
            columns = [x]
            for c in cols:
                columns.append(c)
            columns.append(y)
        
        
        df = pd.DataFrame(columns = columns)
        df[x] = df_orig[x]
        for c in cols:
            df[c] = df_orig[c]
        
        df = df.groupby(by = 'Type').sum()
        
        if name_used:
            for name in df.index:
                if name not in name_used:
                    df = df.drop(name)
        if sum_col:
            df[sum_col + ' - others'] = 2*df[sum_col] - df.sum(axis = 1)
        else:
            df[y] = df.sum(axis = 1)
        df = df.reset_index()
        df = df.sort_values(by=[y], ascending=False)
        for col in df.columns:
            if df[col].dtype not in ['object', 'str']:
                df[col] = df[col]/1000
        if sum_col:
            y_s = list(df.columns)
            y_s.remove(x)
            y_s.remove(y)
            fig = px.bar(df, x=x, y=y_s, title = title)
            
            data = []
            for i in y_s:
                data.append(go.Bar(x = df[x], y = df[i], text = df[i], name = i))
            layout = go.Layout(barmode='stack', plot_bgcolor='rgba(0,0,0,0)', title=title)
            total_labels = [{"x": x, "y": total + 100, "text": str(total), "showarrow": False} for x, total in zip(df[x], df[sum_col])]
            fig = go.Figure(data=data, layout=layout)
            fig = fig.update_layout(annotations=total_labels)
        else:
            fig = px.bar(df, x=x, y=cols, title = title)
        
        return fig
        
    def horizontal_stacked_bar(df_orig, y_s, cut = 0.03, sum_col = None, name_used = False, title = False):
        '''
        '''
        colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet',\
                  'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue',\
                  'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid',\
                  'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink',\
                  'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', \
                  'gold', 'goldenrod', 'gray', 'grey', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', \
                  'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral']
        cols = ['Type']
        for y in y_s:
            cols.append(y)
        
        df = pd.DataFrame(columns = cols)
        
        for c in cols:
            df[c] = df_orig[c]
        df = df.groupby(by = 'Type').sum()    
        if name_used:
            for name in df.index:
                if name not in name_used:
                    df = df.drop(name)
        for y in y_s:
            df[y] = df[y] / df[y].sum()
        
        x_data = []
        y_data = ['총 사용 금액', '재난 지원금', '총 사용 건수', '재난 지원금<br>사용 건수']
        for c in df.columns:
            x_data.append(df[c])
        np.random.seed(42)
        rand = np.random.uniform(-100, 100, 100)
        fig = go.Figure()
        for i in range(0, len(x_data[0])):
            cur_color = colors[-1 * i]
            fig.add_trace(go.Bar(
                y = y_data,
                x = df.iloc[i].tolist(),
                name = df.iloc[i].name,
                orientation='h',
                marker=dict(
                    color = code_table.loc[code_table['type'] == df.iloc[i].name]['color'].values[0],
                    line=dict(color='rgb(45, 45, 45)', width=1)
                    )
                ))
        fig.update_layout(title = title, barmode = 'stack', paper_bgcolor='rgb(255, 255, 255)',\
                          plot_bgcolor='rgb(255, 255, 255)',margin=dict(l=30, r=40, t=30, b=30), showlegend=True)
        
        
        annotations = []
        for yd, xd in zip(y_data, x_data):
            if xd[0] > cut:
                annotations.append(dict(xref='x', yref='y',
                                x=xd[0] / 2, y=yd,
                                text=str(int(round(xd[0],2) * 100)) + '%',
                                font=dict(family='Arial', size=14,
                                          color='rgb(67, 67, 67)'),
                                showarrow=False))
            space = xd[0]
            for i in range(1, len(xd)):
                    # labeling the rest of percentages for each bar (x_axis)
                    if xd[i] > cut:
                        annotations.append(dict(xref='x', yref='y',
                                                x=space + (xd[i]/2), y=yd,
                                                text=str(int(round(xd[i],2) * 100)) + '%',
                                                font=dict(family='Arial', size=14,
                                                          color='rgb(67, 67, 67)'),
                                                showarrow=False))
                    space += xd[i]
        fig.update_layout(annotations=annotations)
        return fig
        
        
        
    # a_label, a_freq = pie_chart(df_dict[time_s]['a'], type_dict['a'], month)
    
    
    name_used_dict = dict() # 각 업종 타입별 분석대상으로 선정된 업종들을 저장 and 반환
    fig_list = []
    
    
    
    def gis_analysis(df, label, map_path, fc_class, month, time_s): 

        filepath = map_path
        center = [33.3989, 126.60]
        
        
        df_5a = df
        
        
        df_total_spent_top = df_5a.sort_values(by='TotalSpent', ascending=False).groupby('FranClass').head(100)
        labels = df_total_spent_top['Type'].unique().tolist()
        
        df_disspent_top = df_5a.sort_values(by='DisSpent', ascending=False).groupby('FranClass').head(100)
        llabeld = df_total_spent_top['Type'].unique().tolist()
        
        df_num_total_spent_top = df_5a.sort_values(by='NumofSpent', ascending=False).groupby('FranClass').head(100)
        llabels = df_total_spent_top['Type'].unique().tolist()
        
        df_num_disspent_top = df_5a.sort_values(by='NumofDisSpent', ascending=False).groupby('FranClass').head(100)
        llabelds = df_total_spent_top['Type'].unique().tolist()
        
        def df_spent_sum(df_5z, labels):   
            df_5a_finance = pd.DataFrame(data=[], columns=['Type', 'TotalSpent', 'DisSpent', 'NumofSpent', 'NumofDisSpent'])   
            def spent_sum(df_5z, typee):
                total_spent_ls = []
                dis_spent_ls = []
                num_spent_ls = []
                num_disspent_ls = []
                
                for i in range(len(df_5z)):      
                    tt = df_5z.iloc[i]['Type']
                   
                    if tt in labels:          
                        if tt == typee: 
                        
                            total_spent = df_5z.iloc[i]['TotalSpent']
                            total_spent_ls.append(total_spent)
                            
                            dis_spent = df_5z.iloc[i]['DisSpent']
                            dis_spent_ls.append(dis_spent)
                            
                            num_spent = df_5z.iloc[i]['NumofSpent']
                            num_spent_ls.append(num_spent)
                            
                            num_disspent = df_5z.iloc[i]['NumofDisSpent']
                            num_disspent_ls.append(num_disspent)
                        
                sum_total_spent = sum(total_spent_ls)
                sum_disspent = sum(dis_spent_ls)
                sum_num_spent = sum(num_spent_ls)
                sum_num_disspent = sum(num_disspent_ls)
                
                return sum_total_spent, sum_disspent, sum_num_spent, sum_num_disspent
            
            for i in range(len(labels)):
                sum_total_spent, sum_disspent, sum_num_spent, sum_num_disspent = spent_sum(df_5z, labels[i])
                df_5a_finance.loc[i] = [labels[i], sum_total_spent, sum_disspent, sum_num_spent, sum_num_disspent]
        
            return df_5a_finance
            
        
        
        total_spent_top = df_spent_sum(df_total_spent_top, labels)
        total_spent_top = total_spent_top[['Type', 'TotalSpent']]
        disspent_top = df_spent_sum(df_disspent_top, llabeld)
        disspent_top = disspent_top[['Type', 'DisSpent']]
        num_total_spent_top = df_spent_sum(df_num_total_spent_top, llabels)
        num_total_spent_top = num_total_spent_top[['Type', 'NumofSpent']]
        num_disspent_top = df_spent_sum(df_num_disspent_top, llabelds)
        num_disspent_top = num_disspent_top[['Type', 'NumofDisSpent']]
        df_all = pd.concat([total_spent_top.reset_index(drop=True), disspent_top, num_total_spent_top, num_disspent_top], axis=1)
        types = total_spent_top['Type'].to_list()
        rows = list(set(df_all.columns.to_list()))
        rows.remove('Type')
        new_df = pd.DataFrame(index = [], columns = types)
        
        df_all['TotalSpent'].T
        
        cur_s = df_all['TotalSpent']
        cur_s.index = types
        new_df = new_df.append(cur_s)
        
        cur_s = df_all['DisSpent']
        cur_s.index = types
        new_df = new_df.append(cur_s)
        
        cur_s = df_all['NumofSpent']
        cur_s.index = types
        new_df = new_df.append(cur_s)
        
        cur_s = df_all['NumofDisSpent']
        cur_s.index = types
        new_df = new_df.append(cur_s)
        
        nnn = new_df.T
        
        nnn['TotalSpent_percentage'] = 100 * nnn.TotalSpent/nnn.TotalSpent.sum()
        nnn['DisSpent_percentage'] = 100 * nnn.DisSpent/nnn.DisSpent.sum()
        nnn['NumofSpent_percentage'] = 100 * nnn.NumofSpent/nnn.NumofSpent.sum()
        nnn['NumofDisSpent_percentage'] = 100 * nnn.NumofDisSpent/nnn.NumofDisSpent.sum()
        

        
        
        
        df_total_spent_top = df_5a.sort_values(by='TotalSpent', ascending=False).groupby('FranClass').head(100)
        labels = df_total_spent_top['Type'].unique().tolist()
        
        df_disspent_top = df_5a.sort_values(by='DisSpent', ascending=False).groupby('FranClass').head(100)
        llabeld = df_disspent_top['Type'].unique().tolist()
        
        df_num_total_spent_top = df_5a.sort_values(by='NumofSpent', ascending=False).groupby('FranClass').head(100)
        llabels = df_num_total_spent_top['Type'].unique().tolist()
        
        df_num_disspent_top = df_5a.sort_values(by='NumofDisSpent', ascending=False).groupby('FranClass').head(100)
        llabelds = df_num_disspent_top['Type'].unique().tolist()
        
        
        center = [33.3989, 126.60]
        m2 = folium.Map(location=center, tiles='openstreetmap', zoom_start=11)
        heat_df = df_5a[['POINT_Y', 'POINT_X']]
        heat_data = [[row['POINT_Y'],row['POINT_X']] for index, row in heat_df.iterrows()]
        HeatMap(heat_data).add_to(m2)
        
        
        
        for idx, row in df_total_spent_top.iterrows():
            g = Marker([row['POINT_Y'], row['POINT_X']], popup=row['Type'], icon=folium.Icon(color='green')).add_to(m2)
        
        for idx, row in df_disspent_top.iterrows():
            p = Marker([row['POINT_Y'], row['POINT_X']], popup=row['Type'], icon=folium.Icon(color='purple')).add_to(m2)
        
        for idx, row in df_num_total_spent_top.iterrows():
            b = Marker([row['POINT_Y'], row['POINT_X']], popup=row['Type'], icon=folium.Icon(color='blue')).add_to(m2)
        
        for idx, row in df_num_disspent_top.iterrows():
            k = Marker([row['POINT_Y'], row['POINT_X']], popup=row['Type'], icon=folium.Icon(color='red')).add_to(m2)
           
        m2.save(filepath + '{}_{}월_{}시_히트맵_상위100개.html'.format(fc_class, month, time_s))
          
    
    
    
    for t in type_dict.keys():
        # 파이 차트 생성
        
        df, fig = pie_chart(df_dict[time_s][t], 'Type', 'Type', cut = 0.03, top = 10, \
                        title = '{}월 {} 제주도 {} 업종 별 파이'.format(month, time_dict[time_s], type_dict[t]))
        fig.write_image('{}월 {} 제주도 {} 업종 별 파이.png'.format(month, time_dict[time_s], type_dict[t]))
        name_used = df.names.unique().tolist()
        if '기타' in name_used:
            name_used.remove('기타')
        name_used_dict[t] = name_used
        
        # stacked bar 차트 생성
        sum_col = 'TotalSpent'
        fig = stacked_bar(df_dict[time_s][t], 'Type', ['TotalSpent', 'DisSpent'], sum_col = sum_col, name_used = name_used,\
                          title = '{} 업종 별 총 사용금액 대비 재난지원금 사용금액: {}월 {} [단위: 1000원]'.format(type_dict[t], month, time_dict[time_s]))
        fig.write_image('{} 업종 별 총 사용금액 대비 재난지원금 사용금액_{}월 {}.png'.format(type_dict[t], month, time_dict[time_s]))
        

        fig= horizontal_stacked_bar(df_dict[time_s][t], ['TotalSpent', 'DisSpent', 'NumofSpent', 'NumofDisSpent'], cut = 0.03, name_used = name_used,\
                          title = '{} 업종 별 재난지원금 사용건수 및 금액: {}월 {}'.format(type_dict[t], month, time_dict[time_s]))
        fig.write_image('{} 업종 별 재난지원금 사용건수 및 금액_ {}월 {}.png'.format(type_dict[t], month, time_dict[time_s]))
        
        
        # gis analysis (히트맵 상위 100개)
        gis_analysis(df_dict[time_s][t], name_used, map_path, type_dict[t], month, time_s = time_s)
    
    


def emdli_gis_analysis(df_d, month, map_path, extra_data_path, preprocessed): 
    '''
    1. 읍면동리 .shp 파일 출처: http://www.gisdeveloper.co.kr/?p=2332
        - 읍면동 2020년 5월 파일, 리 2020 5월 파일 다운로드 
    2. 법정동명 및 코드 데이터 출처: https://www.code.go.kr/stdcode/regCodeL.do
        - 지역선택: 제주특별자치도 선택 
        - [조회] 버튼 클릭
        - [사용자 검색자료] 버튼 클릭을 클릭하여 제주특별자치도의 법정동코드와 법정동명 데이터 다운로드
    '''
    os.chdir(extra_data_path)
    
    # 읍면동 리 파일 불러오기  
    # emd = gpd.read_file('emd.shp', encoding = 'euc-kr')
    # li = gpd.read_file('li.shp', encoding = 'euc-kr')
    
    # # 제주도만 뽑아내고 합치기 (읍과 면은 제외)
    # emdli = gpd.GeoDataFrame(columns = ['CODE', 'ENG_NM', 'KOR_NM'], geometry = [])
    # for i, row in emd.iterrows():
    #     if row['EMD_CD'][:2] == '50' and row['EMD_KOR_NM'][-1] != '읍' and row['EMD_KOR_NM'][-1] != '면' : 
    #         emdli = emdli.append({'CODE' : row['EMD_CD'], 'ENG_NM' : row['EMD_ENG_NM'], 'KOR_NM' : row['EMD_KOR_NM'], 'geometry' : row['geometry']}, ignore_index = True)
    # for i, row in li.iterrows():
    #     if row['LI_CD'][:2] == '50': 
    #         emdli = emdli.append({'CODE' : row['LI_CD'], 'ENG_NM' : row['LI_ENG_NM'], 'KOR_NM' : row['LI_KOR_NM'], 'geometry' : row['geometry']}, ignore_index = True)
    
    # # 좌표계 변환 (UTM-K -> GRS80)
    # emdli.crs = 'EPSG:5178'
    # jeju = emdli.to_crs("EPSG:4326")
    # jeju.to_file('jeju.shp', encoding = 'utf-8')
    
    
    # 지도 불러오기
    jeju = gpd.read_file('jeju.shp', encoding = 'utf-8')
    df_new_crs = df_d
    
    
    # 각 거래기록의 좌표를 기준으로 주소와 행정구역 코드 부여
    if preprocessed == False:
        coords = []
        for i in tqdm(range(len(df_d))):
            row = df_d.iloc[i]
            if (row['POINT_X'], row['POINT_Y']) not in coords:
                coords.append((row['POINT_X'], row['POINT_Y']))
        ADDs = dict()
        for k in tqdm(range(len(coords))):
            for i in range(len(jeju)):
                cur_point = Point(coords[k])
                if jeju.iloc[i]['geometry'].contains(cur_point):
                    ADDs[k] = (coords[k], jeju.iloc[i]['CODE'], jeju.iloc[i]['KOR_NM'])
                    break
            if i == len(jeju) - 1:
                ADDs[k] = (coords[k], 'NaN', 'NaN')
        ADD_dict = dict()
        CODE_dict = dict()
        for i in tqdm(range(len(ADDs))):
            ids = df_d.loc[(df_d['POINT_X'] == ADDs[i][0][0]) & (df_d['POINT_Y'] == ADDs[i][0][1])]['OBJECTID']
            for j in ids:
                CODE_dict[j] = ADDs[i][1]
                ADD_dict[j] = ADDs[i][2]
        adds = [i[1] for i in sorted(ADD_dict.items())]
        codes = [i[1] for i in sorted(CODE_dict.items())]
        df_d['CODE'] = codes        
        df_d['ADDRESS'] = adds
        os.chdir(map_path)
        os.chdir('../')
        write_data(df_d, 'df_{}월_add'.format(month))
        df_new_crs = df_d
    
    else: # 전처리된 데이터프레임 불러오기
        os.chdir(map_path)
        os.chdir('../')
        df_new_crs = load_data('df_{}월_add'.format(month)) 
    
    
    # 딕셔너리를 이용해서 'TotalSpent', 'DisSpent', 'NumofSpent', 'NumofDisSpent' 각 값을 더해주기
    # 비교 대상은 행정코드를 기준으로
    sum_dict = dict()
    for k in jeju['CODE'].to_list():
        sum_dict[k] = [0, 0, 0, 0]
    
    
    for idx, row in df_new_crs.iterrows():
        num_of_missing = 0
        try:    
            cur_code = row['CODE']
            sum_dict[cur_code][0] += row.TotalSpent
            sum_dict[cur_code][1] += row.DisSpent
            sum_dict[cur_code][2] += row.NumofSpent
            sum_dict[cur_code][3] += row.NumofDisSpent
        except: 
            num_of_missing += 1
            continue
            
    for i in range(len(jeju)): 
        
        cur_code = jeju.loc[i, ['CODE']][0]
        jeju.loc[i, ['TotalSpent']] = sum_dict[cur_code][0]
        jeju.loc[i, ['DisSpent']] = sum_dict[cur_code][1]
        jeju.loc[i, ['NumofSpent']] = sum_dict[cur_code][2]
        jeju.loc[i, ['NumofDisSpent']] = sum_dict[cur_code][3]
        
        
    emdli_json = jeju.to_json()
    emdli_json = json.loads(emdli_json)
    jeju2 = jeju.copy()
    # folium에서 데이터 프레임의 id는 string이어야 함.
    jeju2['id'] = np.array([i for i in range(len(jeju2))]).astype('str')        
        
    
    os.chdir(map_path)
    m1 = folium.Map(location=[33.3948, 126.237], zoom_start=11)
    folium.Choropleth(
        geo_data = emdli_json,
        data = jeju2,
        columns = ['id', 'TotalSpent'],
        fill_color = 'YlGnBu', #puRd, YlGnBu
        key_on = 'feature.id',
        bins = 8,
        legend_name='Total Spent'
        ).add_to(m1)
    
    folium.LayerControl().add_to(m1)
    m1.save('{}_TotalSpent.html'.format(month))
    
    
    
    m2 = folium.Map(location=[33.3948, 126.237], tiles='openstreetmap', zoom_start=11)
    folium.Choropleth(
        geo_data = emdli_json,
        data = jeju2,
        columns = ['id', 'DisSpent'],
        fill_color = 'YlGnBu', #puRd, YlGnBu
        key_on = 'feature.id',
        bins = 8,
        legend_name='Dis Spent'
        ).add_to(m2)
    m2.save('{}_DisSpent.html'.format(month))
   
    
    
    m3 = folium.Map(location=[33.3948, 126.237], tiles='openstreetmap', zoom_start=11)
    folium.Choropleth(
        geo_data = emdli_json,
        data = jeju2,
        columns = ['id', 'NumofSpent'],
        fill_color = 'YlGnBu', #puRd, YlGnBu
        key_on = 'feature.id',
        bins = 8,
        legend_name='Num of Spent'
        ).add_to(m3)
    m3.save('{}_NumofSpent.html'.format(month))
    
    

    m4 = folium.Map(location=[33.3948, 126.237], tiles='openstreetmap', zoom_start=11)
    folium.Choropleth(
        geo_data = emdli_json,
        data = jeju2,
        columns = ['id', 'NumofDisSpent'],
        fill_color = 'YlGnBu', #puRd, YlGnBu
        key_on = 'feature.id',
        bins = 8,
        legend_name='Num of Dis Spent'
        ).add_to(m4)
    
    m4.save('{}_NumofDisSpent.html'.format(month))
    
    
    return sum_dict
    
    
    
#============================================================================================================
#============================================================================================================

def part_0():
    macro_path = os.path.join(os.getcwd(), 'macro_stat')
    macro_analysis(macro_path)
    

def part_1(jj, preprocessed = True):                       
    
    mainpath = os.getcwd()
    data_path = os.path.join(mainpath, 'jeju')
    extra_data_path = os.path.join(mainpath, 'extra_data')
    
    pkl_path = os.path.join(mainpath, jj)
    if not os.path.exists(pkl_path):
        os.mkdir(pkl_path)
        
    map_path = os.path.join(pkl_path, 'map/')
    if not os.path.exists(map_path):
        os.makedirs(map_path)
    
    mpl.rcParams['axes.unicode_minus'] = False


    time_seq = ['AM1', 'AM2', 'PM1', 'PM2']
    type_seq = ['a', 'b', 'c', 'd', 'e']
    
    for time in time_seq: 
        if not os.path.exists(pkl_path + '/' + time + '/'):
            os.mkdir(pkl_path + '/' + time + '/')
        

    os.chdir(data_path)
    df = pd.read_csv(jj)
    month = jj[-5]
    os.chdir(mainpath)
    
    
    # 1. 결측값 여부 조사 
    info = isnull(df)
    print(" ")
    print('>> {}월 데이터 결측값 여부'.format(month))
    print(info)
    print(" ")
    
    
    if preprocessed == False:
        print("Micro 0. 데이터 전처리")
        # 2. 좌표계 변환 및 피클로 저장 및 불러오기
        os.chdir(jj)
        df_d = change_crs(df, pkl_path, month) # (불러오기 시 주석 처리)
    
    
    
        # 3. 시간대별 분할 (불러오기 시 주석 처리)
        df_AM1, df_AM2, df_PM1, df_PM2 = time_split(df_d)
        
        
        # 4. 소상공인별 분할 및 파이차트 (불러오기 시 주석 처리)
        franClass_split(df_AM1, 'AM1')
        franClass_split(df_AM2, 'AM2')
        franClass_split(df_PM1, 'PM1')
        franClass_split(df_PM2, 'PM2')
        
    else:
        os.chdir(jj)
        df_d = load_data('df_{}월'.format(month))
    
    
    
    
    
    
    print("Micro 1. 시간대 별 업종 분석")
    
    df_dict = dict()
    for time in time_seq:
        df_dict[time] = dict()
        
    for time in time_seq:
        for t in type_seq:
            # df_dict[time][t] = pd.read_pickle('df_{}_{}.pkl'.format(time, t))
            df_dict[time][t] = load_data('df_{}_{}'.format(time, t))
    for time in time_seq: 
        print('{}({}) 시간대 분석'.format(time, time_dict[time]))
        os.chdir(os.path.join(pkl_path, time))
        df_used_dict = franClass_type_analysis_plotly(df_dict, month, map_path, time_s = time) #time_s='00시 - 06')
    
    print("Micro 2. 시간대/지역 별 공간 분석")
    sum_dict = emdli_gis_analysis(df_d, month, map_path, extra_data_path, preprocessed)  
    
    return sum_dict, month, mainpath, data_path, extra_data_path, map_path
    
    
def part_2 (all_sum_dict, month, mainpath, data_path, extra_data_path, map_path): 
    
    
    
    
    def bjd_preprocessing(bjd_code): 
        new_code_ls = []
        for i in range(len(bjd_code)): 
            if bjd_code.loc[i, ['법정동명']][0].endswith('동') or bjd_code.loc[i, ['법정동명']][0].endswith('읍') or bjd_code.loc[i, ['법정동명']][0].endswith('면'):
                new_code = str(bjd_code.loc[i, ['법정동코드']][0])[:-2] #동으로 끝나면 법정동코드 끝 두자리 삭제 
                new_code_ls.append(new_code)      
            else: 
                new_code_ls.append(str(bjd_code.loc[i, ['법정동코드']][0]))
                
                
        bjd_code['법정동코드'] = new_code_ls
        
        new_bjd_code = bjd_code
    
        return new_bjd_code
    
    # 6. 법정동코드 및 법정동명 엑셀 파일 불러오기 및 전처리
    bjd_code = pd.read_excel(extra_data_path + '/BJD_CODE.xls')
    bjd_copy = bjd_code.copy()
    bjd_code_ = bjd_preprocessing(bjd_code)
    
    
    for i in range(len(all_sum_dict)): 
        cur_month = str(i+5)
        cur_dict = all_sum_dict[cur_month]
        for k, v in cur_dict.items():
            sum_code = k
            jj_adrs = bjd_code['법정동명'][bjd_code['법정동코드'] == sum_code].values[0]
            cur_dict[k].append(jj_adrs)
        
    
    
    # 빈으로 저장
    os.chdir('../')
    location_stat_path = os.path.join(os.getcwd(), '지역별_통계')
    
        
        
    all_sum_df = pd.DataFrame.from_dict(all_sum_dict, orient='index').T
    all_sum_df = all_sum_df.reset_index()
    all_sum_df = all_sum_df.rename(columns = {'index' : 'CODE'})
    months = [i for i in all_sum_dict.keys()]
    
    if not os.path.exists(location_stat_path):
        os.mkdir(location_stat_path)
    
    
    os.chdir(location_stat_path)
    
    
    fig1_ls = [] #금액
    fig2_ls = [] #건수
    
    for i in range(len(all_sum_dict['5'])): 
        
        area = all_sum_df.iloc[i]['5'][4] 
        
        total_spent_5 = all_sum_df.iloc[i]['5'][0]/1000
        dis_spent_5 = all_sum_df.iloc[i]['5'][1]/1000
        num_spent_5 = all_sum_df.iloc[i]['5'][2]
        num_disspent_5 = all_sum_df.iloc[i]['5'][3]
        
        total_spent_6 = all_sum_df.iloc[i]['6'][0]/1000
        dis_spent_6 = all_sum_df.iloc[i]['6'][1]/1000
        num_spent_6 = all_sum_df.iloc[i]['6'][2]
        num_disspent_6 = all_sum_df.iloc[i]['6'][3]
        
        total_spent_7 = all_sum_df.iloc[i]['7'][0]/1000
        dis_spent_7 = all_sum_df.iloc[i]['7'][1]/1000
        num_spent_7 = all_sum_df.iloc[i]['7'][2]
        num_disspent_7 = all_sum_df.iloc[i]['7'][3]
        
        total_spent_8 = all_sum_df.iloc[i]['8'][0]/1000
        dis_spent_8 = all_sum_df.iloc[i]['8'][1]/1000
        num_spent_8 = all_sum_df.iloc[i]['8'][2]
        num_disspent_8 = all_sum_df.iloc[i]['8'][3]
        
        fig = go.Figure()
        fig2 = go.Figure()
        
        fig.add_trace(go.Bar(x=months,
                    y=[total_spent_5, total_spent_6, total_spent_7, total_spent_8], 
                    name='총 사용금액 (1000원)',
                    marker_color='rgb(204, 153, 255)',
                    text = [total_spent_5, total_spent_6, total_spent_7, total_spent_8], 
                    textposition="outside"
                    ))
        
        fig.add_trace(go.Bar(x=months,
                    y=[dis_spent_5, dis_spent_6, dis_spent_7, dis_spent_8], 
                    name='재난지원금 사용금액 (1000원)',
                    marker_color='rgb(000, 255, 102)',
                    text = [dis_spent_5, dis_spent_6, dis_spent_7, dis_spent_8], 
                    textposition="outside"
                    ))
        
        fig2.add_trace(go.Bar(x=months,
                    y=[num_spent_5, num_spent_6, num_spent_8, num_spent_8], 
                    name='사용건수 (회)',
                    marker_color='rgb(102, 051, 255)',
                    text = [num_spent_5, num_spent_6, num_spent_8, num_spent_8], 
                    textposition="outside"
                    ))
        
        fig2.add_trace(go.Bar(x=months,
                    y=[num_disspent_5, num_disspent_6, num_disspent_7, num_disspent_8], 
                    name='재난지원금 사용건수 (회)',
                    marker_color='rgb(000, 153, 153)',
                    text = [num_disspent_5, num_disspent_6, num_disspent_7, num_disspent_8], 
                    textposition="outside"
                    ))
        
        fig.update_layout(
        title='{}의 월별 결제금액 대비 재난지원금 결제금액 비교'.format(area),
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='금액 (1000원)',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
        )
        
        
        fig2.update_layout(
        title='{}의 월별 결제건수 대비 재난지원금 이용건수 비교'.format(area),
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='건수 (회)',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
        )
        
        
        os.chdir(location_stat_path)
        fig.write_image('사용금액_{}.png'.format(area))
        fig2.write_image('사용건수_{}.png'.format(area))                 
    
                            
    return None
    
    
#============================================================================================================


path = os.getcwd()
print('Current Path: {}'.format(path))
os.chdir(os.path.join(path, 'jeju'))
jeju_list = os.listdir()


extra_data_path = os.path.join(path, 'extra_data')


if __name__=='__main__':   
    
    
    print("======== PART {} ANALYSIS START ========= ".format(0))
    os.chdir('../')
    
    part_0()
    print("======== PART {} ANALYSIS DONE  ========= ".format(0))
    
    
    print("======== PART {} ANALYSIS START ========= ".format(1))
    all_sum_dict = dict()
    for jj in jeju_list: 
        # if jj[-5] is not '6':
        #     continue
        print(" ")
        print("======== MONTH {} ANALYSIS START ========= ".format(jj[-5]))
        
        os.chdir(path)
        start_time = time.time() 
        print(" ")        
        
        sum_dict, month, mainpath, data_path, extra_data_path, map_path = part_1(jj)     
        end_time = time.time()   
        
        print(" ")
        print("Running time : ", end_time - start_time)
        print("======== MONTH {} ANALYSIS DONE ========= ".format(jj[-5]))
        
        all_sum_dict[month] = sum_dict  
    
    os.chdir(extra_data_path)
    write_data(all_sum_dict, 'all_sum_dict')
    all_sum_dict = load_data('all_sum_dict')
    print("======== PART {} ANALYSIS DONE  ========= ".format(1))
    print("======== PART {} ANALYSIS START ========= ".format(2))
    part_2(all_sum_dict, month, mainpath, data_path, extra_data_path, map_path)         
    











