# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:15:42 2020

@author: user
"""
import pandas as pd
import os 
import folium
from folium import plugins
from pyproj import Proj, transform
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.io import curdoc
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle 
import matplotlib.font_manager as fm
import numpy as np
from folium import Choropleth, Circle, Marker
from folium import plugins
from folium.plugins import HeatMap
import time
import plotly.express as px
import plotly.graph_objects as go
import random
from tqdm.notebook import tqdm
type_dict = {'a' : '영세', 'b' : '일반', 'c' : '중소' , 'd' : '중소1', 'e' : '중소2'}
time_dict = {'AM1' : '0시 ~ 6시', 'AM2' : '6시 ~ 12시', 'PM1' : '12시 ~ 18시' , 'PM2' : '18시 ~ 24시'}

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



def change_crs(df, pkl_path, month): 
    
    df_5 = df#.sample(frac=0.001, random_state=77).reset_index()
    
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
    os.chdir(pkl_path)
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
 
    
def franClass_type_analysis_plotly(df_dict, month, time_s):
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
                    color = cur_color,
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
    
    



# ========================================================================================================================
#                                                    지도 시각화
# ========================================================================================================================



def gis_analysis(df, label, pkl_path, map_path, fc_class, month, time_s): 

    filepath = map_path
    center = [33.3989, 126.60]
    
    # m1 = folium.Map(location=center, tiles='openstreetmap', zoom_start=11)
    # df_5a = df
    # a_label = label
    
    # def color_producer(val, a_label):
    #     color_list = ['#00FFFF', '#006400','#00FF7F', '#FFFF00', '#B22222', '#FA8072', '#FF0000', '#FF1493', '#FF00FF',
    #                   '#A020F0', '#0000FF', '#00BFFF', '#00F5FF', '#7FFFD4', '#B24040', '#AE905E', '#FFA0FF', '#54BD54', '#FF8200',
    #                   'green', 'Orange', 'darkblue', 'red', 'lightblue', 'brown', 'grey', 'yellow', 'purple', 'cadetblue', 'pink', 
    #                   'lightgreen', 'blue', 'beige', 'white', 'white', 'white', 'white','#00FFFF', '#006400','#00FF7F', '#FFFF00', '#B22222', '#FA8072', '#FF0000', '#FF1493', '#FF00FF',
    #                   '#A020F0', '#0000FF', '#00BFFF', '#00F5FF', '#7FFFD4', '#B24040', '#AE905E', '#FFA0FF', '#54BD54', '#FF8200',
    #                   'green', 'Orange', 'darkblue', 'red', 'lightblue', 'brown', 'grey', 'yellow', 'purple', 'cadetblue', 'pink', 
    #                   'lightgreen', 'blue', 'beige', 'white', 'white', 'white', 'white','#00FFFF', '#006400','#00FF7F', '#FFFF00', '#B22222', '#FA8072', '#FF0000', '#FF1493', '#FF00FF',
    #                   '#A020F0', '#0000FF', '#00BFFF', '#00F5FF', '#7FFFD4', '#B24040', '#AE905E', '#FFA0FF', '#54BD54', '#FF8200',
    #                   'green', 'Orange', 'darkblue', 'red', 'lightblue', 'brown', 'grey', 'yellow', 'purple', 'cadetblue', 'pink', 
    #                   'lightgreen', 'blue', 'beige', 'white', 'white', 'white', 'white']
      
    #     for i in range(len(a_label)): 
    #         if val == a_label[i]: 
    #             return color_list[i]
    
    
    # for i in range(len(df_5a)): 
    #     Circle(location = [df_5a.iloc[i]['POINT_Y'], df_5a.iloc[i]['POINT_X']],
    #            radius = 20,
    #            color = color_producer(df_5a.iloc[i]['Type'], a_label)).add_to(m1)
               
    # m1.save(filepath + '{}_{}월_{}시_자료_위치_분포.html'.format(fc_class, month, time_s))
    
    
    
    
    
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
    
    ddd = nnn.T.copy()
    ddd = ddd.loc['TotalSpent_percentage':'NumofDisSpent_percentage']
    
    dffig = ddd.plot(kind='barh', xlim=[0,120], 
             title='{} 업종 별 재난지원금 사용건수 및 금액: {}월 {}시'.format(fc_class, month, time_s), 
             legend='reverse', figsize=(10,5), stacked=True)
    fig = dffig.get_figure()
    fig.savefig('{}_업종별_재난지원금_사용건수_및_금액_{}월_{}시.png'.format(fc_class, month, time_s), dpi=300)
    
    
    
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
            
    
    
    
        
#============================================================================================================
#============================================================================================================


def main(jj):                       
    
    mainpath = os.getcwd()
    data_path = os.path.join( mainpath, 'jeju')
    
    pkl_path = os.path.join(mainpath, jj)
    if not os.path.exists(pkl_path):
        os.mkdir(pkl_path)
        
    map_path = os.path.join(pkl_path, 'map')
    if not os.path.exists(map_path):
        os.makedirs(map_path)
    
    mpl.rcParams['axes.unicode_minus'] = False


    time_seq = ['AM1', 'AM2', 'PM1', 'PM2']
    
    for time in time_seq: 
        if not os.path.exists(pkl_path + '/' + time + '/'):
            os.mkdir(pkl_path + '/' + time + '/')
        

    os.chdir(data_path)
    df = pd.read_csv(jj)
    month = jj[-5]
    os.chdir(mainpath)
    
    
    # 1. 결측값 여부 조사 
    info = isnull(df)
    print('>> {}월 데이터 결측값 여부'.format(month))
    print(info)
    
    
    # 2. 좌표계 변환 및 피클로 저장 및 불러오기
    df_d = change_crs(df, pkl_path, month) # (불러오기 시 주석 처리)
    # os.chdir(jj)
    df_d = load_data('df_{}월'.format(month))
    
    
    # 3. 시간대별 분할 (불러오기 시 주석 처리)
    df_AM1, df_AM2, df_PM1, df_PM2 = time_split(df_d)
    
    
    # 4. 소상공인별 분할 및 파이차트 (불러오기 시 주석 처리)
    franClass_split(df_AM1, 'AM1')
    franClass_split(df_AM2, 'AM2')
    franClass_split(df_PM1, 'PM1')
    franClass_split(df_PM2, 'PM2')
    
    df_dict = dict()
    
    for time in time_seq:
        df_dict[time] = dict()
    type_seq = ['a', 'b', 'c', 'd', 'e']
    for time in time_seq:
        for t in type_seq:
            # df_dict[time][t] = pd.read_pickle('df_{}_{}.pkl'.format(time, t))
            df_dict[time][t] = load_data('df_{}_{}'.format(time, t))
    for time in time_seq: 
        os.chdir(os.path.join(pkl_path, time))
        df_used_dict = franClass_type_analysis_plotly(df_dict, month, time_s = time) #time_s='00시 - 06')
    
    
    
    
    
    

#============================================================================================================


path = os.getcwd()

print('Current Path: {}'.format(path))

os.chdir(os.path.join(path, 'jeju'))

jeju_list = os.listdir()



if __name__=='__main__':   
    for jj in jeju_list: 
        # if jj[-5] is not '5':
        #     continue
        print(" ")
        print("======== {} ANALYSIS START ========= ".format(jj))
        os.chdir(path)
        start_time = time.time()
        main(jj)
        end_time = time.time()
        
        print("Running time : ", end_time - start_time)










