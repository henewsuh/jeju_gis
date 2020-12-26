
import pandas as pd
import numpy as np
import os
import pickle

# 분류 출처 : 소상공인진흥공단 소상공인 업종 대분류
code_dict = {0: '음식', 1: '소매', 2 : '생활 서비스', 3: '학문, 교육', 4: '숙박', 5: '관광, 여가 ,오락', 6:'문화, 예술, 종교', 7:'도매, 유통, 무역',\
             8: '부동산', 9: '스포츠' ,10: '의료', 11:'교통, 운송', 12:'전자, 정보통신', 13: '기타'}

def write_data(data, name):
    with open(name + '.bin', 'wb') as f:
        pickle.dump(data, f)
        
def load_data(name):
    with open(name + '.bin', 'rb') as f:
        data = pickle.load(f)
    return data        
dfs = dict()
for i in range(5, 9):
    path = 'C:/Users/user/Desktop/jeju_gis/KRI-DAC_Jeju_data' + str(i) + '.txt'
    os.chdir(path)
    dfs[i] = load_data('df_' + str(i) + '월')
# df6 = load_data('df_6월')
# df7 = load_data('df_7월')
# df8 = load_data('df_8월')
df_all = pd.concat(dfs.values())


# 전체 기간 모든 TYPE 별 값들
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
    type_df.loc[idx] = cur_type
    
os.chdir('../')
code_table = pd.read_csv('code_table.csv', encoding = 'euc-kr', sep = ',')
type_df['code'] = code_table['code']
code_df = type_df.groupby(by = 'code').sum()
code_df['ratio (%)'] = round(code_df['DisSpent']/code_df['TotalSpent']*100, 2)
code_df['class'] = code_dict.values()
            
# 월별 매출액 and 재난지원금 사용률
monthly_spent = pd.DataFrame(columns = ['month'] + cols + ['ratio (%)', 'perDisSpent'])
for m in [5, 6, 7, 8]:
    cur_df = dfs[m]
    cur_df = cur_df[cols].sum()
    monthly_spent = monthly_spent.append({'month': m, 'TotalSpent' : cur_df['TotalSpent'], 'DisSpent' : cur_df['DisSpent'],\
                                          'NumofSpent' : cur_df['NumofSpent'], 'NumofDisSpent' : cur_df['NumofDisSpent'],'ratio (%)': \
                                          round(cur_df['DisSpent']/cur_df['TotalSpent']*100, 2),\
                                              'perDisSpent': round(cur_df['DisSpent']/cur_df['NumofDisSpent'])}, ignore_index = True)

