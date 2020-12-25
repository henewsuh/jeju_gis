# 공간정보 탐색적 데이터 분석 경진대회
> ## Project Title  
>
> 제주도 재난지원금 사용건수 및 사용금액 비교 분석 (소상공인구분별/업종별/월별/시간대별)
>
> ## Motivation 
>
> 2회에 걸쳐 지급된 재난지원금의 사용금액과 이용건수를 다각도에서 세분화하여 분석하고자 함.
>
> - 5월, 6월, 7월, 8월별 
> - 시간대별 (00시 - 06시 / 06시 - 12시 / 12시 -18시 / 18시 - 24시)
> - 소상공인구분별 + 업종별 
> - 지역별 (읍면동리 기반)
>
> ## Dependencies
>
> Recent versions of the following packages for Python 3 are required: 
>
> - bokeh==2.2.3
> - folium==0.11.0
> - geopandas==0.8.1
> - geopy==2.0.0
> - kaleido==0.1.0.post1
> - matplotlib==3.3.3
> - networkx==2.5
> - numpy==1.19.4
> - orca==1.5.4
> - osmnx==0.16.2
> - pandas==1.1.5
> - plotly==4.14.1
> - pyproj==3.0.0.post1
> - scikit-learn==0.23.2
> - seaborn==0.11.1
> - selenium==3.141.0
> - Shapely @ file:///C:/Users/user/Desktop/Shapely-1.7.1-cp38-cp38-win_amd64.whl
> - sklearn==0.0
>
> ## Datasets
>
> The raw datasets are available at: 
>
> - DACON: https://dacon.io/competitions/official/235682/data/
> - 읍면동리 SHAPE 파일: http://www.gisdeveloper.co.kr/?p=2332
>   - 읍면동 2020년 5월 파일, 리 2020 5월 파일 사용
> - 법정동명 및 법정동코드: https://www.code.go.kr/stdcode/regCodeL.do
>   - 지역선택: 제주특별자치도 선택
>   - [조회] 버튼 클릭
>   - [사용자 검색자료] 버튼 클릭을 클릭하여 제주특별자치도의 법정동코드와 법정동명 데이터 다운로드
>
> ## Usage
>
> 1. Clone the  current repository 
> 2. Make sure that all raw datasets (total 4) are in the folder 'jeju' under 'jeju_gis'.
> 3. Execute the following command from the project home directory: 
>    - python jeju_gis_analysis_v22_for_git.py
>
> ## Updates
>
> #### 2020.12.25 updates:
>
> - 건수 기반 히트맵 및 사용금액/재난지원금 사용금액/이용건수/재난지원금 이용건수 상위 100개의 업종에 대한 지도 시각화 추가 
>   - 확인 경로 예시: KRI-DAC_Jeju_data5.txt > map > 영세 5월 AM1시 히트맵 상위 100개.html
>     - 설명: 5월 영세업자의 00시 - 06시 (AM1) 사용금액/재난지원금 사용금액/이용건수/재난지원금 이용건수 상위 100개 표시 (pinpoint 마커를 클릭하면 업종의 종류 가시화)
> - 5월, 6월, 7월, 8월 전체 데이터에 대한 결과 이미지 폴더에 추가 
>   - 확인 경로 예시: KRI-DAC_Jeju_data5.txt > AM1 > ___.png 
> - 법정동코드 엑셀, 읍면동리 SHAPE 파일 업로드
> - 월별 읍면동리 지역기반 사용금액/재난지원금 사용금액/이용건수/재난지원금 지도 시각화 추가
>   - 확인 경로 예시: KRI-DAC_Jeju_data5.txt > map > 5_DisSpent.html
>     - 설명: 5월 한달동안의 재난지원금 사용금액을 읍면동리 지역기반으로 지도에 시각화



