import pandas as pd

for y in ['2025', '2030', '2035', '2040', '2045', '2050']:
    for wy in [#'WY2015', 'WY2016', 'WY2017', 'WY2018', 'WY2019', 
        'WY2020',
         # 'WY2021', 'WY2022'
         ]:
        pump = pd.read_csv('00. Input/' + y + '/'+wy+'/09. Pumped hydro generators.csv', index_col=0, encoding='latin1')
        pump.to_csv('00. Input/' + y + '/'+wy+'/09. Pumped hydro generators.csv')
        heat = pd.read_csv('00. Input/' + y + '/'+wy+'/14. DH plants.csv', index_col=0, encoding='latin1')
        heat.to_csv('00. Input/' + y + '/'+wy+'/14. DH plants.csv',)
        PtX = pd.read_csv('00. Input/' + y + '/'+wy+'/13. PtX plants.csv', index_col=0, encoding='latin1')
        PtX.to_csv('00. Input/' + y + '/'+wy+'/13. PtX plants.csv',)