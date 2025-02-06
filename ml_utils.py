import os
import sys
from math import pi, sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from shapely.geometry import Point
import fiona
import geopandas as gpd

def area(shape,height,width,nbarrels):
    ## 1 = ARCH, 2 = CIRC, 3 = OREC, 4 = RECT
    ## inches to feet
    height = height/12
    width = width/12
    if shape == 1:
        cs_area = width*(height-width/2) + pi/2*(width/2)**2
    if shape == 2:
        cs_area = pi*(width/2)**2
    if shape == 3:
        cs_area = width*height
    if shape == 4:
        cs_area = width*height
    cs_area = cs_area * nbarrels
    return cs_area

def pipe_area(width,depth):
    ## inches to feet
    depth = depth/12
    width = width/12
    if depth == 0:
        cs_area = pi*(width/2)**2
    else:
        cs_area = width*depth
    ## square inch to square feet
    cs_area = cs_area/12
    return cs_area

def evaluate(ml_algrthm,pred_y,test_y):
    print ('-'*len(ml_algrthm))
    print (ml_algrthm)
    print ('-'*len(ml_algrthm))

    ## r-squared
    rsq = round(r2_score(test_y,pred_y),2)
    print ("R-squared = {}".format(rsq))
    
    ## mean absulute percent error
    #mape = round(mean_absolute_percentage_error(test_y,pred_y)*100,1)
    mape = round(np.mean(np.abs(test_y-pred_y)/test_y)*100,1)
    print ("Mean Absolute Percent Error = {}%".format(mape))

    ## root mean squared error
    rmse = round(mean_squared_error(test_y,pred_y,squared=False),1)
    print ("Root Mean Squared Error = {}-cfs".format(rmse))

    ## mean absolute error
    mae = round(mean_absolute_error(test_y,pred_y),1)
    print ("Mean Absolute Error = {}-cfs".format(mae))
   
    df_out = pd.DataFrame({'Modeled Flow (cfs)':test_y,'ML Predicted Flow (cfs)':pred_y})
    df_out.plot(kind='line',figsize=(12,6))
    plt.show()

    df_out['Error (cfs)'] = df_out['Modeled Flow (cfs)']-df_out['ML Predicted Flow (cfs)']
    df_out.plot.scatter(x='Modeled Flow (cfs)',y='ML Predicted Flow (cfs)',c='orange',edgecolors='black',xlim=(0,3500),ylim=(0,3500),grid=True)
    plt.show()

def rfr_model(ml_algrthm,train_x,train_y,x_list,test_x,test_y):
    rf = RandomForestRegressor(n_estimators=1000,random_state=0)
    rf.fit(train_x,train_y)
    
    ## plot feature importances
    plt.barh(x_list,rf.feature_importances_)
    plt.show()

    pred_y = rf.predict(test_x)
    
    evaluate(ml_algrthm,pred_y,test_y)
    
    return rf

def knn_model(ml_algrthm,train_x,train_y,x_list,test_x,test_y):
    knn = KNeighborsRegressor(weights='distance')
    knn.fit(train_x,train_y)

    pred_y = knn.predict(test_x)
    
    evaluate(ml_algrthm,pred_y,test_y)
    
    return knn

def gbr_model(ml_algrthm,train_x_sc,train_y,x_list,test_x_sc,test_y):
    gbr = GradientBoostingRegressor(n_estimators=1000,random_state=0)
    gbr.fit(train_x_sc,train_y)

    ## plot feature importances
    plt.barh(x_list,gbr.feature_importances_)
    plt.show()
    
    pred_y = gbr.predict(test_x_sc)
    
    evaluate(ml_algrthm,pred_y,test_y)
     
    return gbr

def svm_model(ml_algrthm,train_x_sc,train_y,x_list,test_x_sc,test_y):
    #svm = SVR(kernel='linear')
    svm = SVR(kernel='rbf',gamma=0.5,C=10.0)
    svm.fit(train_x_sc,train_y)

    pred_y = svm.predict(test_x_sc)
    
    evaluate(ml_algrthm,pred_y,test_y)
    
    return svm

def mlp_model(ml_algrthm,train_x_sc,train_y,x_list,test_x_sc,test_y):
    mlp = MLPRegressor(max_iter=20000,learning_rate='adaptive',activation='tanh',random_state=0,alpha=0.05)
    mlp.fit(train_x_sc,train_y)

    pred_y = mlp.predict(test_x_sc)
    
    evaluate(ml_algrthm,pred_y,test_y)
    
    return mlp

def read_sw_data(gdb):
    lyrs = fiona.listlayers(gdb)
    for lyr in lyrs:
        if lyr == 'SW_LINES':
            print ("Reading {}".format(lyr))
            gdf = gpd.read_file(gdb,driver='FileGDB',layer=lyr)

    print ("Screening SW_LINES for culverts and mains")
    gdf = gdf[gdf['GEN_TYPE'].isin(['Culvert','Main','Lateral'])]
    
    print ("Calculating SW_LINES % longitudinal slope")
    gdf['Gradient (%)'] = (gdf['FNI_DS_ELEV'] - gdf['FNI_US_ELEV'])/gdf['CONDLEN'] * 100.0
    
    print ("Calculating cross-sectional area")
    gdf['Area (sft)'] = gdf.apply(lambda x: pipe_area(x['LINEWIDTH'],x['LINEDEPTH']),axis=1)
    
    gdf['Length (ft)'] = gdf['CONDLEN']
    gdf['DS invert (ft)'] = gdf['FNI_DS_ELEV']
    gdf['US invert (ft)'] = gdf['FNI_US_ELEV']
    
    df = pd.DataFrame(gdf,copy=True)
    
    return df

def main():
    ## user inputs
    icm_data_file = r"C:\Users\ah4318\Documents\ah4318_Projects\35773.001-000004_CSDSA-LOS\ICM_TS_output\ML_input.txt"
    x_list = ['Gradient (%)','Area (sq-ft)','Length (ft)','DS ground level (ft)','US ground level (ft)','DS invert (ft)','US invert (ft)']
    x_list = ['Gradient (%)','Area (sq-ft)','Length (ft)','DS invert (ft)','US invert (ft)']
    #y_col = 'Max velocity (fps)'
    y_col = 'Max flow (cfs)'
    min_len = 100
    min_wid = 18
    txt_cod = r"C:\Users\ah4318\Documents\ah4318_Projects\35773_CSDSA\scripts\sw_data.txt"
    
    min_area = pi*(min_wid/12)**2/4
    
    ## read ICM output
    df = pd.read_csv(icm_data_file,header=0,sep='\t')
    ## remove rows with no data
    df.dropna(axis=0,how='any',inplace=True)
    
    ## calculate Zscore
    df['Zscore'] = abs((df[y_col] - df[y_col].mean())/df[y_col].std())
    ## screen based on Zscore
    #df = df[df['Zscore'] < 3.0]
    
    ## calculate maximum of design and simulated discharge
    if y_col == 'Max flow (cfs)':
        df[y_col] = df[['Full capacity (cfs)','Max flow (cfs)']].max(axis=1)
    
    ## calculate cross-sectional area of pipe
    df['Area (sq-ft)'] = df.apply(lambda x: area(x['Shape'],x['Height (in)'],x['Width (in)'],x['Number of barrels']),axis=1)
    #stats = df.describe()
    #df['Gradient (%)'] = abs(df['Gradient (%)'])

    ## screen based on conduit properties
    #df = df[df['Width (in)'] >= min_wid]
    #df = df[df['Length (ft)'] >= min_len]
    df = df[df['Shape'].isin([1,2,4])]
    #df = df[df['Area (sq-ft)'] > min_area]
    
    y = np.array(df[y_col])
    print (y.shape)
    x = df[x_list]
    x = np.array(x)
    
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.20,random_state=0)
    
    sc = StandardScaler()
    scaler = sc.fit(train_x)
    train_x_sc = scaler.transform(train_x)
    test_x_sc = scaler.transform(test_x)
    
    ## k-Nearest Neighbors
    #knn = knn_model("k-Nearest Neighbors",train_x,train_y,x_list,test_x,test_y)
    
    ## Support Vector Machine
    #svm = svm_model("Support Vector Machine",train_x_sc,train_y,x_list,test_x_sc,test_y)

    ## Random Forest
    rfr = rfr_model("Random Forest",train_x,train_y,x_list,test_x,test_y)
    
    ## Gradient Boost
    #gbr = gbr_model("Gradient Boost",train_x_sc,train_y,x_list,test_x_sc,test_y)

    ## Artificial Neural Network
    #mlp = mlp_model("Artificial Neural Network",train_x_sc,train_y,x_list,test_x_sc,test_y)
    
    ## Read stormwater database
    '''
    df_cod = read_sw_data(gdb_cod)
    df_cod = df_cod[['GBAID'] + x_list]
    df_cod.dropna(axis=0,how='any',inplace=True)
    '''
    '''
    df_cod = pd.read_csv(txt_cod,header=0,sep='\t')
    x_cod = x = np.array(df_cod[x_list])
    df_cod['Gradient (%)'] = abs(df_cod['Gradient (%)'])
    q_cod = rfr.predict(x_cod)
    df_cod['Max Discharge (cfs)'] = q_cod
    df_cod.to_csv(os.path.join(os.getcwd(),'predicted_discharges.txt'),index=False,header=True,sep='\t')
    '''
if __name__ == '__main__': main()