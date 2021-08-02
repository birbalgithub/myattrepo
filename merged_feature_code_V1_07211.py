# Databricks notebook source
# MAGIC %md
# MAGIC # Table of contents
# MAGIC 1. [Session 1: Feature creation code to get current cohort dataframe by Daan](#session1)
# MAGIC 2. [Session 2: Generate hist treatment df and create features for hist treatment](#session2)<br>
# MAGIC     2.1 [Run part of first session code with 6 hist month cohorts to get hist treatment df](#subsession21)<br>
# MAGIC     2.2 [Create historical treatment features by Wen](#subsession22)<br>
# MAGIC 3. [Session 3: Hist PA feature creation by Rishabh](#session3)
# MAGIC 4. [Session 4: Numerical feature creation for grid actions by Wen](#session4)

# COMMAND ----------

# MAGIC %md
# MAGIC # Session 1: Feature creation code to get current cohort dataframe by Daan <a name="session1"></a>
# MAGIC - Enabler Cohorts from August 2019/2020 - Standarized to US Pacific Timezone

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading files 
# MAGIC #### Importing libraries and setting options

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import re
from functools import reduce
from datetime import datetime, timedelta, date, timezone, time
import pytz
import os
import warnings
warnings.filterwarnings('ignore')
#import xgboost as xgb
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.metrics import accuracy_score,r2_score, mean_absolute_error, mean_squared_error, auc, mean_absolute_error
#from xgboost import plot_importance

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from pandas.tseries.offsets import MonthBegin,MonthEnd,BMonthBegin
import matplotlib.pyplot as plt

from time import sleep
from tqdm import tqdm

import holidays

us_holidays = holidays.UnitedStates()

#for date, name in sorted(holidays.UnitedStates(years=[2019, 2020, 2021]).items()):
 #   print(date, name)

sns.set_style(style="darkgrid")

pd.set_option('display.max_colwidth',100000)
pd.set_option("display.max_rows", 100000)
pd.set_option('display.max_columns', 100000)

# COMMAND ----------

import textdistance
from fuzzywuzzy import fuzz

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reading input files *(none special order)*

# COMMAND ----------

# [4:06 PM] Flores Rodriguez, Zureyma
#     # For NBA model feature creation, first try to use full files data01-09 for Aug2019 cohorts
# df_DATA01_Aug2020 = spark.read.csv('/mnt/deptwarehouse/TRIAL_DATA01_082020.csv', sep='|', header=True)
# [4:06 PM] Flores Rodriguez, Zureyma
#     import pandas as pd
# df_DATA01_Aug2020_pd = df_DATA01_Aug2020.toPandas()


# COMMAND ----------

#df_str = pd.read_csv('/dbfs/dbfs/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File9_Strata_10k.csv', index_col=0) 
df_DATA01_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File9_Strata_10k.csv', header=True)
df_DATA02_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File1_MainCohort_10k.csv', header=True)
df_DATA03_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File5_CriticalActvt_10k.csv', header=True)
df_DATA04_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File3_Calls_10k.csv', header=True)
df_DATA05_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File4_Messages_10k.csv', header=True)
df_DATA06_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File2_PayAdj_10k.csv', header=True)
df_DATA07_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File8_Promises_10k.csv', header=True)
df_DATA08_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File6_Bills_10k.csv', header=True)
df_DATA09_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File7_Surveys_10k.csv', header=True)

df_DATA10_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_Aug19_PAR_Recommender_10k.csv', header=True)
df_DATA11_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_Aug20_PAR_Recommender_10k.csv', header=True)

# df_coh = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File1_MainCohort_10k.csv', index_col=0) #(not used)
# df_act = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File5_CriticalActvt_10k.csv', index_col=0)
# df_calls = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File3_Calls_10k.csv', index_col=0)
# df_treatment_timestamp = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File4_Messages_10k.csv', index_col=0)
# df_pmt_timestamp = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File2_PayAdj_10k.csv', index_col=0)
# df_pa = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File8_Promises_10k.csv', index_col=0)
# df_billing = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File6_Bills_10k.csv', index_col=0)
# df_survey = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File7_Surveys_10k.csv', index_col=0)

# # PAR Files to get recommended suspension date if available
# df_PAR_2019 = pd.read_csv('Enabler/ENBLR_iASCE Trial_Aug19_PAR_Recommender_10k.csv', index_col=0)
# df_PAR_2020 = pd.read_csv('Enabler/ENBLR_iASCE Trial_Aug20_PAR_Recommender_10k.csv', index_col=0)


df_str = df_DATA01_Aug2020.toPandas()
df_coh = df_DATA02_Aug2020.toPandas()
df_act = df_DATA03_Aug2020.toPandas()
df_calls = df_DATA04_Aug2020.toPandas()
df_treatment_timestamp = df_DATA05_Aug2020.toPandas()
df_pmt_timestamp = df_DATA06_Aug2020.toPandas()
df_pa = df_DATA07_Aug2020.toPandas()
df_billing = df_DATA08_Aug2020.toPandas()
df_survey = df_DATA09_Aug2020.toPandas()

df_PAR_2019 = df_DATA10_Aug2020.toPandas()
df_PAR_2020 = df_DATA11_Aug2020.toPandas()



# COMMAND ----------

df_str

# COMMAND ----------

# %%time

# # 01-09 Data Frames WITH timestamp
# df_str = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File9_Strata_10k.csv', index_col=0) 
# df_coh = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File1_MainCohort_10k.csv', index_col=0) #(not used)
# df_act = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File5_CriticalActvt_10k.csv', index_col=0)
# df_calls = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File3_Calls_10k.csv', index_col=0)
# df_treatment_timestamp = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File4_Messages_10k.csv', index_col=0)
# df_pmt_timestamp = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File2_PayAdj_10k.csv', index_col=0)
# df_pa = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File8_Promises_10k.csv', index_col=0)
# df_billing = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File6_Bills_10k.csv', index_col=0)
# df_survey = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File7_Surveys_10k.csv', index_col=0)

# # PAR Files to get recommended suspension date if available
# df_PAR_2019 = pd.read_csv('Enabler/ENBLR_iASCE Trial_Aug19_PAR_Recommender_10k.csv', index_col=0)
# df_PAR_2020 = pd.read_csv('Enabler/ENBLR_iASCE Trial_Aug20_PAR_Recommender_10k.csv', index_col=0)

# #File with complete Strara Info, not used this time for Enabler
# #df_Full_Strata_2019 = pd.read_csv('.csv', index_col=0)
# #df_Full_Strata_2020 = pd.read_csv('.csv', index_col=0)

# COMMAND ----------

#fmt = '%Y-%m-%d %H:%M:%S %Z%z'

pacific_tz = pytz.timezone("US/Pacific")
eastern_tz = pytz.timezone('US/Eastern')
central_tz = pytz.timezone('US/Central')
mountain_tz = pytz.timezone('US/Mountain')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Testing Data Dimensions

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC print('df_strata: '+str(df_str.shape))
# MAGIC print('df_cohort: '+str(df_coh.shape))
# MAGIC print('df_critical_action: '+str(df_act.shape))
# MAGIC print('df_calls: '+str(df_calls.shape))
# MAGIC print('df_messages_timestamp: '+str(df_treatment_timestamp.shape))
# MAGIC print('df_payments_timestamp: '+str(df_pmt_timestamp.shape))
# MAGIC print('df_payment_arrengements: '+str(df_pa.shape))
# MAGIC print('df_billing: '+str(df_billing.shape))
# MAGIC print('df_survey: '+str(df_survey.shape))
# MAGIC 
# MAGIC print('df_PAR_2019: '+str(df_PAR_2019.shape))
# MAGIC print('df_PAR_2020: '+str(df_PAR_2020.shape))

# COMMAND ----------

# MAGIC %md
# MAGIC ## <span style='color:Yellow'> **Current Functions** <span/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### <span style='color:Brown'> **Function 1** <span/>

# COMMAND ----------

# Wen's function
# define a function to detect CLTN and Cure/Write-off/OCA pair per BAN, 
# output a list of row index that need to be dropped from dataframe CURES

ls_valid_crit_act_cd = ['CLTN', 'CURE', 'OCAE', 'WOFF']

def detect_rows_to_be_dropped(index_comb, crit_act_comb, ls_valid_crit_act_cd):
    ls_index_comb =[int(i.strip()) for i in index_comb.split(', ')]
    ls_crit_act_comb = crit_act_comb.split(', ')
    
    list_rows_to_be_dropped = []  #list to store the row index that needs to be removed from the dataframe
    list_valid_act_cd = []
    cltn_stack = [] # create a stack to pair up CLTN and its nearest CURE/Write-off/OCA
    for i, act_cd in enumerate(ls_crit_act_comb):
        if act_cd not in ls_valid_crit_act_cd: 
            list_rows_to_be_dropped.append(ls_index_comb[i])
        elif act_cd == 'CLTN' and not cltn_stack:
            list_valid_act_cd.append('CLTN')
            cltn_stack.append(('CLTN', ls_index_comb[i])) #find the first 'CLTN' code as well as its row index
        elif act_cd == 'CLTN' and cltn_stack: #another 'CLTN' comes after
            list_rows_to_be_dropped.append(cltn_stack.pop()[1]) #put the ealier CLTN index into drop list
            cltn_stack.append(('CLTN', ls_index_comb[i]))  # update the stack with the latest CLTN           
        elif act_cd !='CLTN' and cltn_stack:
            cltn_stack.pop()  # clear this stack once a matched Cure/Write-off/OCA is found
            list_valid_act_cd.append(act_cd)
        elif act_cd !='CLTN' and not cltn_stack:  #no 'CLTN' to be paired with
            list_rows_to_be_dropped.append(ls_index_comb[i])
     
    if cltn_stack: # still have 'CLTN' left in the stack, without any Cure/Write-off/OCA to be matched with
        # remove this row since no valid code found to pair with this CLTN
        list_rows_to_be_dropped.append((cltn_stack.pop()[1]))
        list_valid_act_cd.pop()
    
    return (list_rows_to_be_dropped, list_valid_act_cd)
            
            
# test an example:
detect_rows_to_be_dropped('0, 1, 2', 'CLTN, CURE, CURE', ls_valid_crit_act_cd)


# COMMAND ----------

# test more examples:
detect_rows_to_be_dropped('9, 10, 11, 12, 13, 14, 15', 'CURE, CLTN, SUSP, RSTR, CURE, CURE, CURE', ls_valid_crit_act_cd)

# COMMAND ----------

# Per discussion with Daan, update the function to only keep latest CLTN to pair with nearest CURE in the following example:

detect_rows_to_be_dropped('9, 10, 11, 12, 13, 14, 15', 'CLTN, CLTN, CURE, RSTR, CURE, CLTN, CURE', ls_valid_crit_act_cd)

# COMMAND ----------

# MAGIC %md
# MAGIC ### <span style='color:Brown'> **Function 2** <span/>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Function  for critical actions, messages and calls to get datetime with time in proper format from original integer time variable

# COMMAND ----------

#Getting the date with timestamp and timezone
def new_datetime(df, time_variable, date_variable, timezone):
    
    time_variable = str(time_variable)
    date_variable = str(date_variable)
    
    timestamp = str(df[time_variable])
    
    if df['len_time'] == 6:
        timestamp = timestamp
    elif df['len_time'] == 5:
        timestamp = '0'+timestamp
    elif df['len_time'] == 4:
        timestamp = '00'+timestamp
    elif df['len_time'] == 3:
        timestamp = '000'+timestamp
    elif df['len_time'] == 2:
        timestamp = '0000'+timestamp
    else:
        timestamp = '00000'+timestamp
    
    a = int(timestamp[0:2])
    b = int(timestamp[2:4])
    c = int(timestamp[4:6])
    
    tm1_op2 = time(a, b, c)    
    new_date = datetime.combine(df[date_variable], tm1_op2)    
    new_date_tz = timezone.localize(new_date, is_dst=True)
    
    return new_date_tz

# COMMAND ----------

# MAGIC %md
# MAGIC ### <span style='color:Brown'> **Function 3** <span/>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Function for payments to add different timezones and convert to pacific timezone

# COMMAND ----------

#Based on timezone add to the datetime variable
def add_timezone(df, timezone_variable, date_variable):
    
    tm_fake = time(0, 0, 0)
    
    timezone_variable = str(timezone_variable)
    date_variable = str(date_variable)
    date_variable_tz = str(date_variable+'_tz')
    date_variable_tz_pacific = str(date_variable_tz+'_pacific')
    
    df[date_variable_tz] = df.apply(lambda x: pd.to_datetime(x[date_variable], format="%m/%d/%Y %H:%M:%S")
                                    if (x['fncl_trans_type_cd'] == 'PYM')
                                    else datetime.combine(x.fncl_trans_dt, tm_fake), axis=1)


    def assess_tz(row, row2, value):
        if row == 'EST':
            value = eastern_tz.localize(value, is_dst=True)
            return value
        elif row == 'PST':
            value = pacific_tz.localize(value, is_dst=True)  
            return value
        elif ((row == 'CST') | (row2 == 'ADJ')):
            value = central_tz.localize(value, is_dst=True) 
            return value
        elif row == 'MST':
            value = mountain_tz.localize(value, is_dst=True)
            return value
        else:
            value = pd.NaT
            return value

    df[date_variable_tz] = df.apply(lambda x: assess_tz(x[timezone_variable], x['fncl_trans_type_cd'], x[date_variable_tz]), axis=1)
    df['pmt_timezone'] = df.apply(lambda x: x[date_variable_tz].tzinfo, axis=1)    
    df[date_variable_tz_pacific] = df.apply(lambda x: x[date_variable_tz].astimezone(pacific_tz), axis=1)
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ### <span style='color:Brown'> **Function 4** <span/>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Function to add laboral days to calculate projected suspension date

# COMMAND ----------

def adding_business_days(from_date, add_days):
    business_days_to_add = add_days
    current_date = from_date
    while business_days_to_add > 0:
        current_date += timedelta(days=1)
        weekday = current_date.weekday()
        if weekday >= 5: # sunday = 6
            continue
        if current_date in us_holidays:
            continue
        business_days_to_add -= 1
    return current_date

#demo:
print('10 business days from today:')
print(date.today())
print(adding_business_days(date.today(), 5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### <span style='color:Brown'> **Function 5** <span/>

# COMMAND ----------

#Adding days to suspension and projected suspension date to Strata  OR TREATMENTS BEFOR SUSP
def timeline_days(risk, stage):
    if stage == 1:
        return df_timeline_susp.loc[risk]['Frie_SMS_Email_Days']
    elif stage == 2:
        return df_timeline_susp.loc[risk]['Ent_Dlnq_SMS_Days']
    elif stage == 3:
        return df_timeline_susp.loc[risk]['PreSusp_SMS_Email_Days']
    elif stage == 4:
        return df_timeline_susp.loc[risk]['PreSusp_Lett_Days']
    else:
        return df_timeline_susp.loc[risk]['Susp_Days']

# COMMAND ----------

# MAGIC %md
# MAGIC ### <span style='color:Brown'> **Function 6** <span/>

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>  Below cell of code has three functions that will be used later by the module
# MAGIC </h3>    <ol>  
# MAGIC        <li> create_date_range  : takes 2 inputs "start_dt and end_dt"
# MAGIC              The function will create a date range wrt the start and end dates </li>
# MAGIC        <li> create_new_df      : takes input as a basic dataframe with  
# MAGIC            <ol>
# MAGIC            <li> BAN numbers </li>
# MAGIC            <li> SUSP_DT </li>
# MAGIC            <li> DLNQ_DT </li>
# MAGIC            <li> MONTHYR </li>
# MAGIC            </ol>
# MAGIC         and creates a new dataframe for all the sample BAN's based on months the customer transacted on.
# MAGIC         </li>
# MAGIC         <li> fit_prev_data :: The function takes the older dataframe from input step and does a merge with 
# MAGIC              the newer data with multiple date rows (date range from DLNQ till SUSP)
# MAGIC         </li>
# MAGIC                    
# MAGIC                

# COMMAND ----------

## Function to generate missing date rows between on START_DT and END_DT

def create_date_range(start_dt,end_dt):
    date_range_datetime=0
    date_index=0
    date_range_datetime = pd.date_range(start= start_dt, end=end_dt, freq='D', )
    date_index = list(date_range_datetime.strftime('%Y-%m-%d').values)
#     print("create date range function")
    return date_index
    

## Function to create dataframe with n rows [TRANS_DT range returned  from above function call] for BAN number with additional features like CRITICAL ACTION fields 
## ['BAN','DLNQ_DT','SUSP_DT','CRIT_ACT_DT','CRIT_ACT_TYPE_CD','TOTAL_DUE_AMT', 'last_cltn_risk_segment_cd','MONTHYR']
## Will try to get the hardcoding of column names automated
## The input to the below function will be a dataframe with sample BAN's
## DF structure :   BAN | DLNQ_DT | SUSP_DT | CRIT_ACT_DT | CRIT_ACT_TYPE_CD | TOTAL_DUE_AMT | last_cltn_risk_segment_cd |TRANS_MONY

def create_new_df(df):
    
    new_df = pd.DataFrame()
    df_group=df.groupby(['BAN', 'RISK_CD', 'INSTR_CD', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR', 'CURE_DT_PACIFIC', 'LG_SUSP_DT',
                       'LST_EVENT_DT', 'DAYS_TO_EVENT', 'TOT_DUE_AMT', 'TOT_DLNQ_AMT', 'T1', 'T2', 'T3', 'T4', 'T5']
                        , as_index = False).agg({'TRANS_MONYR': ' '.join})
    df_group.columns=['BAN', 'RISK_CD', 'INSTR_CD', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR', 'CURE_DT_PACIFIC', 'LG_SUSP_DT',
                       'LST_EVENT_DT', 'DAYS_TO_EVENT', 'TOT_DUE_AMT', 'TOT_DLNQ_AMT',
                      'T1', 'T2', 'T3', 'T4', 'T5','MONTHYR']
    first_time_counter=0
    
    for idx in tqdm(range(0, len(df_group.index))):
            temp_df=pd.DataFrame()
            start_of_mon=df_group.DLNQ_DT_PACIFIC[idx] 
            end_of_mon = min(df_group.LST_EVENT_DT[idx], df_group.LG_SUSP_DT[idx])
            temp_df['TRANS_DT'] = create_date_range(start_of_mon,end_of_mon)
            temp_df['BAN']=df_group.BAN[idx]
            temp_df['RISK_CD']=df_group.RISK_CD[idx]
            temp_df['INSTR_CD']=df_group.INSTR_CD[idx]
            temp_df['DLNQ_DT_PACIFIC']=df_group.DLNQ_DT_PACIFIC[idx]
            temp_df['DLNQ_DT_STR']=df_group.DLNQ_DT_STR[idx]
            temp_df['CURE_DT_PACIFIC']=df_group.CURE_DT_PACIFIC[idx]
            temp_df['LG_SUSP_DT']=df_group.LG_SUSP_DT[idx]
            temp_df['LST_EVENT_DT']=df_group.LST_EVENT_DT[idx]
            temp_df['DAYS_TO_EVENT']=df_group.DAYS_TO_EVENT[idx] 
            temp_df['TOT_DUE_AMT']=df_group.TOT_DUE_AMT[idx]
            temp_df['TOT_DLNQ_AMT']=df_group.TOT_DLNQ_AMT[idx]            
            temp_df['T1']=df_group.T1[idx]
            temp_df['T2']=df_group.T2[idx]
            temp_df['T3']=df_group.T3[idx]            
            temp_df['T4']=df_group.T4[idx]
            temp_df['T5']=df_group.T5[idx]            
            
            if(first_time_counter==0):
                new_df=temp_df
                first_time_counter+=1
            else:
                new_df = new_df.append(temp_df)            
    return new_df

## Function to help merge the 
##     a) newly created dataset with continuous TRANS_DT between the delinquent date and suspension date 
## and b) old dataframe with additional features [dummy based on TRANS_TYPE and TRANS_SUB_TYPE] 

def fit_prev_data(df_old, df_new):
    
    print(df_old.columns)
    print(df_new.columns)
    
    df_analysis=pd.DataFrame()
    
    df_analysis = df_new.merge(df_old,
                               how='left',
                               on=['BAN', 'DLNQ_DT_PACIFIC', 'TRANS_DT'],
                               indicator = True)

    df_analysis.sort_values(by=['BAN', 'DLNQ_DT_PACIFIC'],inplace=True)

    #df_analysis = df_analysis[(df_analysis["TRANS_DT"] >= df_analysis["DLNQ_DT"]) & (df_analysis["TRANS_DT"] <= df_analysis["DLNQ_DT"]+timedelta(days=int(29)))]
    
    df_analysis.reset_index(drop=True, inplace=True)
    #df_analysis.drop(['TRANS_MONYR','_merge'],inplace=True, axis=1)
    return df_analysis


# COMMAND ----------

# MAGIC %md
# MAGIC # <span style='color:Red'>  /////// 1st Block \\\\\\\\\\\\\ </span>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part l. Get historical account cycles from Aug 19/20

# COMMAND ----------

# MAGIC %md
# MAGIC ### DF Action <span style='color:Pink'> (Central to Pacific Time) </span>
# MAGIC    Select correct dlnq dt and cure dt for those whose cure dt is available, also consider no cure accnts at second moment (Part 2), so we get each dlnq cycle doing this:    
# MAGIC    1. Fix cure indicator label and get the fisrt cltn indicator label
# MAGIC    2. Based on these cltn indexes get the first cure indicator label available
# MAGIC    3. Keep and save indicators lost in this process to reintroduce them later

# COMMAND ----------

df_action = df_act[['acct_nbr', 'critical_action_dt', 'critical_action_type_cd', 'critical_action_ar_tot_due_amt']]
df_action.columns = ['BAN', 'CRIT_ACT_DT', 'CRIT_ACT_TYPE_CD', 'AR_TOTAL_DUE_AMT']

df_action['BAN'] = df_action['BAN'].astype(str)
df_action['TRANS_MONYR']=pd.to_datetime(df_action['CRIT_ACT_DT']).dt.strftime('%Y%m')
df_action['TRANS_MONYR'] = df_action['TRANS_MONYR'].astype(int)
df_action['CRIT_ACT_DT'] = df_action['CRIT_ACT_DT'].astype('datetime64[D]')
df_action['CRIT_ACT_TYPE_CD'] = df_action['CRIT_ACT_TYPE_CD'].str.strip()

df_action = df_action.reset_index(drop=True)

df_action = df_action.sort_values(by=['BAN','CRIT_ACT_DT','TRANS_MONYR'])

# COMMAND ----------

#%%time

s = df_action.groupby('TRANS_MONYR').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s.pivot_table(index = ['TRANS_MONYR'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Keep cohort periods with 3 months performance window

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC cohort_months = [201908,201909,201910,201911,
# MAGIC                  202008,202009,202010,202011]
# MAGIC 
# MAGIC df_action = df_action[df_action["TRANS_MONYR"].isin(cohort_months)]

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC s = df_action.groupby('TRANS_MONYR').size().reset_index(name='FREQ')
# MAGIC s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
# MAGIC s.pivot_table(index = ['TRANS_MONYR'],
# MAGIC              margins = True, 
# MAGIC              margins_name='Total',
# MAGIC              aggfunc=sum)

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC cohort_aug = [201908, 202008]
# MAGIC 
# MAGIC Accts_201908 = df_action.BAN[df_action["TRANS_MONYR"] == cohort_aug[0]].unique().tolist()
# MAGIC 
# MAGIC Cohort_201908 = df_action[(df_action["TRANS_MONYR"].isin(cohort_months[0:4])) &
# MAGIC                           (df_action["BAN"].isin(Accts_201908))]
# MAGIC 
# MAGIC Accts_202008 = df_action.BAN[df_action["TRANS_MONYR"] == cohort_aug[1]].unique().tolist()
# MAGIC 
# MAGIC Cohort_202008 = df_action[(df_action["TRANS_MONYR"].isin(cohort_months[4:8])) &
# MAGIC                           (df_action["BAN"].isin(Accts_202008))]
# MAGIC 
# MAGIC print(len(Accts_201908))
# MAGIC print(len(Cohort_201908))
# MAGIC print(len(Accts_202008))
# MAGIC print(len(Cohort_202008))

# COMMAND ----------

Cohort_Aug = pd.concat([Cohort_201908, Cohort_202008], axis=0)

# COMMAND ----------

s = Cohort_Aug.groupby('TRANS_MONYR').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s.pivot_table(index = ['TRANS_MONYR'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

s = Cohort_Aug.groupby('CRIT_ACT_TYPE_CD').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s.pivot_table(index = ['CRIT_ACT_TYPE_CD'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

Cohort_Aug["CRIT_ACT_TYPE_CD"].unique()

# COMMAND ----------

Cohort_Aug.head()

# COMMAND ----------

Cohort_Aug = Cohort_Aug.sort_values(['BAN','CRIT_ACT_DT'])

Cohort_Aug = Cohort_Aug.reset_index(drop=True)

# COMMAND ----------

Cohort_Aug['AR_TOTAL_DUE_AMT'] = pd.to_numeric(Cohort_Aug['AR_TOTAL_DUE_AMT'], errors='coerce')
Cohort_Aug['AR_TOTAL_DUE_AMT'].dtype

# COMMAND ----------

Cohort_Aug = Cohort_Aug.drop_duplicates()
print(Cohort_Aug.shape)

# COMMAND ----------

Cohort_Aug = Cohort_Aug.reset_index(drop=True)
Cohort_Aug.head()

# COMMAND ----------

Cohort_Aug.CRIT_ACT_DT = Cohort_Aug.CRIT_ACT_DT.astype('datetime64[D]')
Cohort_Aug.CRIT_ACT_DT.dtype

# COMMAND ----------

Cohort_Aug['TRANS_YEAR'] = Cohort_Aug['CRIT_ACT_DT'].dt.year

# COMMAND ----------

Cohort_Aug_02 = Cohort_Aug.reset_index()
Cohort_Aug_02.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pair first delinquent indicator (CLTN) with nearest event  between (WO, OCA, CURE)

# COMMAND ----------

Cohort_Aug_02['index'] = Cohort_Aug_02['index'].astype(str)
Cohort_Aug_GROUP = Cohort_Aug_02.groupby(['BAN', 'TRANS_YEAR'])[['index', 'CRIT_ACT_TYPE_CD']].agg(', '.join).reset_index()
Cohort_Aug_GROUP.columns = ['BAN', 'TRANS_YEAR', 'index_comb', 'CRIT_ACT_TYPE_CD_comb']
Cohort_Aug_GROUP.head()

# COMMAND ----------

Cohort_Aug_GROUP['ROW_CLEAN_UP'] = Cohort_Aug_GROUP.apply(lambda x: detect_rows_to_be_dropped(x['index_comb'], \
                                                            x['CRIT_ACT_TYPE_CD_comb'], ls_valid_crit_act_cd), axis=1)

# COMMAND ----------

Cohort_Aug_GROUP.head()

# COMMAND ----------

total_index_dropped = Cohort_Aug_GROUP['ROW_CLEAN_UP'].apply(lambda x: x[0]).tolist()

# COMMAND ----------

list_total_index_dropped = [item for sublist in total_index_dropped for item in sublist]

# COMMAND ----------

Cohort_Aug_NEW = Cohort_Aug.drop(list_total_index_dropped)
Cohort_Aug_NEW.shape

# COMMAND ----------

pd.set_option('display.max_rows', None)
Cohort_Aug_NEW['FLAG_CLTN'] = np.where(Cohort_Aug_NEW['CRIT_ACT_TYPE_CD'] == 'CLTN', 1, 0)  #add flag for CLTN

Cohort_Aug_NEW = Cohort_Aug_NEW.reset_index(drop=True)
#Cohort_Aug_NEW

# COMMAND ----------

s = Cohort_Aug_NEW.groupby('CRIT_ACT_TYPE_CD').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s.pivot_table(index = ['CRIT_ACT_TYPE_CD'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

Cohort_Aug_NEW['FIRST_EVENT'] = np.where(Cohort_Aug_NEW['CRIT_ACT_TYPE_CD'] != 'CLTN', 1, 0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cycle indicator

# COMMAND ----------

Cohort_Aug_NEW["FLAG_CYC"]=0

Cohort_Aug_NEW.loc[(Cohort_Aug_NEW['FLAG_CLTN']== 1) | (Cohort_Aug_NEW['FIRST_EVENT'] == 1),"FLAG_CYC"]=1

# COMMAND ----------

Cohort_Aug_NEW.head()

# COMMAND ----------

Cohort_Aug_NEW.shape

# COMMAND ----------

CYC_CURE = Cohort_Aug_NEW[Cohort_Aug_NEW["FLAG_CYC"]==1]

CYC_CURE = CYC_CURE.reset_index(drop=True)

# COMMAND ----------

CYC_CURE.shape[0]

# COMMAND ----------

s = CYC_CURE.groupby('CRIT_ACT_TYPE_CD').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s.pivot_table(index = ['CRIT_ACT_TYPE_CD'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

CYC_CURE.shape

# COMMAND ----------

# MAGIC %md
# MAGIC #### Assign an index for each pair

# COMMAND ----------

num = int(len(CYC_CURE.index)/2)+1

num

# COMMAND ----------

import itertools

lst = range(1,num)

xlist =  list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in lst))


# COMMAND ----------

len(xlist)

# COMMAND ----------

CYC_CURE['COMODIN'] = xlist


# COMMAND ----------

CYC_CURE['BANCOMODIN'] = CYC_CURE['BAN'].astype(str) + ' '+ CYC_CURE['COMODIN'].astype(str)


# COMMAND ----------

CYC_CURE_H = CYC_CURE.pivot_table(index=['BANCOMODIN'],
                                          columns = 'CRIT_ACT_TYPE_CD',
                                          aggfunc=min,
                                          values='CRIT_ACT_DT')


# COMMAND ----------

CYC_CURE_H.shape

# COMMAND ----------

CYC_CURE_H = CYC_CURE_H.reset_index()

# COMMAND ----------

CYC_CURE_H.info()

# COMMAND ----------

CYC_CURE_H['BAN'] = CYC_CURE_H.apply(lambda x: x['BANCOMODIN'].split()[0], axis=1)

# COMMAND ----------

CYC_CURE_H.head()

# COMMAND ----------

CYC_CURE_H = CYC_CURE_H.drop('BANCOMODIN', axis=1)

CYC_CURE_H = CYC_CURE_H[['BAN', 'CLTN', 'CURE',  'WOFF', 'OCAE']]

CYC_CURE_H.columns = ['BAN', 'DLNQ_DT', 'CURE_DT', 'WO_DT', 'OCA_DT']

# COMMAND ----------

CYC_CURE_H['CURE_DT'] = CYC_CURE_H.apply(lambda x: x['CURE_DT']+timedelta(days=int(1)), axis=1)

# COMMAND ----------

CYC_CURE_H.head(12)

# COMMAND ----------

CYC_CURE_H.tail(12)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Verify duplicates and na's

# COMMAND ----------

dup = CYC_CURE_H[CYC_CURE_H.duplicated(['BAN', 'DLNQ_DT'])]

dup

# COMMAND ----------

CYC_CURE_H.CURE_DT.isna().sum()

# COMMAND ----------

CYC_CURE_H.DLNQ_DT.isna().sum()

# COMMAND ----------

CYC_CURE_H.WO_DT.isna().sum()

# COMMAND ----------

CYC_CURE_H.OCA_DT.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Take those from august at dlnq dt since we could have more dlnq cohorts because window expansion

# COMMAND ----------

CYC_CURE_H.shape

# COMMAND ----------

CYC_CURE_H = CYC_CURE_H[CYC_CURE_H.DLNQ_DT.dt.month <= 8]

# COMMAND ----------

CYC_CURE_H.shape

# COMMAND ----------

len(CYC_CURE_H.BAN.unique())

# COMMAND ----------

dup = CYC_CURE_H[CYC_CURE_H.duplicated(['BAN', 'DLNQ_DT'])]

dup

# COMMAND ----------

CYC_CURE_H.head()

# COMMAND ----------

CYC_CURE_H.DLNQ_DT.dt.month.unique()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insert time

# COMMAND ----------

df_act.columns

# COMMAND ----------

df_action_tm = df_act[['acct_nbr', 'critical_action_dt', 'critical_action_time', 'critical_action_type_cd']]
df_action_tm.columns = ['BAN', 'CRIT_ACT_DT', 'CRIT_ACT_TIME', 'CRIT_ACT_TYPE_CD']

df_action_tm['BAN'] = df_action_tm['BAN'].astype(str)

df_action_tm['CRIT_ACT_TIME'] = pd.to_numeric(df_action_tm['CRIT_ACT_TIME'], errors='coerce')

df_action_tm['CRIT_ACT_TIME'] = df_action_tm['CRIT_ACT_TIME'].fillna(0)
df_action_tm['CRIT_ACT_TIME'] = df_action_tm.apply(lambda x: round(x['CRIT_ACT_TIME'], 0), axis=1)
df_action_tm['CRIT_ACT_TIME'] = df_action_tm['CRIT_ACT_TIME'].astype(int)
df_action_tm['CRIT_ACT_TIME'] = df_action_tm['CRIT_ACT_TIME'].astype(str)

df_action_tm['CRIT_ACT_DT'] = df_action_tm['CRIT_ACT_DT'].astype('datetime64[D]')
df_action_tm.CRIT_ACT_TIME = df_action_tm.apply(lambda x: x.CRIT_ACT_TIME.strip(), axis=1)

df_action_tm['CRIT_ACT_DT'] = df_action_tm.apply(lambda x: x['CRIT_ACT_DT']+timedelta(days=int(1)) 
                                                 if x.CRIT_ACT_TYPE_CD == 'CURE' else x['CRIT_ACT_DT'], axis=1)

df_action_tm = df_action_tm.sort_values(by=['BAN','CRIT_ACT_DT'])
df_action_tm = df_action_tm.reset_index(drop=True)

df_action_tm['len_time'] = df_action_tm.apply(lambda x: len(x.CRIT_ACT_TIME), axis=1)

df_action_tm.head()

# COMMAND ----------

df_action_tm['ACTION_DTTM_CLTN'] = df_action_tm.apply(lambda x: new_datetime(x, 'CRIT_ACT_TIME', 'CRIT_ACT_DT', central_tz), axis=1)
df_action_tm['ACTION_DTTM_TIMEZONE_CLTN'] = df_action_tm.apply(lambda x: x.ACTION_DTTM_CLTN.tzinfo, axis=1)
df_action_tm['ACTTION_DTTM_PACIFIC_TZ_CLTN'] = df_action_tm.apply(lambda x: x['ACTION_DTTM_CLTN'].astimezone(pacific_tz), axis=1)

# COMMAND ----------

df_action_tm.columns

# COMMAND ----------

df_action_tm = df_action_tm[['BAN', 'CRIT_ACT_DT', 'ACTION_DTTM_CLTN','ACTION_DTTM_TIMEZONE_CLTN', 'ACTTION_DTTM_PACIFIC_TZ_CLTN']]

# COMMAND ----------

df_action_tm[df_action_tm['BAN'] == '100057601']

# COMMAND ----------

dup = df_action_tm[df_action_tm.duplicated(['BAN', 'CRIT_ACT_DT'])]

dup.shape

# COMMAND ----------

df_action_tm = df_action_tm.drop_duplicates(['BAN', 'CRIT_ACT_DT'], keep='first')

# COMMAND ----------

dup = df_action_tm[df_action_tm.duplicated(['BAN', 'CRIT_ACT_DT'])]

dup.shape

# COMMAND ----------

CYC_CURE_H.shape

# COMMAND ----------

df_action_tm.columns = ['BAN', 'DLNQ_DT', 'ACTION_DTTM_CLTN','ACTION_DTTM_TIMEZONE_CLTN', 'ACTION_DTTM_PACIFIC_TZ_CLTN']

# COMMAND ----------

CYC_CURE_H = pd.merge(CYC_CURE_H,
                      df_action_tm, 
                      how='left',
                      left_on=['BAN', 'DLNQ_DT'],
                      right_on=['BAN', 'DLNQ_DT'])

# COMMAND ----------

df_action_tm.columns = ['BAN', 'CURE_DT', 'ACTION_DTTM_CURE','ACTION_DTTM_TIMEZONE_CURE', 'ACTION_DTTM_PACIFIC_TZ_CURE']

# COMMAND ----------

CYC_CURE_H = pd.merge(CYC_CURE_H,
                      df_action_tm, 
                      how='left',
                      left_on=['BAN', 'CURE_DT'],
                      right_on=['BAN', 'CURE_DT'])

# COMMAND ----------

CYC_CURE_H.shape

# COMMAND ----------

CYC_CURE_H[CYC_CURE_H['BAN'] == '100057601']

# COMMAND ----------

CYC_CURE_H.head()

# COMMAND ----------

CYC_CURE_H.dtypes

# COMMAND ----------

CYC_CURE_H.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part ll. Set the boundaries of dlnq cycles and prepare treatments

# COMMAND ----------

# MAGIC %md
# MAGIC ### New DF Call <span style='color:Pink'> (Central to Pacific Time) </span>

# COMMAND ----------

df_calls.head(1)

# COMMAND ----------

df_calls.call_type_cd.unique()

# COMMAND ----------

df_call_tz = df_calls[['acct_nbr', 'call_dt', 'call_time', 'call_type_cd', 'call_center_cd']]
df_call_tz.columns = ['BAN', 'CALL_DT', 'CALL_TIME', 'TRANS_TYPE', 'TRANS_SUB_TYPE']

df_call_tz['BAN'] = df_call_tz['BAN'].astype(str)
df_call_tz['CALL_DT'] = df_call_tz['CALL_DT'].astype('datetime64[D]')

df_call_tz = df_call_tz[df_call_tz['TRANS_TYPE'] != 'IB']

df_call_tz = df_call_tz.sort_values(by =['BAN','CALL_DT'], ascending=True)
df_call_tz = df_call_tz.reset_index(drop=True)

df_call_tz['TREAT_MSG_CD'] = df_call_tz.apply(lambda x: 'OUTBOUNDCALL' if x.TRANS_TYPE == 'OB' else 
                                              ('AD_CALL' if x.TRANS_TYPE == 'AD' else 'INBOUNDCALL'), axis=1)

print('Estas son las llamadas incluidas:'+str(df_call_tz.TREAT_MSG_CD.unique()))

#New datetime

df_call_tz['len_time'] = df_call_tz.apply(lambda x: len(str(x.CALL_TIME)), axis=1)
df_call_tz['CALL_DT_TZ'] = df_call_tz.apply(lambda x: new_datetime(x, 'CALL_TIME', 'CALL_DT', central_tz), axis=1)
df_call_tz['CALL_TIMEZONE'] = df_call_tz.apply(lambda x: x.CALL_DT_TZ.tzinfo, axis=1)
df_call_tz['CALL_DT_TZ_PACIFIC'] = df_call_tz.apply(lambda x: x['CALL_DT_TZ'].astimezone(pacific_tz), axis=1)

df_call_tz = df_call_tz.sort_values(by =['BAN','CALL_DT_TZ_PACIFIC'], ascending=True)
df_call_tz = df_call_tz.reset_index(drop=True)


df_call_tz['CALL_DT_PACIFIC'] = df_call_tz.apply(lambda x: x['CALL_DT_TZ_PACIFIC'].date(), axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### How many changes in day we would have

# COMMAND ----------

df_call_tz['IND_TIME_CHANGED_DAY'] = df_call_tz.apply(lambda x: 0 if x['CALL_DT_TZ'].date() ==  x['CALL_DT_TZ_PACIFIC'].date() else 1, axis=1)

# COMMAND ----------

df_call_tz['IND_TIME_CHANGED_DAY'].sum() #There are 70725 changes of day because of time zone difference

# COMMAND ----------

df_call_tz.shape

# COMMAND ----------

df_call_tz.head()

# COMMAND ----------

df_call_tz['CALL_DT_PACIFIC'] = df_call_tz['CALL_DT_PACIFIC'].astype('datetime64[D]')

# COMMAND ----------

df_call_tz_cut = df_call_tz[['BAN', 'CALL_DT', 'CALL_DT_TZ', 'TRANS_TYPE', 'TRANS_SUB_TYPE', 'TREAT_MSG_CD', 'CALL_DT_PACIFIC', 'CALL_DT_TZ_PACIFIC']]

# COMMAND ----------

df_call_tz_cut[(df_call_tz_cut["BAN"] == '100142994') & (df_call_tz_cut['CALL_DT_TZ_PACIFIC'].dt.month == 8)]

# COMMAND ----------

df_call_tz_cut.dtypes

# COMMAND ----------

dup = df_call_tz_cut[df_call_tz_cut.duplicated(['BAN', 'CALL_DT', 'TREAT_MSG_CD', 'CALL_DT_PACIFIC', 'CALL_DT_TZ_PACIFIC'])]

print(len(dup))

dup.head()

# COMMAND ----------

df_call_tz_cut = df_call_tz_cut.drop_duplicates(['BAN', 'CALL_DT', 'TREAT_MSG_CD', 'CALL_DT_PACIFIC', 'CALL_DT_TZ_PACIFIC'], keep='first')

# COMMAND ----------

dup = df_call_tz_cut[df_call_tz_cut.duplicated(['BAN', 'CALL_DT', 'TREAT_MSG_CD', 'CALL_DT_PACIFIC', 'CALL_DT_TZ_PACIFIC'])]

print(len(dup))

dup.head()

# COMMAND ----------

df_call_tz_cut.shape

# COMMAND ----------

df_call_tz_cut.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ### DF MESSAGE <span style='color:Pink'> (Central to Pacific Time) </span>

# COMMAND ----------

df_treatment_timestamp.head(1)

# COMMAND ----------

# df_treatment_timestamp.timezone_cd.unique()

# COMMAND ----------

df_message_tz = df_treatment_timestamp[['acct_nbr', 'message_dt', 'message_type_cd', 'message_subtype_cd',
                                'message_cltn_sys_letter_cd', 'Message_Dt_Tm', 'timezone_cd']]

df_message_tz.columns = ['BAN', 'MESSAGE_DT', 'TRANS_TYPE', 'TRANS_SUB_TYPE', 'TREAT_MSG_CD',
                         'MESSAGE_DT_TM', 'TIMEZONE_CD']

df_message_tz['BAN'] = df_message_tz['BAN'].astype(str)
df_message_tz['MESSAGE_DT'] = df_message_tz['MESSAGE_DT'].astype('datetime64[D]')
df_message_tz['MESSAGE_DT_TM'] = pd.to_datetime(df_message_tz['MESSAGE_DT_TM'], errors = 'coerce')

df_message_tz = df_message_tz.sort_values(by =['BAN','MESSAGE_DT'], ascending=True)
df_message_tz = df_message_tz.reset_index(drop=True)

# COMMAND ----------

df_message_tz['tm_msg_trans'] = df_message_tz.apply(lambda x: time(x.MESSAGE_DT_TM.hour, 
                                                                   x.MESSAGE_DT_TM.minute,
                                                                   x.MESSAGE_DT_TM.second), axis=1)

# COMMAND ----------

df_message_tz['MESSAGE_DT']= pd.to_datetime(df_message_tz['MESSAGE_DT'], format="%m/%d/%Y %H:%M:%S")
df_message_tz['MESSAGE_DATETIME'] = df_message_tz.apply(lambda x: datetime.combine(x.MESSAGE_DT, x.tm_msg_trans), axis=1)
df_message_tz['MESSAGE_DATETIME'] = df_message_tz.apply(lambda x: central_tz.localize(x['MESSAGE_DATETIME'], is_dst=True), axis=1)
df_message_tz['MESSAGE_DATETIME_PACIFIC'] = df_message_tz.apply(lambda x: x['MESSAGE_DATETIME'].astimezone(pacific_tz), axis=1)

# COMMAND ----------

df_message_tz['MESSAGE_DT_PACIFIC'] = df_message_tz.apply(lambda x: x.MESSAGE_DATETIME_PACIFIC.date(), axis=1)

# COMMAND ----------

df_message_tz['MESSAGE_DT_PACIFIC'] = df_message_tz['MESSAGE_DT_PACIFIC'].astype('datetime64[D]')

# COMMAND ----------

df_message_tz = df_message_tz.sort_values(by=['BAN', 'MESSAGE_DATETIME_PACIFIC', 'MESSAGE_DT_PACIFIC', 'TREAT_MSG_CD'], 
                                               ascending=[True, True, True, True])
df_message_tz = df_message_tz.reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Dismiss messages codes

# COMMAND ----------

df_message_tz_cut = df_message_tz[['BAN', 'MESSAGE_DT', 'MESSAGE_DATETIME', 'TRANS_TYPE', 'TRANS_SUB_TYPE', 
                                   'TREAT_MSG_CD', 'MESSAGE_DT_PACIFIC', 'MESSAGE_DATETIME_PACIFIC']]   

# COMMAND ----------

df_message_tz_cut[(df_message_tz_cut["BAN"] == '100142994') & (df_message_tz_cut['MESSAGE_DATETIME_PACIFIC'].dt.month == 8)]

# COMMAND ----------

dup = df_message_tz_cut[df_message_tz_cut.duplicated(['BAN', 'MESSAGE_DT', 'TREAT_MSG_CD', 'MESSAGE_DT_PACIFIC', 'MESSAGE_DATETIME_PACIFIC'])]

print(len(dup))

dup.head()


# COMMAND ----------

df_message_tz_cut[(df_message_tz_cut["BAN"] == '100057601')]

# COMMAND ----------

df_message_tz_cut = df_message_tz_cut.drop_duplicates(['BAN', 'MESSAGE_DT', 'TREAT_MSG_CD', 'MESSAGE_DT_PACIFIC', 'MESSAGE_DATETIME_PACIFIC'], keep='first')

# COMMAND ----------

dup = df_message_tz_cut[df_message_tz_cut.duplicated(['BAN', 'MESSAGE_DT', 'TREAT_MSG_CD', 'MESSAGE_DT_PACIFIC', 'MESSAGE_DATETIME_PACIFIC'])]

print(len(dup))

dup.head()



# COMMAND ----------

df_message_tz_cut[(df_message_tz_cut["BAN"] == '100057601')]

# COMMAND ----------

df_message_tz_cut.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC # <span style='color:Orange'> Approach </span>
# MAGIC - Mix calls and messages to replace TRANS_DT with the earliest time treatment from MESSAGE_DT_TM_PACIFIC or CALL_DT_TZ_PACIFIC ocurred the same day using Pacif time zone
# MAGIC - If no treatment then let same TRANS_DT with 00:00:00 hour in pacific time zone

# COMMAND ----------

df_message_tz_cut.columns = ['BAN', 'TREAT_DT', 'TREAT_DTTM', 'TRANS_TYPE', 'TRANS_SUB_TYPE',
                             'TREAT_MSG_CD', 'TREAT_DT_PACIFIC', 'TREAT_DTTM_PACIFIC']

# COMMAND ----------

df_call_tz_cut.columns = ['BAN', 'TREAT_DT', 'TREAT_DTTM', 'TRANS_TYPE', 'TRANS_SUB_TYPE',
                             'TREAT_MSG_CD', 'TREAT_DT_PACIFIC', 'TREAT_DTTM_PACIFIC']

# COMMAND ----------

df_treatment_tz = pd.concat([df_message_tz_cut, df_call_tz_cut], axis=0)

# COMMAND ----------

df_treatment_tz = df_treatment_tz.sort_values(by =['BAN','TREAT_DTTM_PACIFIC'], ascending=True)
df_treatment_tz = df_treatment_tz.reset_index(drop=True)

# COMMAND ----------

df_treatment_tz.dtypes

# COMMAND ----------

df_treatment_tz.TRANS_TYPE = df_treatment_tz.apply(lambda x: x.TRANS_TYPE.upper(), axis=1)

# COMMAND ----------

df_treatment_tz.TRANS_TYPE.unique()

# COMMAND ----------

df_treatment_tz.TRANS_SUB_TYPE.unique()

# COMMAND ----------

df_treatment_tz.TREAT_MSG_CD.unique()

# COMMAND ----------

df_treatment_tz[(df_treatment_tz["BAN"] == '100142994') & (df_treatment_tz['TREAT_DTTM_PACIFIC'].dt.month >= 8) & (df_treatment_tz['TREAT_DTTM_PACIFIC'].dt.year >= 2019)]

# COMMAND ----------

df_treatment_tz[(df_treatment_tz["BAN"] == '100057601') & (df_treatment_tz['TREAT_DTTM_PACIFIC'].dt.month >= 8) & (df_treatment_tz['TREAT_DTTM_PACIFIC'].dt.year >= 2019)]

# COMMAND ----------

df_treatment_tz.shape

# COMMAND ----------

df_treatment_tz_first = df_treatment_tz.drop_duplicates(['BAN','TREAT_DT_PACIFIC'], keep='first')

# COMMAND ----------

df_treatment_tz_first.shape

# COMMAND ----------

df_treatment_tz_first = df_treatment_tz_first.sort_values(by =['BAN','TREAT_DTTM_PACIFIC'], ascending=True)
df_treatment_tz_first = df_treatment_tz_first.reset_index(drop=True)

# COMMAND ----------

df_treatment_tz_first[(df_treatment_tz_first["BAN"] == '100142994') & (df_treatment_tz_first['TREAT_DTTM_PACIFIC'].dt.month >= 8) & (df_treatment_tz_first['TREAT_DTTM_PACIFIC'].dt.year >= 2019)]

# COMMAND ----------

df_treatment_tz_first[(df_treatment_tz_first["BAN"] == '100057601') & (df_treatment_tz_first['TREAT_DTTM_PACIFIC'].dt.month >= 8) & (df_treatment_tz_first['TREAT_DTTM_PACIFIC'].dt.year >= 2019)]

# COMMAND ----------

df_treatment_tz_first.columns

# COMMAND ----------

df_treatment_tz_first.columns = ['BAN', 'TREAT_DT', 'TREAT_DTTM',
                                 'TRANS_TYPE_TIMECUTOFF', 'TRANS_SUB_TYPE_TIMECUTOFF',
                                 'TREAT_MSG_CD_TIMECUTOFF',
                                 'TREAT_DT_PACIFIC_CUTOFF', 'TREAT_DTTM_PACIFIC_CUTOFF']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part lll. Calculate proj susp dt

# COMMAND ----------

# MAGIC %md
# MAGIC #### Merge with Strata before

# COMMAND ----------

Dict_risk = {"NT": 0, "LL": 1, "LM": 2, "LH": 3, "ML": 4, "MM": 5,
             "MH": 6, "HL": 7, "HM": 8, "HH": 9, "FP": 10, "SH": 11,
             "LT":12, "NF":13, "CR":14}

# COMMAND ----------

# MAGIC %md
# MAGIC ###  DF Strata 
# MAGIC 
# MAGIC    Matching strata by decision date (dlnq date) with critical action file with corresponging delinquent date. It has the information we need instead of Cohort DF, we get from this df next variables:
# MAGIC    1. Risk code
# MAGIC    2. Strata Instruction code
# MAGIC    3. Total due amount and total delinquent amount

# COMMAND ----------

# Extract BAN, DLNQ_DT, RISK, DUE AMT:
df_strata = df_str[["acct_nbr", "strata_decision_dt", "strata_cltn_risk_segment_cd", "strata_instruction_cd", "strata_tot_due_amt", "strata_tot_dlnq_amt"]]
df_strata.columns = ['BAN', 'DLNQ_DT_STR', 'RISK_CD', "INSTR_CD", 'TOT_DUE_AMT', 'TOT_DLNQ_AMT']

df_strata['BAN'] = df_strata['BAN'].astype(str)
df_strata['DLNQ_DT_STR'] = df_strata['DLNQ_DT_STR'].astype('datetime64[D]')
df_strata['TRANS_MONYR']=pd.to_datetime(df_strata['DLNQ_DT_STR']).dt.strftime('%Y%m')

df_strata = df_strata.sort_values(by=['BAN','DLNQ_DT_STR','TRANS_MONYR'])
df_strata = df_strata.reset_index(drop=True)

# COMMAND ----------

s = df_strata.groupby('RISK_CD').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s["index"] = s['RISK_CD'].map(Dict_risk)
s.set_index("index").sort_values("index")
s.pivot_table(index = ['index', 'RISK_CD'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merging delinquet cycles with STRATA DF

# COMMAND ----------

CYC_CURE_H_STR =  pd.merge(CYC_CURE_H, df_strata, how='left', left_on=['BAN'], right_on=['BAN'])

# COMMAND ----------

CYC_CURE_H_STR.shape

# COMMAND ----------

CYC_CURE_H_STR.head()

# COMMAND ----------

CYC_CURE_H_STR.dtypes

# COMMAND ----------

len(CYC_CURE_H_STR.BAN.unique())

# COMMAND ----------

CYC_CURE_H_STR[CYC_CURE_H_STR['BAN']=='120653395'] #To Review

# COMMAND ----------

CYC_CURE_H_STR2 = CYC_CURE_H_STR[(CYC_CURE_H_STR['DLNQ_DT_STR'] <= CYC_CURE_H_STR['DLNQ_DT'])]

# COMMAND ----------

# MAGIC %md
# MAGIC #### There were 173 accts didn't have delinquent date lower than df_Action

# COMMAND ----------

len(CYC_CURE_H_STR2.BAN.unique())

# COMMAND ----------

4225-4052

# COMMAND ----------

accts = sorted(set(CYC_CURE_H_STR.BAN.unique().tolist()).difference(set(CYC_CURE_H_STR2.BAN.unique().tolist())), key=str.lower)
print(accts)

# COMMAND ----------

print(sorted(set(CYC_CURE_H_STR2.BAN.unique().tolist()).difference(set(CYC_CURE_H_STR.BAN.unique().tolist())), key=str.lower))

# COMMAND ----------

CYC_CURE_H_STR2.shape

# COMMAND ----------

# MAGIC %md
# MAGIC #### Drop duplicates with same dlnq dt but different risk

# COMMAND ----------

dup = CYC_CURE_H_STR2[CYC_CURE_H_STR2.duplicated(['BAN', 'DLNQ_DT'])]

len(dup)

# COMMAND ----------

CYC_CURE_H_STR2 = CYC_CURE_H_STR2.sort_values(by=['BAN', 'DLNQ_DT_STR', 'RISK_CD'], ascending=False)

CYC_CURE_H_STR2 = CYC_CURE_H_STR2.reset_index(drop = True)

# COMMAND ----------

CYC_CURE_H_STR2 = CYC_CURE_H_STR2.sort_values(['BAN','DLNQ_DT']).drop_duplicates(['BAN', 'DLNQ_DT'], keep='first')

# COMMAND ----------

CYC_CURE_H_STR2.columns

# COMMAND ----------

CYC_CURE_H_STR2.shape

# COMMAND ----------

CYC_CURE_H.shape

# COMMAND ----------

CYC_CURE_H_STR2.shape[0]-CYC_CURE_H.shape[0]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Replace NAN risk cd with NF = not found

# COMMAND ----------

CYC_CURE_H_STR2.RISK_CD.isna().sum()

# COMMAND ----------

CYC_CURE_H_STR2[CYC_CURE_H_STR2.RISK_CD.isna() == True].head()

# COMMAND ----------

CYC_CURE_H_STR2.RISK_CD = CYC_CURE_H_STR2.RISK_CD.replace(np.nan, 'NF')

# COMMAND ----------

s = CYC_CURE_H_STR2.replace('  ', 'NONE').groupby('RISK_CD').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s["index"] = s['RISK_CD'].map(Dict_risk)
s.set_index("index").sort_values("index")
s.pivot_table(index = ['index', 'RISK_CD'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Merge cycles with first treatment by dlnq dt without standarize

# COMMAND ----------

CYC_CURE_H_STR2.shape

# COMMAND ----------

CYC_CURE_H_STR2.ACTION_DTTM_PACIFIC_TZ_CLTN.dt.month.unique()

# COMMAND ----------

CYC_CURE_H_STR2.ACTION_DTTM_PACIFIC_TZ_CLTN.dt.year.unique()

# COMMAND ----------

CYC_CURE_H_STR2[CYC_CURE_H_STR2.ACTION_DTTM_PACIFIC_TZ_CLTN.dt.month==7]

# COMMAND ----------

CYC_CURE_H_STR2.head()

# COMMAND ----------

CYC_CURE_H_TR = pd.merge(CYC_CURE_H_STR2,
                          df_treatment_tz_first,
                          how='left',
                          left_on=['BAN', 'DLNQ_DT_STR'], right_on=['BAN', 'TREAT_DT'])

# COMMAND ----------

CYC_CURE_H_TR['CURE_DT_PACIFIC'] = CYC_CURE_H_TR.apply(lambda x: x.ACTION_DTTM_PACIFIC_TZ_CURE.date(), axis=1)

# COMMAND ----------

CYC_CURE_H_TR.shape

# COMMAND ----------

CYC_CURE_H_TR[(CYC_CURE_H_TR["BAN"] == '100057601')] 

# COMMAND ----------

df_treatment_tz[(df_treatment_tz["BAN"] == '100057601')] 

# COMMAND ----------

CYC_CURE_H_TR[(CYC_CURE_H_TR["BAN"] == '100142994')] 

# COMMAND ----------

df_treatment_tz[(df_treatment_tz["BAN"] == '100142994')] 

# COMMAND ----------

CYC_CURE_H_TR.isna().sum()

# COMMAND ----------

CYC_CURE_H_TR[(CYC_CURE_H_TR["TREAT_DT"].isnull())].head()

# COMMAND ----------

df_act[(df_act["acct_nbr"] == 100304470)].sort_values(by='critical_action_dt', ascending=True)

# COMMAND ----------

df_treatment_tz[(df_treatment_tz["BAN"] == '100304470')] 

# COMMAND ----------

CYC_CURE_H_TR.head()

# COMMAND ----------

CYC_CURE_H_TR['TREAT_DT'] = CYC_CURE_H_TR['TREAT_DT'].astype('datetime64[D]')

# COMMAND ----------

CYC_CURE_H_TR.head()

# COMMAND ----------

CYC_CURE_H_TR['DLNQ_DT_PACIFIC'] = np.where(CYC_CURE_H_TR['TREAT_DT'].isna(),
                                          CYC_CURE_H_TR['DLNQ_DT_STR'], 
                                                CYC_CURE_H_TR['TREAT_DT_PACIFIC_CUTOFF'])

# COMMAND ----------

CYC_CURE_H_TR.head()

# COMMAND ----------

CYC_CURE_H_TR.columns

# COMMAND ----------

CYC_CURE_H_TR[CYC_CURE_H_TR['BAN']=='120653395']

# COMMAND ----------

CYC_CURE_H_TR2 = CYC_CURE_H_TR[['BAN', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR', 'CURE_DT_PACIFIC', 'WO_DT', 'OCA_DT',
       'RISK_CD', 'INSTR_CD', 'TOT_DUE_AMT', 'TOT_DLNQ_AMT', 'TRANS_MONYR']]

# COMMAND ----------

CYC_CURE_H_TR2.head()

# COMMAND ----------

CYC_CURE_H_TR2.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get suspension date 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Treatment Timeline by Risk

# COMMAND ----------

CYC_CURE_H_TR2.RISK_CD.unique() # There are some blanks

# COMMAND ----------

CYC_CURE_H_TR2.RISK_CD = CYC_CURE_H_TR2.RISK_CD.replace('  ', 'NF')

# COMMAND ----------

s = CYC_CURE_H_STR2.groupby('RISK_CD').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s["index"] = s['RISK_CD'].map(Dict_risk)
s.set_index("index").sort_values("index")
s.pivot_table(index = ['index', 'RISK_CD'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

#### Updatad for EG

data = [['NT',0,0,0,0,32],  #Same as LL?
        ['LL',0,0,0,0,32], 
        ['LM',0,0,0,0,30], 
        ['LH',0,0,0,0,28], 
        ['ML',0,0,0,0,25], 
        ['MM',0,0,0,0,19], 
        ['MH',0,0,0,0,13], 
        ['HL',0,0,0,0,10], 
        ['HM',0,0,0,0,9], 
        ['HH',0,0,0,0,8], 
        ['LT',0,0,0,0,15], 
        ['FP',0,0,0,0,15],
        ['SH',0,0,0,0,15], #Same as LT?
        ['NF',0,0,0,0,15],  #Same as LT if Not Found / Missing
        ['CR',0,0,0,0,15],  #Same as LT?
       ]

df_timeline_susp = pd.DataFrame(data, columns=['Risk', 'Frie_SMS_Email_Days', 'Ent_Dlnq_SMS_Days', 'PreSusp_SMS_Email_Days', 'PreSusp_Lett_Days', 'Susp_Days'])
df_timeline_susp["index"] = df_timeline_susp['Risk'].map(Dict_risk)
df_timeline_susp = df_timeline_susp.set_index('Risk')
df_timeline_susp = df_timeline_susp.sort_values("index")
df_timeline_susp

# COMMAND ----------

CYC_CURE_H_TR2['T1'] = CYC_CURE_H_TR2.apply(lambda x: timeline_days(x['RISK_CD'], 1), axis=1)
CYC_CURE_H_TR2['T2'] = CYC_CURE_H_TR2.apply(lambda x: timeline_days(x['RISK_CD'], 2), axis=1)
CYC_CURE_H_TR2['T3'] = CYC_CURE_H_TR2.apply(lambda x: timeline_days(x['RISK_CD'], 3), axis=1)
CYC_CURE_H_TR2['T4'] = CYC_CURE_H_TR2.apply(lambda x: timeline_days(x['RISK_CD'], 4), axis=1)
CYC_CURE_H_TR2['T5'] = CYC_CURE_H_TR2.apply(lambda x: timeline_days(x['RISK_CD'], 5), axis=1)
CYC_CURE_H_TR2['PROJ_SUSP_DT'] = CYC_CURE_H_TR2.apply(lambda x: (adding_business_days(x['DLNQ_DT_PACIFIC'], x['T5'])), axis=1)

# COMMAND ----------

CYC_CURE_H_TR2.columns

# COMMAND ----------

CYC_CURE_H_TR2.head(15)

# COMMAND ----------

s = CYC_CURE_H_TR2.groupby('TRANS_MONYR').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s.pivot_table(index = ['TRANS_MONYR'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

CYC_CURE_H_TR2.shape

# COMMAND ----------

CYC_CURE_H_TR2.columns

# COMMAND ----------

CYC_CURE_H_TR2.dtypes

# COMMAND ----------

CYC_CURE_H_TR2[CYC_CURE_H_TR2['BAN']=='120653395']

# COMMAND ----------

#After create all the T's
df_base = CYC_CURE_H_TR2[["BAN", "RISK_CD", "INSTR_CD", "DLNQ_DT_PACIFIC", "DLNQ_DT_STR", 'CURE_DT_PACIFIC', 'WO_DT', 'OCA_DT',
                           "PROJ_SUSP_DT", 'TOT_DUE_AMT', 'TOT_DLNQ_AMT', "T1", "T2", "T3", "T4", "T5"]]

df_base = df_base.reset_index(drop = True)

# COMMAND ----------

df_base.head()

# COMMAND ----------

df_base.info()

# COMMAND ----------

df_base['CURE_DT_PACIFIC'] = df_base['CURE_DT_PACIFIC'].astype('datetime64[D]')
df_base["DAYS_TO_CURE"] = (df_base["CURE_DT_PACIFIC"]-df_base["DLNQ_DT_PACIFIC"]).dt.days

df_base = df_base[["BAN", "RISK_CD", "INSTR_CD", "DLNQ_DT_PACIFIC", 'DLNQ_DT_STR', 'CURE_DT_PACIFIC', 'WO_DT', 'OCA_DT',
                   'DAYS_TO_CURE', "PROJ_SUSP_DT", 'TOT_DUE_AMT', 'TOT_DLNQ_AMT', "T1", "T2", "T3", "T4", "T5"]]

df_base = df_base.sort_values(by=["BAN", "DLNQ_DT_PACIFIC"])

# COMMAND ----------

df_base.shape

# COMMAND ----------

df_base.head()

# COMMAND ----------

df_action[df_action['BAN'] == '100057601']

# COMMAND ----------

df_act[df_act['acct_nbr'] == 100057601]

# COMMAND ----------

# MAGIC %md
# MAGIC ## DF PAR
# MAGIC Based on DF PAR and DF Strata get the largest suspension date, prepare PAR file

# COMMAND ----------

df_PAR_2019.head(1)

# COMMAND ----------

df_PAR_2019.trans_dt_tm.unique() # Enabler does not have this field

# COMMAND ----------

# --- DF PAR does have timestamps ---

date_cols = ['pyarr_rcmnd_dttm', 'inpt_req_dttm', 'pyarr_scr_dttm', 'rcmnd_prjctd_supd_dt', 'prjctd_supd_dt',
             'most_recnt_pmt_dt',
            'curr_bl_dt', 'trans_dt_tm', 'pmt_pln_crtn_dt_tm']

for col in date_cols:
        df_PAR_2019[col] = pd.to_datetime(df_PAR_2019[col], errors = 'coerce')

df_PAR_2019[['pyarr_rcmnd_dttm', 'inpt_req_dttm', 'pyarr_scr_dttm', 'rcmnd_prjctd_supd_dt', 'prjctd_supd_dt',
             'most_recnt_pmt_dt',
            'curr_bl_dt', 'trans_dt_tm', 'pmt_pln_crtn_dt_tm']].head(1)

# COMMAND ----------

date_cols = ['pyarr_rcmnd_dttm', 'inpt_req_dttm', 'pyarr_scr_dttm', 'rcmnd_prjctd_supd_dt', 'prjctd_supd_dt',
             'most_recnt_pmt_dt',
            'curr_bl_dt', 'trans_dt_tm', 'pmt_pln_crtn_dt_tm']

for col in date_cols:
        df_PAR_2020[col] = pd.to_datetime(df_PAR_2020[col], errors = 'coerce')

df_PAR_2020[['pyarr_rcmnd_dttm', 'inpt_req_dttm', 'pyarr_scr_dttm', 'rcmnd_prjctd_supd_dt', 'prjctd_supd_dt',
             'most_recnt_pmt_dt',
            'curr_bl_dt', 'trans_dt_tm', 'pmt_pln_crtn_dt_tm']].head(1)

# COMMAND ----------

df_PAR_19 = df_PAR_2019.copy()

df_PAR_19 = df_PAR_19[['acct_nbr', 'acct_sts_cd',
                       'cohort_data_capture_dt', 'curr_bl_dt',
                       'inpt_req_dttm', 'pyarr_rcmnd_dttm', 
                       'prjctd_supd_dt', 'rcmnd_prjctd_supd_dt', 'trans_dt_tm',                   
                       'crdt_card_pmt_ind', 'elec_chk_pmt_ind', 'atpy_ind',
                       'pyarr_scr_dttm', 'pyarr_scr_nbr', 'pyarr_rsk_sgmnt_cd',
                       'tot_due_amt', 'most_recnt_pmt_amt', 'most_recnt_pmt_dt',
                       'pmt_arng_ind', 'exst_pyarr_ind' ,'inpt_app_id', 
                       'rcmnd_exprn_cd', 'rcmnd_exprn_txt', 'rcmnd_type_cd',
                       'blng_st_cd', 'srv_st_cd']]

df_PAR_19.columns = ['BAN', 'STATUS_CD', 
                     'CAPTURE_DT', 'CURR_BILL_DT',
                     'INIT_REQ_DTTM', 'PAR_RCMND_DTTM',
                     'PAR_PROJ_SUSP_DT', 'RCMD_PROJ_SUSP_DT', 'PAR_TRANS_DTTM',
                     'CREDIT_CARD_PMT_IND', 'ELEC_CHK_PMT_IND', 'ATPY_IND',
                     'PAR_SCR_DTTM', 'PAR_SCR_NBR', 'PAR_RISK_SGMT_CD',                       
                     'PAR_TOT_DUE_AMT', 'MOST_RECNT_PMT_AMT', 'MOST_RECNT_PMT_DT',
                     'PMT_ARRNG_IND', 'EXST_PYARR_IND', 'INPT_APP_ID',
                     'RCMD_EXPRN_CD', 'RCMD_EXPRN_TXT', 'RCMD_TYP_CD',
                     'BLNG_ST_CD', 'SRV_ST_CD']

df_PAR_19['BAN'] = df_PAR_19['BAN'].astype(str)

df_PAR_19 = df_PAR_19.sort_values(by=['BAN','CAPTURE_DT','INIT_REQ_DTTM','PAR_RCMND_DTTM','PAR_PROJ_SUSP_DT'],
                                  ascending=[False, False, False, False, False])

df_PAR_19 = df_PAR_19.reset_index(drop=True)

# COMMAND ----------

df_PAR_20 = df_PAR_2020.copy()

df_PAR_20 = df_PAR_20[['acct_nbr', 'acct_sts_cd',
                       'cohort_data_capture_dt', 'curr_bl_dt',
                       'inpt_req_dttm', 'pyarr_rcmnd_dttm',  
                       'prjctd_supd_dt', 'rcmnd_prjctd_supd_dt', 'trans_dt_tm',                     
                       'crdt_card_pmt_ind', 'elec_chk_pmt_ind', 'atpy_ind',
                       'pyarr_scr_dttm', 'pyarr_scr_nbr', 'pyarr_rsk_sgmnt_cd',
                       'tot_due_amt', 'most_recnt_pmt_amt', 'most_recnt_pmt_dt',
                       'pmt_arng_ind', 'exst_pyarr_ind' ,'inpt_app_id', 
                       'rcmnd_exprn_cd', 'rcmnd_exprn_txt', 'rcmnd_type_cd',
                       'blng_st_cd', 'srv_st_cd']]

df_PAR_20.columns = ['BAN', 'STATUS_CD', 
                     'CAPTURE_DT', 'CURR_BILL_DT',
                     'INIT_REQ_DTTM', 'PAR_RCMND_DTTM',
                     'PAR_PROJ_SUSP_DT', 'RCMD_PROJ_SUSP_DT', 'PAR_TRANS_DTTM',
                     'CREDIT_CARD_PMT_IND', 'ELEC_CHK_PMT_IND', 'ATPY_IND',
                     'PAR_SCR_DTTM', 'PAR_SCR_NBR', 'PAR_RISK_SGMT_CD',                       
                     'PAR_TOT_DUE_AMT', 'MOST_RECNT_PMT_AMT', 'MOST_RECNT_PMT_DT',
                     'PMT_ARRNG_IND', 'EXST_PYARR_IND', 'INPT_APP_ID',
                     'RCMD_EXPRN_CD', 'RCMD_EXPRN_TXT', 'RCMD_TYP_CD',
                     'BLNG_ST_CD', 'SRV_ST_CD']

df_PAR_20['BAN'] = df_PAR_20['BAN'].astype(str)


df_PAR_20 = df_PAR_20.sort_values(by=['BAN','CAPTURE_DT','INIT_REQ_DTTM','RCMD_PROJ_SUSP_DT','PAR_PROJ_SUSP_DT'],
                                  ascending=[False, False, False, False, False])

df_PAR_20 = df_PAR_20.reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### concatenating both cohorts 

# COMMAND ----------

df_PAR = pd.concat([df_PAR_19, df_PAR_20], axis=0)

# COMMAND ----------

df_PAR = df_PAR[['BAN', 'CAPTURE_DT', 'INIT_REQ_DTTM', 'PAR_RCMND_DTTM',
                 'PAR_PROJ_SUSP_DT', 'RCMD_PROJ_SUSP_DT', 'PAR_TRANS_DTTM', 'PAR_SCR_DTTM', 'MOST_RECNT_PMT_DT',
                 'STATUS_CD', 'CURR_BILL_DT', 'CREDIT_CARD_PMT_IND', 'ELEC_CHK_PMT_IND', 'ATPY_IND',
                 'PAR_SCR_NBR', 'PAR_RISK_SGMT_CD',                       
                 'PAR_TOT_DUE_AMT', 'MOST_RECNT_PMT_AMT',
                 'PMT_ARRNG_IND', 'EXST_PYARR_IND', 'INPT_APP_ID',
                 'RCMD_EXPRN_CD', 'RCMD_EXPRN_TXT', 'RCMD_TYP_CD',
                 'BLNG_ST_CD', 'SRV_ST_CD']]


# COMMAND ----------

df_PAR.head()

# COMMAND ----------

df_PAR['TRANS_MONYR']=pd.to_datetime(df_PAR['INIT_REQ_DTTM']).dt.strftime('%Y%m')
df_PAR['TRANS_MONYR'] = df_PAR['TRANS_MONYR'].astype(int)

# COMMAND ----------

s = df_PAR.groupby('TRANS_MONYR').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s.pivot_table(index = ['TRANS_MONYR'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Selecting cohorts that will be used (Aug 2019-2020)

# COMMAND ----------

df_PAR = df_PAR[(df_PAR['TRANS_MONYR']==201908) | (df_PAR['TRANS_MONYR']==202008)]

# COMMAND ----------

df_PAR = df_PAR.sort_values(by=['BAN','CAPTURE_DT','INIT_REQ_DTTM','RCMD_PROJ_SUSP_DT','PAR_PROJ_SUSP_DT'],
                                  ascending=[False, False, False, False, False])

df_PAR = df_PAR.reset_index(drop=True)

# COMMAND ----------

s = df_PAR.groupby('TRANS_MONYR').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s.pivot_table(index = ['TRANS_MONYR'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

df_PAR['CAPTURE_DT'] = df_PAR['CAPTURE_DT'].astype('datetime64[D]')

# COMMAND ----------

df_PAR[df_PAR['BAN']=='100057601']

# COMMAND ----------

df_base[df_base['BAN']=='100057601']

# COMMAND ----------

df_base.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merge with DF PAR and get SUSP DT

# COMMAND ----------

df_base_pr = pd.merge(df_base, df_PAR[['BAN', 'CAPTURE_DT', 'INIT_REQ_DTTM', 
                                       'PAR_RCMND_DTTM', 'PAR_PROJ_SUSP_DT', 'RCMD_PROJ_SUSP_DT']], how='left', left_on=['BAN'], right_on=['BAN'])

df_base_pr.sort_values(by=['BAN','DLNQ_DT_PACIFIC', 'INIT_REQ_DTTM'],inplace=True)

df_base_pr.reset_index(drop=True, inplace=True)

# COMMAND ----------

df_base_pr[df_base_pr['BAN']=='101586080']

# COMMAND ----------

df_base_pr[df_base_pr['BAN']=='100057601']

# COMMAND ----------

# MAGIC %md
# MAGIC #### PAR file seems to be in the same lag day as strata, so add 1 day to DLNQ_DT_PACIFIC in filter

# COMMAND ----------

df_base_pr['INIT_REQ_DTTM'] = df_base_pr.apply(lambda x: pd.NaT if x.CAPTURE_DT > x['DLNQ_DT_PACIFIC']+timedelta(days=int(1)) else x.INIT_REQ_DTTM, axis=1)
df_base_pr['PAR_RCMND_DTTM'] = df_base_pr.apply(lambda x: pd.NaT if x.CAPTURE_DT > x['DLNQ_DT_PACIFIC']+timedelta(days=int(1)) else x.PAR_RCMND_DTTM, axis=1)
df_base_pr['PAR_PROJ_SUSP_DT'] = df_base_pr.apply(lambda x: pd.NaT if x.CAPTURE_DT > x['DLNQ_DT_PACIFIC']+timedelta(days=int(1)) else x.PAR_PROJ_SUSP_DT, axis=1)
df_base_pr['RCMD_PROJ_SUSP_DT'] = df_base_pr.apply(lambda x: pd.NaT if x.CAPTURE_DT > x['DLNQ_DT_PACIFIC']+timedelta(days=int(1)) else x.RCMD_PROJ_SUSP_DT, axis=1)
df_base_pr['CAPTURE_DT'] = df_base_pr.apply(lambda x: pd.NaT if x.CAPTURE_DT > x['DLNQ_DT_PACIFIC']+timedelta(days=int(1)) else x.CAPTURE_DT, axis=1)

# COMMAND ----------

df_base_pr[df_base_pr['BAN']=='101586080']

# COMMAND ----------

df_base_pr[df_base_pr['BAN']=='100057601']

# COMMAND ----------

df_base_pr.shape

# COMMAND ----------

df_base_pr = df_base_pr[(df_base_pr['CAPTURE_DT'] <= df_base_pr['DLNQ_DT_PACIFIC']+timedelta(days=int(1))) | (df_base_pr['CAPTURE_DT'].isna())]

# COMMAND ----------

df_base_pr.head()

# COMMAND ----------

df_base_pr[df_base_pr['BAN']=='100057601']

# COMMAND ----------

df_base_pr[df_base_pr['BAN']=='100184538']

# COMMAND ----------

# MAGIC %md
# MAGIC #### Select last pyarr movement (first obs PAR) to drop duplicates

# COMMAND ----------

print(df_base_pr.shape)
print(df_base.shape)

# COMMAND ----------

dup = df_base_pr[df_base_pr.duplicated(['BAN', 'DLNQ_DT_PACIFIC', 'RISK_CD'])]
dup.head()
len(dup)

# COMMAND ----------

17066-13014

# COMMAND ----------

df_base_pr = df_base_pr.sort_values(['BAN','DLNQ_DT_PACIFIC', 'INIT_REQ_DTTM'], ascending=[False, False, False])
df_base_pr = df_base_pr.reset_index(drop=True)

# COMMAND ----------

df_base_pr[df_base_pr['BAN']=='100184538']

# COMMAND ----------

df_base_pr[df_base_pr['BAN']=='101586080']

# COMMAND ----------

df_base_pr[df_base_pr['BAN']=='100057601']

# COMMAND ----------

df_base_pr = df_base_pr.drop_duplicates(['BAN', 'DLNQ_DT_PACIFIC', 'RISK_CD'], keep='first')

# COMMAND ----------

df_base_pr[df_base_pr['BAN']=='100184538']

# COMMAND ----------

print(df_base_pr.shape)
print(df_base.shape)

# COMMAND ----------

print(sorted(set(df_base.BAN.unique().tolist()).difference(set(df_base_pr.BAN.unique().tolist())), key=str.lower))

# COMMAND ----------

df_base_pr[df_base_pr['BAN']=='101586080']

# COMMAND ----------

df_base[df_base['BAN']=='101586080']

# COMMAND ----------

df_PAR[df_PAR['BAN']=='101586080']

# COMMAND ----------

df_base_pr['LG_SUSP_DT'] = df_base_pr[['PROJ_SUSP_DT', 'RCMD_PROJ_SUSP_DT', 'PAR_PROJ_SUSP_DT']].max(axis=1)

# COMMAND ----------

df_base_pr.columns

# COMMAND ----------

df_base_pr = df_base_pr[['BAN', 'RISK_CD', 'INSTR_CD', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR',
                         'LG_SUSP_DT', 'PROJ_SUSP_DT', 'RCMD_PROJ_SUSP_DT', 'PAR_PROJ_SUSP_DT',
                         'INIT_REQ_DTTM', 'PAR_RCMND_DTTM',
                         'CURE_DT_PACIFIC', 'DAYS_TO_CURE', 'WO_DT', 'OCA_DT',
                         'TOT_DUE_AMT', 'TOT_DLNQ_AMT', 'T1', 'T2', 'T3', 'T4','T5']]

df_base_pr = df_base_pr.reset_index(drop = True)

df_base_pr.head()

# COMMAND ----------

df_base_pr.shape

# COMMAND ----------

df_base_pr.columns

# COMMAND ----------

df_base_pr['LST_EVENT_DT']=np.where((df_base_pr['CURE_DT_PACIFIC'].isna() & df_base_pr['WO_DT'].isna()),
                                  df_base_pr['OCA_DT'], 
                                    np.where((df_base_pr['CURE_DT_PACIFIC'].isna() & df_base_pr['OCA_DT'].isna()),
                                      df_base_pr['WO_DT'], 
                                        df_base_pr['CURE_DT_PACIFIC']))

# COMMAND ----------

df_base_pr["DAYS_TO_EVENT"] = (df_base_pr["LST_EVENT_DT"]-df_base_pr["DLNQ_DT_PACIFIC"]).dt.days

# COMMAND ----------

df_base_pr = df_base_pr[['BAN', 'RISK_CD', 'INSTR_CD', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR', 'CURE_DT_PACIFIC', 'WO_DT',
               'OCA_DT', 'LST_EVENT_DT', 'DAYS_TO_EVENT', 'LG_SUSP_DT',
               'PROJ_SUSP_DT', 'RCMD_PROJ_SUSP_DT', 'PAR_PROJ_SUSP_DT',
               'INIT_REQ_DTTM', 'PAR_RCMND_DTTM', 'TOT_DUE_AMT', 'TOT_DLNQ_AMT', 'T1', 'T2', 'T3', 'T4', 'T5']]

# COMMAND ----------

df_base_pr.head()

# COMMAND ----------

df_base_pr.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge treatment data with cycles

# COMMAND ----------

len(df_base_pr.BAN.unique())

# COMMAND ----------

df_treatment_tz_step1 = df_treatment_tz[['BAN', 'TREAT_DT_PACIFIC', 'TRANS_TYPE', 'TRANS_SUB_TYPE', 'TREAT_MSG_CD']]
df_treatment_tz_step1['TRANS_MONYR']=pd.to_datetime(df_treatment_tz_step1['TREAT_DT_PACIFIC']).dt.strftime('%Y%m')
df_treatment_tz_step1.head()                           

# COMMAND ----------

df = pd.merge(df_base_pr, df_treatment_tz_step1, how='left', on=['BAN'])

df.sort_values(by=['BAN','DLNQ_DT_PACIFIC'],inplace=True)

df.reset_index(drop=True, inplace=True)

# COMMAND ----------

len(df.BAN.unique())

# COMMAND ----------

df[df["BAN"] == '108648433']

# COMMAND ----------

df[df.TRANS_TYPE.notnull()].shape

# COMMAND ----------

df[df.TRANS_TYPE.isnull()].shape

# COMMAND ----------

df.shape

# COMMAND ----------

df.TRANS_TYPE = df.TRANS_TYPE.fillna('NOT FOUND')
df.TRANS_MONYR = df.TRANS_MONYR.fillna(999999)

# COMMAND ----------

df['MIN_EVENT_SUSP_DT'] = df.apply(lambda x: min(x.LG_SUSP_DT, x.LST_EVENT_DT), axis=1)

# COMMAND ----------

df['TRANS_TYPE'] = df.apply(lambda x: 'NONE' if (((x.TREAT_DT_PACIFIC < x.DLNQ_DT_PACIFIC) | (x.TREAT_DT_PACIFIC > x.MIN_EVENT_SUSP_DT)) & (x.TRANS_TYPE != 'NOT FOUND')) else x.TRANS_TYPE, axis=1)
df['TRANS_SUB_TYPE'] = df.apply(lambda x: 'NONE' if (((x.TREAT_DT_PACIFIC < x.DLNQ_DT_PACIFIC) | (x.TREAT_DT_PACIFIC > x.MIN_EVENT_SUSP_DT)) & (x.TRANS_TYPE != 'NOT FOUND')) else x.TRANS_SUB_TYPE, axis=1)
df['TREAT_MSG_CD'] = df.apply(lambda x:'NONE'if (((x.TREAT_DT_PACIFIC < x.DLNQ_DT_PACIFIC) | (x.TREAT_DT_PACIFIC > x.MIN_EVENT_SUSP_DT)) & (x.TRANS_TYPE != 'NOT FOUND')) else x.TREAT_MSG_CD, axis=1)
df['TRANS_MONYR'] = df.apply(lambda x: 999999 if (((x.TREAT_DT_PACIFIC < x.DLNQ_DT_PACIFIC) | (x.TREAT_DT_PACIFIC > x.MIN_EVENT_SUSP_DT)) & (x.TRANS_TYPE != 'NOT FOUND')) else x.TRANS_MONYR, axis=1)
df['TREAT_DT_PACIFIC'] = df.apply(lambda x: pd.NaT if (((x.TREAT_DT_PACIFIC < x.DLNQ_DT_PACIFIC) | (x.TREAT_DT_PACIFIC > x.MIN_EVENT_SUSP_DT)) & (x.TRANS_TYPE != 'NOT FOUND')) else x.TREAT_DT_PACIFIC, axis=1)

# COMMAND ----------

df[df["BAN"] == '108648433']

# COMMAND ----------

df[df["BAN"] == '101197732']

# COMMAND ----------

df[df["BAN"] == '307181897']

# COMMAND ----------

df[df["BAN"] == '146202570']

# COMMAND ----------

len(df.BAN.unique())

# COMMAND ----------

print(sorted(set(df_base_pr.BAN.unique().tolist()).difference(set(df.BAN.unique().tolist())), key=str.lower))

# COMMAND ----------

df[df["BAN"] == '108648433']

# COMMAND ----------

df_base_pr[df_base_pr["BAN"] == '108648433']

# COMMAND ----------

len(df.BAN.unique())

# COMMAND ----------

df.shape

# COMMAND ----------

dup = df[df.duplicated(['BAN', 'DLNQ_DT_PACIFIC', 'RISK_CD',
                        'TREAT_DT_PACIFIC', 'TRANS_TYPE', 'TRANS_SUB_TYPE', 'TREAT_MSG_CD', 'TRANS_MONYR'])]

print(len(dup))

dup.head()

# COMMAND ----------

dup[dup["BAN"] == '100057601']

# COMMAND ----------

df[df["BAN"] == '100057601']

# COMMAND ----------

df = df.sort_values(['BAN','DLNQ_DT_PACIFIC', 'RISK_CD', 'TREAT_DT_PACIFIC'])
df = df.reset_index(drop=True)

# COMMAND ----------

df[df["BAN"] == '100057601']

# COMMAND ----------

df = df.drop_duplicates(['BAN', 'DLNQ_DT_PACIFIC', 'RISK_CD',
                        'TREAT_DT_PACIFIC', 'TRANS_TYPE', 'TRANS_SUB_TYPE', 'TREAT_MSG_CD', 'TRANS_MONYR'], keep='first')

# COMMAND ----------

df[df["BAN"] == '100057601']

# COMMAND ----------

df[df["BAN"] == '146202570']

# COMMAND ----------

df[df["BAN"] == '307181897']

# COMMAND ----------

df[df["BAN"] == '101197732']

# COMMAND ----------

df[df["BAN"] == '108648433']

# COMMAND ----------

df["TIMES"] = df.groupby('BAN')['BAN'].transform('count')

# COMMAND ----------

df.shape

# COMMAND ----------

df.dtypes

# COMMAND ----------

df2 = df[((df['TIMES'] == 1) & (df['TRANS_MONYR'] == 999999)) |
        ((df['TIMES'] == 1) & ((df['TRANS_MONYR'] == '201908') | (df['TRANS_MONYR'] == '202008'))) |
        ((df['TIMES'] > 1) & (df['TRANS_MONYR'] != 999999))]

# COMMAND ----------

df2.shape

# COMMAND ----------

df2.reset_index(drop=True, inplace=True)

# COMMAND ----------

df2[df2["BAN"] == '100057601']

# COMMAND ----------

df2[df2["BAN"] == '146202570']

# COMMAND ----------

df2[df2["BAN"] == '307181897']

# COMMAND ----------

df2[df2["BAN"] == '108648433']

# COMMAND ----------

len(df2.BAN.unique())

# COMMAND ----------

df2.columns

# COMMAND ----------

df2.info()

# COMMAND ----------

df2 = df2[['BAN', 'RISK_CD', 'INSTR_CD', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR', 'CURE_DT_PACIFIC',
       'WO_DT', 'OCA_DT', 'LST_EVENT_DT', 'DAYS_TO_EVENT', 'LG_SUSP_DT',
       'PROJ_SUSP_DT', 'RCMD_PROJ_SUSP_DT', 'PAR_PROJ_SUSP_DT',
       'INIT_REQ_DTTM', 'PAR_RCMND_DTTM', 'TOT_DUE_AMT', 'TOT_DLNQ_AMT', 'T1',
       'T2', 'T3', 'T4', 'T5', 'TREAT_DT_PACIFIC', 'TRANS_TYPE', 'TRANS_SUB_TYPE',
       'TREAT_MSG_CD', 'TRANS_MONYR']]

# COMMAND ----------

df2.head()

# COMMAND ----------

df2.isna().sum()

# COMMAND ----------

df2.dtypes

# COMMAND ----------

df2["BAN"] = df2["BAN"].astype('str')
df2["RISK_CD"] = df2["RISK_CD"].astype('str')
df2["TRANS_MONYR"] = df2["TRANS_MONYR"].astype('str')

# COMMAND ----------

#df2.to_csv("df2_6_30_v2.csv")

# COMMAND ----------

df2.shape

# COMMAND ----------

#df2 = pd.read_csv('df2_6_24.csv', index_col=0)

# COMMAND ----------

#len(df2.BAN.unique())

# COMMAND ----------

'''df2["BAN"] = df2["BAN"].astype('str')
df2["RISK_CD"] = df2["RISK_CD"].astype('str')
df2["TRANS_MONYR"] = df2["TRANS_MONYR"].astype('str')
df2["INSTR_CD"] = df2["INSTR_CD"].astype('str')
df2["DLNQ_DT"] = df2['DLNQ_DT'].astype('datetime64[D]')
df2["DLNQ_DT_STR"] = df2['DLNQ_DT_STR'].astype('datetime64[D]')
df2['CURE_DT'] = pd.to_datetime(df2['CURE_DT'])
df2["WO_DT"] = df2['WO_DT'].astype('datetime64[D]')
df2["OCA_DT"] = df2['OCA_DT'].astype('datetime64[D]')
df2["LG_SUSP_DT"] = df2['LG_SUSP_DT'].astype('datetime64[D]')
df2["PROJ_SUSP_DT"] = df2['PROJ_SUSP_DT'].astype('datetime64[D]')
df2["RCMD_PROJ_SUSP_DT"] = df2['RCMD_PROJ_SUSP_DT'].astype('datetime64[D]')
df2["PAR_PROJ_SUSP_DT"] = df2['PAR_PROJ_SUSP_DT'].astype('datetime64[D]')
df2["PROJ_SUSP_DT"] = df2['PROJ_SUSP_DT'].astype('datetime64[D]')
df2["TRANS_DT"] = df2['TRANS_DT'].astype('datetime64[D]')
df2["LST_EVENT_DT"] = df2['LST_EVENT_DT'].astype('datetime64[D]')
df2['INIT_REQ_DTTM'] = pd.to_datetime(df2['INIT_REQ_DTTM'], errors = 'coerce')
df2['PAR_RCMND_DTTM'] = pd.to_datetime(df2['PAR_RCMND_DTTM'], errors = 'coerce')
'''

# COMMAND ----------

pd.datetime.max.date()

# COMMAND ----------

pd.Timestamp.max.date()

# COMMAND ----------

df2['CURE_DT_PACIFIC'] = df2['CURE_DT_PACIFIC'].fillna(pd.Timestamp.max.date())

df2['CURE_DT_PACIFIC'] = pd.to_datetime(df2['CURE_DT_PACIFIC']).dt.date

# COMMAND ----------

df2.head()

# COMMAND ----------

df2.dtypes

# COMMAND ----------

df2 = df2.reset_index(drop=True)

# COMMAND ----------

final_df = create_new_df(df2)

final_df.head()

# COMMAND ----------

final_df.shape

# COMMAND ----------

final_df.info()

# COMMAND ----------

len(final_df.BAN.unique())

# COMMAND ----------

print(sorted(set(df2.BAN.unique().tolist()).difference(set(final_df.BAN.unique().tolist())), key=str.lower))

# COMMAND ----------

df2[df2["BAN"] == '111507517']

# COMMAND ----------

final_df[final_df["BAN"] == '111507517']

# COMMAND ----------

final_df.dtypes

# COMMAND ----------

final_df['TRANS_DT'] = final_df['TRANS_DT'].astype('datetime64[D]')
final_df['CURE_DT_PACIFIC'] = final_df['CURE_DT_PACIFIC'].astype('datetime64[D]')

final_df['DAYS_TO_EVENT'] = final_df.apply(lambda x: (min(x['LST_EVENT_DT'], x['LG_SUSP_DT'])-x['TRANS_DT']).days, axis=1)

# COMMAND ----------

final_df.columns

# COMMAND ----------

final_df.shape

# COMMAND ----------

final_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Make a call to function (fit_prev_data) 
# MAGIC    Merge the data of the new model with the old one thereby retaining the TRANS_TYPE and TRANS_SUB_TYPE [merge indicator = both] 

# COMMAND ----------

df2.head()

# COMMAND ----------

df_cut = df2.copy()
df_cut = df_cut[['BAN', 'DLNQ_DT_PACIFIC', 'TREAT_DT_PACIFIC', 'TRANS_TYPE', 'TRANS_SUB_TYPE', 'TREAT_MSG_CD']]
df_cut.columns = ['BAN', 'DLNQ_DT_PACIFIC', 'TRANS_DT', 'TRANS_TYPE', 'TRANS_SUB_TYPE', 'TREAT_MSG_CD']

df_for_analysis = fit_prev_data(df_cut, final_df)

df_for_analysis.head(10)

# COMMAND ----------

df_for_analysis.head()

# COMMAND ----------

df_for_analysis[df_for_analysis["BAN"] == '100142994']

# COMMAND ----------

# MAGIC %md
# MAGIC #### FILL 'NA' for features where data is not available

# COMMAND ----------

df_for_analysis['TRANS_TYPE']=df_for_analysis['TRANS_TYPE'].fillna('NONE')
df_for_analysis['TRANS_SUB_TYPE']=df_for_analysis['TRANS_SUB_TYPE'].fillna('NONE')
df_for_analysis['TREAT_MSG_CD']=df_for_analysis['TREAT_MSG_CD'].fillna('NONE')

df_for_analysis[df_for_analysis["BAN"] == '100142994']

# COMMAND ----------

df_analysis = df_for_analysis.copy()

# COMMAND ----------

df_analysis.columns

# COMMAND ----------

intermediate_df = pd.concat([df_analysis, pd.get_dummies(df_analysis[['TRANS_TYPE','TRANS_SUB_TYPE', 'TREAT_MSG_CD']])], 1).groupby(['TRANS_DT',
       'BAN', 'RISK_CD', 'INSTR_CD', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR',
       'CURE_DT_PACIFIC', 'LG_SUSP_DT', 'LST_EVENT_DT', 'DAYS_TO_EVENT', 'TOT_DUE_AMT',
       'TOT_DLNQ_AMT', 'T1', 'T2', 'T3', 'T4', 'T5', 'TRANS_TYPE',
       'TRANS_SUB_TYPE', 'TREAT_MSG_CD']).sum().reset_index()

# COMMAND ----------

intermediate_df.sort_values(['BAN','TRANS_DT'])

intermediate_df[intermediate_df["BAN"] == '100142994']

# COMMAND ----------

intermediate_df.columns

# COMMAND ----------

intermediate_df.reset_index(drop=True, inplace=True)

# COMMAND ----------

intermediate_df.shape

# COMMAND ----------

test=intermediate_df.groupby(['TRANS_DT', 'BAN', 'RISK_CD', 'INSTR_CD', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR',
       'CURE_DT_PACIFIC', 'LG_SUSP_DT', 'LST_EVENT_DT', 'DAYS_TO_EVENT', 'TOT_DUE_AMT',
       'TOT_DLNQ_AMT', 'T1', 'T2', 'T3', 'T4', 'T5', 'TRANS_TYPE',
       'TRANS_SUB_TYPE', 'TREAT_MSG_CD']).sum().reset_index()


# COMMAND ----------

test.reset_index(drop=True, inplace=True)

# COMMAND ----------

test.columns

# COMMAND ----------

test[test["BAN"] == '100142994']

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### For accounts with multiple TREATMENT_TYPE and TREATMENT_SUB_TYPE on the same day, join the treatments. 
# MAGIC      The objective is to create  
# MAGIC          a) UNIQUE date rows per BAN 
# MAGIC          b) RETAIN the information about what multiple TREATMENT TYPE and SUB_TYPES occured
# MAGIC          

# COMMAND ----------

subset_test=test[['TRANS_DT', 'BAN', 'RISK_CD', 'INSTR_CD', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR',
       'CURE_DT_PACIFIC', 'LG_SUSP_DT', 'LST_EVENT_DT', 'DAYS_TO_EVENT', 'TOT_DUE_AMT',
       'TOT_DLNQ_AMT', 'T1', 'T2', 'T3', 'T4', 'T5', 'TRANS_TYPE',
       'TRANS_SUB_TYPE', 'TREAT_MSG_CD']]

right_df = subset_test.groupby(['TRANS_DT', 'BAN', 'RISK_CD', 'INSTR_CD',
                                'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR', 'CURE_DT_PACIFIC', 'LG_SUSP_DT']).agg({'TRANS_TYPE':'|'.join,'TRANS_SUB_TYPE':'|'.join,'TREAT_MSG_CD':'|'.join}).astype('str').reset_index()

print(right_df.columns)
    
left=intermediate_df.copy()

print(left.columns)


# COMMAND ----------

left.drop(['TRANS_TYPE','TRANS_SUB_TYPE', 'TREAT_MSG_CD'],axis=1, inplace=True)

# COMMAND ----------

left_df=left.groupby(['TRANS_DT', 'BAN', 'RISK_CD', 'INSTR_CD', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR',
       'CURE_DT_PACIFIC', 'LG_SUSP_DT', 'LST_EVENT_DT', 'DAYS_TO_EVENT', 'TOT_DUE_AMT',
       'TOT_DLNQ_AMT', 'T1', 'T2', 'T3', 'T4', 'T5']).sum().reset_index()

# COMMAND ----------

left_df = left_df.sort_values(['BAN','DLNQ_DT_PACIFIC'])

# COMMAND ----------

right_df.sort_values(by=['BAN', 'DLNQ_DT_PACIFIC'],inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Final DF

# COMMAND ----------

merged_df = pd.merge(left_df,
                    right_df,
                    how='inner',
                    on=['TRANS_DT', 'BAN', 'RISK_CD', 'INSTR_CD', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR', 'CURE_DT_PACIFIC', 'LG_SUSP_DT'])

merged_df = merged_df.sort_values(['BAN','DLNQ_DT_PACIFIC'])

# COMMAND ----------

merged_df[merged_df["BAN"] == '100142994']

# COMMAND ----------

merged_df[merged_df["BAN"] == '100057601']

# COMMAND ----------

merged_df.shape

# COMMAND ----------

final_df.shape

# COMMAND ----------

len(merged_df.BAN.unique())

# COMMAND ----------

dup = merged_df[merged_df.duplicated(['BAN', 'TRANS_DT', 'DLNQ_DT_PACIFIC', 'RISK_CD'])]

print(len(dup))

dup

# COMMAND ----------

merged_df.shape

# COMMAND ----------

merged_df.head()

# COMMAND ----------

#merged_df.to_csv("Enb_merged_df_6_30_21_v2.csv")

# COMMAND ----------

#merged_df = pd.read_csv('Enb_merged_df_6_25_21.csv', index_col=0)

# COMMAND ----------

print(set(df2.columns.tolist()).difference(set(merged_df.columns.tolist())))

# COMMAND ----------

df2[df2["BAN"] == '100142994']

# COMMAND ----------

merged_df2 = pd.merge(merged_df,
                     df2[['BAN', 'RISK_CD', 'DLNQ_DT_PACIFIC', 'WO_DT', 'OCA_DT', 'RCMD_PROJ_SUSP_DT',
                          'PAR_RCMND_DTTM', 'PAR_PROJ_SUSP_DT', 'INIT_REQ_DTTM', 'PROJ_SUSP_DT', 'TRANS_MONYR']],
                     how='left',
                     left_on=['BAN', 'RISK_CD', 'DLNQ_DT_PACIFIC'],
                     right_on=['BAN', 'RISK_CD', 'DLNQ_DT_PACIFIC'])

# COMMAND ----------

merged_df2.shape

# COMMAND ----------

merged_df2.columns

# COMMAND ----------

merged_df2 = merged_df2[['TRANS_DT', 'BAN', 'RISK_CD', 'INSTR_CD', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR',
       'CURE_DT_PACIFIC', 'WO_DT', 'OCA_DT',  'LST_EVENT_DT', 'LG_SUSP_DT', 'RCMD_PROJ_SUSP_DT', 
       'PAR_PROJ_SUSP_DT', 'PROJ_SUSP_DT', 'INIT_REQ_DTTM', 'PAR_RCMND_DTTM', 
       'DAYS_TO_EVENT', 'TOT_DUE_AMT',
       'TOT_DLNQ_AMT', 'T1', 'T2', 'T3', 'T4', 'T5',
       'TRANS_TYPE', 'TRANS_SUB_TYPE', 'TREAT_MSG_CD',
       'TRANS_TYPE_AD', 'TRANS_TYPE_EMAIL', 'TRANS_TYPE_LTR',
       'TRANS_TYPE_NONE', 'TRANS_TYPE_OB', 'TRANS_TYPE_SMS',
       'TRANS_TYPE_TVMSG', 'TRANS_SUB_TYPE_AOB ', 'TRANS_SUB_TYPE_CARE',
       'TRANS_SUB_TYPE_CLTN', 'TRANS_SUB_TYPE_FINAL', 'TRANS_SUB_TYPE_HCI ',
       'TRANS_SUB_TYPE_NONE', 'TRANS_SUB_TYPE_PRE-CNCL',
       'TRANS_SUB_TYPE_PRE-MNSV', 'TRANS_SUB_TYPE_PRE-SUSP',
       'TRANS_SUB_TYPE_REMINDER', 'TRANS_SUB_TYPE_THANKYOU',
       'TREAT_MSG_CD_1PDTX', 'TREAT_MSG_CD_1PDTXT', 'TREAT_MSG_CD_AD_CALL',
       'TREAT_MSG_CD_CHAMPPRE', 'TREAT_MSG_CD_EML1', 'TREAT_MSG_CD_EMLLF',
       'TREAT_MSG_CD_EMLLFS', 'TREAT_MSG_CD_EMLPDL', 'TREAT_MSG_CD_EMLPR',
       'TREAT_MSG_CD_EMLPRESN', 'TREAT_MSG_CD_EMLPRS', 'TREAT_MSG_CD_EMLPSN1',
       'TREAT_MSG_CD_EMLPSN2', 'TREAT_MSG_CD_EMLRE', 'TREAT_MSG_CD_EMLSN',
       'TREAT_MSG_CD_EMLTY', 'TREAT_MSG_CD_FBLTR1', 'TREAT_MSG_CD_NONE',
       'TREAT_MSG_CD_OUTBOUNDCALL', 'TREAT_MSG_CD_PDL', 'TREAT_MSG_CD_PDLHSI',
       'TREAT_MSG_CD_PTPSN1', 'TREAT_MSG_CD_PTPSN2', 'TREAT_MSG_CD_QG1SUSP',
       'TREAT_MSG_CD_QG2SUSP', 'TREAT_MSG_CD_QGASUSP', 'TREAT_MSG_CD_QGBSUSP',
       'TREAT_MSG_CD_QGDSUSP', 'TREAT_MSG_CD_QS1SUSP', 'TREAT_MSG_CD_QS2SUSP',
       'TREAT_MSG_CD_QSASUSP', 'TREAT_MSG_CD_SMSPD', 'TREAT_MSG_CD_SMSPRESN',
       'TREAT_MSG_CD_UDASMSG1', 'TREAT_MSG_CD_UDASMSG2', 'TREAT_MSG_CD_VSN']]

# COMMAND ----------

merged_df2[merged_df2["BAN"] == '100142994']

# COMMAND ----------

dup = merged_df2[merged_df2.duplicated(['BAN', 'TRANS_DT', 'DLNQ_DT_PACIFIC', 'RISK_CD'])]

print(len(dup))

#dup

# COMMAND ----------

merged_df2 = merged_df2.drop_duplicates(['BAN', 'TRANS_DT', 'DLNQ_DT_PACIFIC', 'RISK_CD'], keep='first')

# COMMAND ----------

merged_df2.shape

# COMMAND ----------

merged_df.shape

# COMMAND ----------

merged_df2[merged_df2["BAN"] == '100142994']

# COMMAND ----------

#merged_df2.to_csv("Enb_merged_df2_6_30_21_v2.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Timestamp Limits by Day

# COMMAND ----------

print(merged_df2.columns.tolist())

# COMMAND ----------

merged_df2_cut = merged_df2.copy()

# COMMAND ----------

merged_df2_cut = merged_df2_cut[['TRANS_DT', 'BAN', 'RISK_CD', 'INSTR_CD', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR', 'CURE_DT_PACIFIC',
                                 'WO_DT', 'OCA_DT', 'LST_EVENT_DT', 'LG_SUSP_DT', 'RCMD_PROJ_SUSP_DT',
                                 'PAR_PROJ_SUSP_DT', 'PROJ_SUSP_DT', 'INIT_REQ_DTTM', 'PAR_RCMND_DTTM',
                                 'DAYS_TO_EVENT', 'TOT_DUE_AMT', 'TOT_DLNQ_AMT', 'T1', 'T2', 'T3', 'T4', 'T5',
                                 'TRANS_TYPE', 'TRANS_SUB_TYPE', 'TREAT_MSG_CD']]

# COMMAND ----------

merged_df2_cut.dtypes

# COMMAND ----------

merged_df2_cut[merged_df2_cut["BAN"] == '100142994']

# COMMAND ----------

# MAGIC %md
# MAGIC #### Merge curret final df with treatments times 

# COMMAND ----------

df_treatment_tz_first.head()

# COMMAND ----------

df_treatment_tz_first.columns

# COMMAND ----------

merged_df2_cut_tm = pd.merge(merged_df2_cut,
                             df_treatment_tz_first[['BAN', 'TRANS_TYPE_TIMECUTOFF',
                                                    'TRANS_SUB_TYPE_TIMECUTOFF', 'TREAT_MSG_CD_TIMECUTOFF',
                                                    'TREAT_DT_PACIFIC_CUTOFF', 'TREAT_DTTM_PACIFIC_CUTOFF']],
                             how='left',
                             left_on=['BAN', 'TRANS_DT'],
                             right_on=['BAN', 'TREAT_DT_PACIFIC_CUTOFF'])

# COMMAND ----------

merged_df2_cut_tm.head()

# COMMAND ----------

merged_df2_cut_tm.shape

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create new start and end transaction time

# COMMAND ----------

tm_trans = time(0, 0, 0)

merged_df2_cut_tm['TRANS_DT']= pd.to_datetime(merged_df2_cut_tm['TRANS_DT'], format="%m/%d/%Y %H:%M:%S")
merged_df2_cut_tm['TRANS_DT_FAKE'] = merged_df2_cut_tm.apply(lambda x: datetime.combine(x.TRANS_DT, tm_trans), axis=1)
merged_df2_cut_tm['TRANS_DT_FAKE'] = merged_df2_cut_tm.apply(lambda x: pacific_tz.localize(x['TRANS_DT_FAKE'], is_dst=True), axis=1)

# COMMAND ----------

merged_df2_cut_tm['TRANS_DTTM_PACIFIC_END']=np.where(merged_df2_cut_tm['TREAT_DTTM_PACIFIC_CUTOFF'].isna(),
                                                      merged_df2_cut_tm['TRANS_DT_FAKE'], 
                                                      merged_df2_cut_tm['TREAT_DTTM_PACIFIC_CUTOFF'])

# COMMAND ----------

merged_df2_cut_tm['TRANS_DTTM_PACIFIC_END']= pd.to_datetime(merged_df2_cut_tm['TRANS_DTTM_PACIFIC_END'], format="%m/%d/%Y %H:%M:%S")

# COMMAND ----------

# comment it out by Wen, since it errored out
#merged_df2_cut_tm['TRANS_DTTM_PACIFIC_END'] = merged_df2_cut_tm.apply(lambda x: pacific_tz.localize(x['TRANS_DTTM_PACIFIC_END'], is_dst=True), axis=1)

# COMMAND ----------

merged_df2_cut_tm['TRANS_DTTM_PACIFIC_END'] = merged_df2_cut_tm.apply(lambda x: x['TRANS_DTTM_PACIFIC_END']-timedelta(hours=7), axis=1)

# COMMAND ----------

merged_df2_cut_tm['TRANS_MONYR']=pd.to_datetime(merged_df2_cut_tm['DLNQ_DT_PACIFIC']).dt.strftime('%Y%m')
merged_df2_cut_tm['TRANS_MONYR'] = merged_df2_cut_tm['TRANS_MONYR'].astype(int)

# COMMAND ----------

merged_df2_cut_tm['TRANS_DTTM_PACIFIC_START'] = merged_df2_cut_tm.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])['TRANS_DTTM_PACIFIC_END'].shift(1)

# COMMAND ----------

# comment out last line by Wen, since it errored out, need to check with Daan on this part
merged_df2_cut_tm['TRANS_DTTM_PACIFIC_START']=np.where(merged_df2_cut_tm['TRANS_DTTM_PACIFIC_START'].isna(),
                                                      merged_df2_cut_tm['TRANS_DT_FAKE'], 
                                                      merged_df2_cut_tm['TRANS_DTTM_PACIFIC_START'])

merged_df2_cut_tm['TRANS_DTTM_PACIFIC_START']= pd.to_datetime(merged_df2_cut_tm['TRANS_DTTM_PACIFIC_START'], format="%m/%d/%Y %H:%M:%S")
merged_df2_cut_tm['TRANS_DTTM_PACIFIC_START']= merged_df2_cut_tm.apply(lambda x: pacific_tz.localize(x['TRANS_DTTM_PACIFIC_START'], is_dst=True), axis=1)

# COMMAND ----------

merged_df2_cut_tm['TRANS_DTTM_PACIFIC_START'] = merged_df2_cut_tm.apply(lambda x: x['TRANS_DTTM_PACIFIC_START']-timedelta(hours=7), axis=1)

# COMMAND ----------

merged_df2_cut_tm.columns

# COMMAND ----------

merged_df2_cut_tm = merged_df2_cut_tm[['TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END', 'TRANS_DT', 'BAN',
                                    'RISK_CD', 'INSTR_CD',
                                   'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR', 'CURE_DT_PACIFIC', 'WO_DT', 'OCA_DT',
                                    'LST_EVENT_DT', 'LG_SUSP_DT',
                                    'RCMD_PROJ_SUSP_DT', 'PAR_PROJ_SUSP_DT', 'PROJ_SUSP_DT',
                                   'INIT_REQ_DTTM', 'PAR_RCMND_DTTM', 'DAYS_TO_EVENT',
                                    'TOT_DUE_AMT','TOT_DLNQ_AMT', 'T1', 'T2', 'T3', 'T4', 'T5', 
                                   'TRANS_TYPE', 'TRANS_SUB_TYPE', 'TREAT_MSG_CD', 
                                      'TREAT_MSG_CD_TIMECUTOFF',
                                       'TREAT_DT_PACIFIC_CUTOFF', 'TREAT_DTTM_PACIFIC_CUTOFF', 'TRANS_DT_FAKE']]

# COMMAND ----------

merged_df2_cut_tm['DAYS_IN_DLNQ']=(merged_df2_cut_tm['TRANS_DT'] - merged_df2_cut_tm['DLNQ_DT_PACIFIC']).dt.days

# COMMAND ----------

# MAGIC %md
# MAGIC #### Substract 24hrs to the first start datetime of each dlnq cyle

# COMMAND ----------

merged_df2_cut_tm['TRANS_DTTM_PACIFIC_START'] = merged_df2_cut_tm.apply(lambda x: x['TRANS_DTTM_PACIFIC_END']-timedelta(hours=24) if 
                                                                        x['DAYS_IN_DLNQ']==0
                                                                        else x['TRANS_DTTM_PACIFIC_START'], axis=1)

# COMMAND ----------

merged_df2_cut_tm.TREAT_MSG_CD_TIMECUTOFF = merged_df2_cut_tm.TREAT_MSG_CD_TIMECUTOFF.fillna('NONE')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Add 23hrs 59 min and 59 secs to the last end datetime of each dlnq cyle if got cured

# COMMAND ----------

merged_df2_cut_tm['TRANS_DTTM_PACIFIC_END'] = merged_df2_cut_tm.apply(lambda x: (x['TRANS_DTTM_PACIFIC_END']+timedelta(hours=23, minutes=59, seconds=59)) if 
                                                                        ((x['TREAT_MSG_CD_TIMECUTOFF']=='NONE') & (x['DAYS_TO_EVENT']==0))
                                                                        else x['TRANS_DTTM_PACIFIC_END'], axis=1)

# COMMAND ----------

merged_df2_cut_tm.head(50)

# COMMAND ----------

merged_df2_cut_tm.dtypes

# COMMAND ----------

#merged_df2_cut_tm.to_csv("merged_df2_cut_tm_06_30_21_v2.csv")

# COMMAND ----------


'''
merged_df2_cut_tm = pd.read_csv('merged_df2_cut_tm_06_30_21.csv', index_col=0) 

merged_df2_cut_tm["BAN"] = merged_df2_cut_tm["BAN"].astype('str')
merged_df2_cut_tm["RISK_CD"] = merged_df2_cut_tm["RISK_CD"].astype('str')
#merged_df2_cut_tm["TRANS_MONYR"] = merged_df2_cut_tm["TRANS_MONYR"].astype('str')
merged_df2_cut_tm["INSTR_CD"] = merged_df2_cut_tm["INSTR_CD"].astype('str')
merged_df2_cut_tm["DLNQ_DT_PACIFIC"] = merged_df2_cut_tm['DLNQ_DT_PACIFIC'].astype('datetime64[D]')
merged_df2_cut_tm['CURE_DT_PACIFIC'] = pd.to_datetime(merged_df2_cut_tm['CURE_DT_PACIFIC'])
merged_df2_cut_tm["WO_DT"] = merged_df2_cut_tm['WO_DT'].astype('datetime64[D]')
merged_df2_cut_tm["OCA_DT"] = merged_df2_cut_tm['OCA_DT'].astype('datetime64[D]')
merged_df2_cut_tm["LG_SUSP_DT"] = merged_df2_cut_tm['LG_SUSP_DT'].astype('datetime64[D]')
merged_df2_cut_tm["PROJ_SUSP_DT"] = merged_df2_cut_tm['PROJ_SUSP_DT'].astype('datetime64[D]')
merged_df2_cut_tm["RCMD_PROJ_SUSP_DT"] = merged_df2_cut_tm['RCMD_PROJ_SUSP_DT'].astype('datetime64[D]')
merged_df2_cut_tm["PAR_PROJ_SUSP_DT"] = merged_df2_cut_tm['PAR_PROJ_SUSP_DT'].astype('datetime64[D]')
merged_df2_cut_tm["PROJ_SUSP_DT"] = merged_df2_cut_tm['PROJ_SUSP_DT'].astype('datetime64[D]')
merged_df2_cut_tm["TRANS_DT"] = merged_df2_cut_tm['TRANS_DT'].astype('datetime64[D]')
merged_df2_cut_tm["LST_EVENT_DT"] = merged_df2_cut_tm['LST_EVENT_DT'].astype('datetime64[D]')
merged_df2_cut_tm["TRANS_DT_FAKE"] = merged_df2_cut_tm['TRANS_DT_FAKE'].astype('datetime64[D]')

merged_df2_cut_tm['INIT_REQ_DTTM'] = pd.to_datetime(merged_df2_cut_tm['INIT_REQ_DTTM'], errors = 'coerce')
merged_df2_cut_tm['PAR_RCMND_DTTM'] = pd.to_datetime(merged_df2_cut_tm['PAR_RCMND_DTTM'], errors = 'coerce')
merged_df2_cut_tm['TRANS_DTTM_PACIFIC_START'] = pd.to_datetime(merged_df2_cut_tm['TRANS_DTTM_PACIFIC_START'], errors = 'coerce')
merged_df2_cut_tm['TRANS_DTTM_PACIFIC_END'] = pd.to_datetime(merged_df2_cut_tm['TRANS_DTTM_PACIFIC_END'], errors = 'coerce')
merged_df2_cut_tm['TREAT_DTTM_PACIFIC_CUTOFF'] = pd.to_datetime(merged_df2_cut_tm['TREAT_DTTM_PACIFIC_CUTOFF'], errors = 'coerce')

#merged_df2_cut_tm['TRANS_DTTM_PACIFIC_START'] = merged_df2_cut_tm.apply(lambda x: pacific_tz.localize(x.TRANS_DTTM_PACIFIC_START, is_dst=True), axis = 1) 
#merged_df2_cut_tm['TRANS_DTTM_PACIFIC_END'] = merged_df2_cut_tm.apply(lambda x: pacific_tz.localize(x.TRANS_DTTM_PACIFIC_END, is_dst=True), axis = 1)
#merged_df2_cut_tm['TREAT_DTTM_PACIFIC_CUTOFF'] = merged_df2_cut_tm.apply(lambda x: pacific_tz.localize(x.TREAT_DTTM_PACIFIC_CUTOFF, is_dst=True), axis = 1)

'''

# COMMAND ----------

merged_df2_cut_tm.shape

# COMMAND ----------

merged_df2_cut_tm.head(50)

# COMMAND ----------

merged_df2_cut_tm.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC # Join Payments

# COMMAND ----------

#AQUI   # comment out by Wen

# COMMAND ----------

df_pmt_timestamp.head(10)

# COMMAND ----------

trans = df_pmt_timestamp.fncl_trans_type_cd.unique().tolist()
trans = trans[1:len(trans)+1]
trans

# COMMAND ----------

df_pmt_timestamp.timezone_cd.unique()

# COMMAND ----------

for i in trans:
    print(str(i))
        
    print(df_pmt_timestamp[df_pmt_timestamp['fncl_trans_type_cd']==str(i)][['fncl_trans_dt_tm']].fillna('F').groupby('fncl_trans_dt_tm').size())

# COMMAND ----------

#%%time

s = df_pmt_timestamp.groupby('fncl_trans_type_cd').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s.pivot_table(index = ['fncl_trans_type_cd'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC df_pmt_tz = df_pmt_timestamp.copy()
# MAGIC 
# MAGIC df_pmt_tz['fncl_trans_dt'] = df_pmt_tz['fncl_trans_dt'].astype('datetime64[D]')
# MAGIC df_pmt_tz.acct_nbr = df_pmt_tz.acct_nbr.astype(str)
# MAGIC df_pmt_tz = df_pmt_tz.sort_values(by = ['acct_nbr', 'fncl_trans_dt'])
# MAGIC df_pmt_tz = df_pmt_tz.reset_index(drop=True)
# MAGIC 
# MAGIC df_pmt_tz = add_timezone(df_pmt_tz, 'timezone_cd', 'fncl_trans_dt_tm')

# COMMAND ----------

df_pmt_tz.columns

# COMMAND ----------

df_pmt_tz.head()

# COMMAND ----------

df_pmt_tz_cut = df_pmt_tz[['acct_nbr', 'fncl_trans_dt', 'fncl_trans_dt_tm',
       'timezone_cd', 'fncl_trans_dt_tm_tz', 'pmt_timezone',
       'fncl_trans_dt_tm_tz_pacific', 'fncl_trans_type_cd', 'fncl_trans_amt',
       'fncl_trans_method_cd', 'fncl_trans_channel_cd']]

df_pmt_tz_cut.columns = ['BAN', 'fncl_trans_dt', 'fncl_trans_dt_tm',
       'timezone_cd', 'fncl_trans_dt_tm_tz', 'pmt_timezone',
       'fncl_trans_dt_tm_tz_pacific', 'fncl_trans_type_cd', 'fncl_trans_amt',
       'fncl_trans_method_cd', 'fncl_trans_channel_cd']  

# COMMAND ----------

df_pmt_tz_cut['fncl_trans_method_cd'] = df_pmt_tz_cut['fncl_trans_method_cd'].replace(np.nan, 'NONE')
df_pmt_tz_cut['fncl_trans_channel_cd'] = df_pmt_tz_cut['fncl_trans_channel_cd'].replace(np.nan, 'NONE')
df_pmt_tz_cut['fncl_trans_method_cd'] = df_pmt_tz_cut['fncl_trans_method_cd'].replace('?', 'NONE')
df_pmt_tz_cut['fncl_trans_channel_cd'] = df_pmt_tz_cut['fncl_trans_channel_cd'].replace('?', 'NONE')
df_pmt_tz_cut['fncl_trans_dt_tm'] = df_pmt_tz_cut['fncl_trans_dt_tm'].replace(np.nan, 'NONE')
df_pmt_tz_cut['fncl_trans_dt_tm'] = df_pmt_tz_cut['fncl_trans_dt_tm'].replace('?', 'NONE')

# COMMAND ----------

dup = df_pmt_tz_cut[df_pmt_tz_cut.duplicated(['BAN', 'fncl_trans_dt', 'fncl_trans_dt_tm',
       'timezone_cd', 'fncl_trans_dt_tm_tz', 'pmt_timezone',
       'fncl_trans_dt_tm_tz_pacific', 'fncl_trans_type_cd', 'fncl_trans_amt',
       'fncl_trans_method_cd', 'fncl_trans_channel_cd'])]

dup.shape

# COMMAND ----------

df_pmt_tz_cut = df_pmt_tz_cut.drop_duplicates(['BAN', 'fncl_trans_dt', 'fncl_trans_dt_tm',
                                               'timezone_cd', 'fncl_trans_dt_tm_tz', 'pmt_timezone',
                                               'fncl_trans_dt_tm_tz_pacific', 'fncl_trans_type_cd', 'fncl_trans_amt',
                                               'fncl_trans_method_cd', 'fncl_trans_channel_cd'], keep='first')

# COMMAND ----------

dup = df_pmt_tz_cut[df_pmt_tz_cut.duplicated(['BAN', 'fncl_trans_dt', 'fncl_trans_dt_tm',
                                               'timezone_cd', 'fncl_trans_dt_tm_tz', 'pmt_timezone',
                                               'fncl_trans_dt_tm_tz_pacific', 'fncl_trans_type_cd', 'fncl_trans_amt',
                                               'fncl_trans_method_cd', 'fncl_trans_channel_cd'])]

dup.shape

# COMMAND ----------

df_pmt_tz_cut[(df_pmt_tz_cut["BAN"] == '100184538')]

# COMMAND ----------

df_pmt_tz_cut[(df_pmt_tz_cut["BAN"] == '100057601') & (df_pmt_tz_cut['fncl_trans_dt_tm_tz_pacific'].dt.month >= 8)
         & (df_pmt_tz_cut['fncl_trans_dt_tm_tz_pacific'].dt.month <= 10) 
              & (df_pmt_tz_cut['fncl_trans_dt_tm_tz_pacific'].dt.year == 2019)]

# COMMAND ----------

df_pmt_tz_cut[(df_pmt_tz_cut["BAN"] == '100142994') & (df_pmt_tz_cut['fncl_trans_dt_tm_tz_pacific'].dt.month >= 8)
         & (df_pmt_tz_cut['fncl_trans_dt_tm_tz_pacific'].dt.month <= 10)
              & (df_pmt_tz_cut['fncl_trans_dt_tm_tz_pacific'].dt.year == 2019)]

# COMMAND ----------

df_pmt_tz_cut[(df_pmt_tz_cut["BAN"] == '100184538') & (df_pmt_tz_cut['fncl_trans_dt_tm_tz_pacific'].dt.month >= 8)
         & (df_pmt_tz_cut['fncl_trans_dt_tm_tz_pacific'].dt.month <= 10)
              & (df_pmt_tz_cut['fncl_trans_dt_tm_tz_pacific'].dt.year == 2019)]

# COMMAND ----------

df_pmt_tz_cut.head()

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc = pd.merge(merged_df2_cut_tm[['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END']],
                                 df_pmt_tz_cut,
                                 how='left', left_on=['BAN'], right_on=['BAN'])

# COMMAND ----------

#df_pmt_tz_cut_dlnqcyc[(df_pmt_tz_cut_dlnqcyc["BAN"] == '100057601')]

# COMMAND ----------

#df_pmt_tz_cut_dlnqcyc[(df_pmt_tz_cut_dlnqcyc["BAN"] == '100142994')]

# COMMAND ----------

#df_pmt_tz_cut_dlnqcyc.dtypes

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc = df_pmt_tz_cut_dlnqcyc[(df_pmt_tz_cut_dlnqcyc['fncl_trans_dt_tm_tz_pacific'] > 
                                               df_pmt_tz_cut_dlnqcyc['TRANS_DTTM_PACIFIC_START']) &
                                              (df_pmt_tz_cut_dlnqcyc['fncl_trans_dt_tm_tz_pacific'] <=
                                               df_pmt_tz_cut_dlnqcyc['TRANS_DTTM_PACIFIC_END'])]

# COMMAND ----------

#df_pmt_tz_cut_dlnqcyc[(df_pmt_tz_cut_dlnqcyc["BAN"] == '100057601')]

# COMMAND ----------

#df_pmt_tz_cut_dlnqcyc[(df_pmt_tz_cut_dlnqcyc["BAN"] == '100142994')] 

# COMMAND ----------

#df_pmt_tz_cut_dlnqcyc[(df_pmt_tz_cut_dlnqcyc["BAN"] == '100184538')]

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc['fncl_trans_type_cd'] = df_pmt_tz_cut_dlnqcyc.apply(lambda x: x.fncl_trans_type_cd 
                                                                          if x.fncl_trans_type_cd in ['BCK', 'ADJ', 'PYM', 'LPC']
                                                                         else 'OTHER', axis=1)

# COMMAND ----------

#df_pmt_tz_cut_dlnqcyc.dtypes

# COMMAND ----------

#df_strata[(df_strata["BAN"] == '100184538')]

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc['BANCOMODIN'] = df_pmt_tz_cut_dlnqcyc['BAN'].astype(str) + ' '+ df_pmt_tz_cut_dlnqcyc['TRANS_DTTM_PACIFIC_START'].astype(str)+ ' '+ df_pmt_tz_cut_dlnqcyc['TRANS_DTTM_PACIFIC_END'].astype(str)

t = df_pmt_tz_cut_dlnqcyc.groupby(['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END', 'fncl_trans_dt_tm_tz_pacific',
                                   'fncl_trans_type_cd', 'fncl_trans_method_cd', 'fncl_trans_channel_cd']).size().reset_index(name='Num_Transactions')

t = t.sort_values(by = ['BAN', 'fncl_trans_dt_tm_tz_pacific'])

t2=t.groupby(['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END', 'fncl_trans_dt_tm_tz_pacific',
                'fncl_trans_type_cd', 'fncl_trans_method_cd', 'fncl_trans_channel_cd'], as_index = False).agg({'Num_Transactions': sum})

t2.Num_Transactions = t2.Num_Transactions.astype(str)

t3 = t2.groupby(['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END', 'fncl_trans_dt_tm_tz_pacific',
                'fncl_trans_method_cd', 'fncl_trans_channel_cd'],
                as_index = False).agg({'fncl_trans_type_cd': ' | '.join, 'Num_Transactions': ' | '.join})

s = df_pmt_tz_cut_dlnqcyc.pivot_table(index=['BANCOMODIN'],
                       columns = 'fncl_trans_type_cd',
                       aggfunc=sum,
                       values='fncl_trans_amt')

s = s.reset_index()

s['BAN'] = s.apply(lambda x: x['BANCOMODIN'].split()[0], axis=1)

s.iloc[10].BANCOMODIN.split()[1]+' '+s.iloc[10].BANCOMODIN.split()[2][0:8]

s['TRANS_DTTM_PACIFIC_START'] = s.apply(lambda x: x['BANCOMODIN'].split()[1]+' '+x['BANCOMODIN'].split()[2][0:8], axis=1)
s['TRANS_DTTM_PACIFIC_START'] = s.apply(lambda x: datetime.strptime(x.TRANS_DTTM_PACIFIC_START, "%Y-%m-%d %H:%M:%S"), axis=1)
s['TRANS_DTTM_PACIFIC_START'] = s.apply(lambda x: pacific_tz.localize(x.TRANS_DTTM_PACIFIC_START, is_dst=True), axis = 1)

s['TRANS_DTTM_PACIFIC_END'] = s.apply(lambda x: x['BANCOMODIN'].split()[3]+' '+x['BANCOMODIN'].split()[4][0:8], axis=1)
s['TRANS_DTTM_PACIFIC_END'] = s.apply(lambda x: datetime.strptime(x.TRANS_DTTM_PACIFIC_END, "%Y-%m-%d %H:%M:%S"), axis=1)
s['TRANS_DTTM_PACIFIC_END'] = s.apply(lambda x: pacific_tz.localize(x.TRANS_DTTM_PACIFIC_END, is_dst=True), axis = 1)

df_pmt_tz_cut_dlnqcyc_sum = pd.merge(s, t3, how='left',
                                     left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                     right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])



# COMMAND ----------

s.head()

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_sum = df_pmt_tz_cut_dlnqcyc_sum.drop(['BANCOMODIN'], axis=1)

df_pmt_tz_cut_dlnqcyc_sum = df_pmt_tz_cut_dlnqcyc_sum[['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END',
                                                       'fncl_trans_dt_tm_tz_pacific', 'fncl_trans_type_cd', 'Num_Transactions',
                                                       'ADJ', 'BCK', 'PYM', 'LPC', 'OTHER', 'fncl_trans_method_cd', 'fncl_trans_channel_cd']]

df_pmt_tz_cut_dlnqcyc_sum.columns = ['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END', 'fncl_trans_dt_tm_tz_pacific', 
                                     'TRANSACTION_TYPE', 'NUM_TRANSACTIONS', 'ADJ_AMOUNT', 'BCK_AMOUNT', 'PYM_AMOUNT',
                                     'LPC_AMOUNT', 'OTHER_AMOUNT', 'fncl_trans_method_cd', 'fncl_trans_channel_cd']

#df_pmt_tz_cut_dlnqcyc_sum.head() 100142994 100184538 100057601

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_sum[(df_pmt_tz_cut_dlnqcyc_sum["BAN"] == '100142994')]

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_sum[(df_pmt_tz_cut_dlnqcyc_sum["BAN"] == '100184538')]

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_sum[(df_pmt_tz_cut_dlnqcyc_sum["BAN"] == '100057601')]

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_sum.columns

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_sum_payments = df_pmt_tz_cut_dlnqcyc_sum[df_pmt_tz_cut_dlnqcyc_sum['TRANSACTION_TYPE']=='PYM']

df_pmt_tz_cut_dlnqcyc_sum_payments = df_pmt_tz_cut_dlnqcyc_sum_payments[['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END',
                                                                       'fncl_trans_dt_tm_tz_pacific', 'NUM_TRANSACTIONS',
                                                                       'PYM_AMOUNT', 'fncl_trans_method_cd',
                                                                       'fncl_trans_channel_cd']]

df_pmt_tz_cut_dlnqcyc_sum_payments.columns = ['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END',
                                            'fncl_trans_dt_tm_tz_pacific_PYMT', 'NUM_TRANSACTIONS_PYMT',
                                            'PYMT_AMOUNT', 'fncl_trans_method_cd_PYMT',
                                            'fncl_trans_channel_cd_PYMT']

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_sum_adjustments = df_pmt_tz_cut_dlnqcyc_sum[df_pmt_tz_cut_dlnqcyc_sum['TRANSACTION_TYPE']=='ADJ']

df_pmt_tz_cut_dlnqcyc_sum_adjustments = df_pmt_tz_cut_dlnqcyc_sum_adjustments[['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END',
                                                                       'fncl_trans_dt_tm_tz_pacific', 'NUM_TRANSACTIONS',
                                                                       'ADJ_AMOUNT', 'fncl_trans_method_cd',
                                                                       'fncl_trans_channel_cd']]

df_pmt_tz_cut_dlnqcyc_sum_adjustments.columns = ['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END',
                                            'fncl_trans_dt_tm_tz_pacific_ADJ', 'NUM_TRANSACTIONS_ADJ',
                                            'ADJ_AMOUNT', 'fncl_trans_method_cd_ADJ',
                                            'fncl_trans_channel_cd_ADJ']

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_sum_bck = df_pmt_tz_cut_dlnqcyc_sum[df_pmt_tz_cut_dlnqcyc_sum['TRANSACTION_TYPE']=='BCK']

df_pmt_tz_cut_dlnqcyc_sum_bck = df_pmt_tz_cut_dlnqcyc_sum_bck[['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END',
                                                                       'fncl_trans_dt_tm_tz_pacific', 'NUM_TRANSACTIONS',
                                                                       'BCK_AMOUNT', 'fncl_trans_method_cd',
                                                                       'fncl_trans_channel_cd']]

df_pmt_tz_cut_dlnqcyc_sum_bck.columns = ['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END',
                                            'fncl_trans_dt_tm_tz_pacific_BCK', 'NUM_TRANSACTIONS_BCK',
                                            'BCK_AMOUNT', 'fncl_trans_method_cd_BCK',
                                            'fncl_trans_channel_cd_BCK']

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_sum_lpc = df_pmt_tz_cut_dlnqcyc_sum[df_pmt_tz_cut_dlnqcyc_sum['TRANSACTION_TYPE']=='LPC']

df_pmt_tz_cut_dlnqcyc_sum_lpc = df_pmt_tz_cut_dlnqcyc_sum_lpc[['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END',
                                                                       'fncl_trans_dt_tm_tz_pacific', 'NUM_TRANSACTIONS',
                                                                       'LPC_AMOUNT', 'fncl_trans_method_cd',
                                                                       'fncl_trans_channel_cd']]

df_pmt_tz_cut_dlnqcyc_sum_lpc.columns = ['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END',
                                            'fncl_trans_dt_tm_tz_pacific_LPC', 'NUM_TRANSACTIONS_LPC',
                                            'LPC_AMOUNT', 'fncl_trans_method_cd_LPC',
                                            'fncl_trans_channel_cd_LPC']

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_sum_other = df_pmt_tz_cut_dlnqcyc_sum[df_pmt_tz_cut_dlnqcyc_sum['TRANSACTION_TYPE']=='OTHER']

df_pmt_tz_cut_dlnqcyc_sum_other = df_pmt_tz_cut_dlnqcyc_sum_other[['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END',
                                                                       'fncl_trans_dt_tm_tz_pacific', 'NUM_TRANSACTIONS',
                                                                       'OTHER_AMOUNT', 'fncl_trans_method_cd',
                                                                       'fncl_trans_channel_cd']]

df_pmt_tz_cut_dlnqcyc_sum_other.columns = ['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END',
                                            'fncl_trans_dt_tm_tz_pacific_OTHER', 'NUM_TRANSACTIONS_OTHER',
                                            'OTHER_AMOUNT', 'fncl_trans_method_cd_OTHER',
                                            'fncl_trans_channel_cd_OTHER']

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_sum_payments[df_pmt_tz_cut_dlnqcyc_sum_payments['BAN']=='100184538']

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_sum_adjustments[df_pmt_tz_cut_dlnqcyc_sum_adjustments['BAN']=='100184538']

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_sum_bck[df_pmt_tz_cut_dlnqcyc_sum_bck['BAN']=='100184538']

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_sum_lpc[df_pmt_tz_cut_dlnqcyc_sum_lpc['BAN']=='100184538']

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_sum_other[df_pmt_tz_cut_dlnqcyc_sum_other['BAN']=='100184538']

# COMMAND ----------

# MAGIC %md
# MAGIC ### Payments as base

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP = pd.merge(df_pmt_tz_cut_dlnqcyc_sum_payments, df_pmt_tz_cut_dlnqcyc_sum_adjustments,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP = pd.merge(df_pmt_tz_cut_dlnqcyc_NODUP, df_pmt_tz_cut_dlnqcyc_sum_bck,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP = pd.merge(df_pmt_tz_cut_dlnqcyc_NODUP, df_pmt_tz_cut_dlnqcyc_sum_lpc,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP = pd.merge(df_pmt_tz_cut_dlnqcyc_NODUP, df_pmt_tz_cut_dlnqcyc_sum_other,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP[df_pmt_tz_cut_dlnqcyc_NODUP['BAN']=='100184538']

# COMMAND ----------

# MAGIC %md
# MAGIC #### Adjustments as base

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_02 = pd.merge(df_pmt_tz_cut_dlnqcyc_sum_adjustments, df_pmt_tz_cut_dlnqcyc_sum_payments,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_02 = pd.merge(df_pmt_tz_cut_dlnqcyc_NODUP_02, df_pmt_tz_cut_dlnqcyc_sum_bck,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_02 = pd.merge(df_pmt_tz_cut_dlnqcyc_NODUP_02, df_pmt_tz_cut_dlnqcyc_sum_lpc,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_02 = pd.merge(df_pmt_tz_cut_dlnqcyc_NODUP_02, df_pmt_tz_cut_dlnqcyc_sum_other,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_02[df_pmt_tz_cut_dlnqcyc_NODUP_02['BAN']=='100184538']

# COMMAND ----------

# MAGIC %md
# MAGIC #### BCKs as base

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_03 = pd.merge(df_pmt_tz_cut_dlnqcyc_sum_bck, df_pmt_tz_cut_dlnqcyc_sum_payments,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_03 = pd.merge(df_pmt_tz_cut_dlnqcyc_NODUP_03, df_pmt_tz_cut_dlnqcyc_sum_adjustments,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_03 = pd.merge(df_pmt_tz_cut_dlnqcyc_NODUP_03, df_pmt_tz_cut_dlnqcyc_sum_lpc,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_03 = pd.merge(df_pmt_tz_cut_dlnqcyc_NODUP_03, df_pmt_tz_cut_dlnqcyc_sum_other,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_03[df_pmt_tz_cut_dlnqcyc_NODUP_03['BAN']=='100184538']

# COMMAND ----------

# MAGIC %md
# MAGIC ### LPC as base

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_04 = pd.merge(df_pmt_tz_cut_dlnqcyc_sum_lpc, df_pmt_tz_cut_dlnqcyc_sum_payments,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_04 = pd.merge(df_pmt_tz_cut_dlnqcyc_NODUP_04, df_pmt_tz_cut_dlnqcyc_sum_adjustments,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_04 = pd.merge(df_pmt_tz_cut_dlnqcyc_NODUP_04, df_pmt_tz_cut_dlnqcyc_sum_bck,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_04 = pd.merge(df_pmt_tz_cut_dlnqcyc_NODUP_04, df_pmt_tz_cut_dlnqcyc_sum_other,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_04[df_pmt_tz_cut_dlnqcyc_NODUP_04['BAN']=='100184538']

# COMMAND ----------

# MAGIC %md
# MAGIC ### OTHER as base

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_05 = pd.merge(df_pmt_tz_cut_dlnqcyc_sum_other, df_pmt_tz_cut_dlnqcyc_sum_payments,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_05 = pd.merge(df_pmt_tz_cut_dlnqcyc_NODUP_05, df_pmt_tz_cut_dlnqcyc_sum_adjustments,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_05 = pd.merge(df_pmt_tz_cut_dlnqcyc_NODUP_05, df_pmt_tz_cut_dlnqcyc_sum_bck,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_05 = pd.merge(df_pmt_tz_cut_dlnqcyc_NODUP_05, df_pmt_tz_cut_dlnqcyc_sum_lpc,
                                       how='left',
                                       left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                       right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_05[df_pmt_tz_cut_dlnqcyc_NODUP_05['BAN']=='100184538']

# COMMAND ----------

# MAGIC %md
# MAGIC #### Concat

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_FL = pd.concat([df_pmt_tz_cut_dlnqcyc_NODUP, 
                                            df_pmt_tz_cut_dlnqcyc_NODUP_02, 
                                            df_pmt_tz_cut_dlnqcyc_NODUP_03,
                                            df_pmt_tz_cut_dlnqcyc_NODUP_04,
                                            df_pmt_tz_cut_dlnqcyc_NODUP_05], axis=0)

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_FL.columns

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_FL = df_pmt_tz_cut_dlnqcyc_NODUP_FL[['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END',
                                            'fncl_trans_dt_tm_tz_pacific_PYMT', 'NUM_TRANSACTIONS_PYMT',
                                            'PYMT_AMOUNT', 'fncl_trans_method_cd_PYMT',
                                            'fncl_trans_channel_cd_PYMT',
                                            'fncl_trans_dt_tm_tz_pacific_ADJ', 'NUM_TRANSACTIONS_ADJ',
                                            'ADJ_AMOUNT', 'fncl_trans_method_cd_ADJ',
                                            'fncl_trans_channel_cd_ADJ',
                                            'fncl_trans_dt_tm_tz_pacific_BCK', 'NUM_TRANSACTIONS_BCK',
                                            'BCK_AMOUNT', 'fncl_trans_method_cd_BCK',
                                            'fncl_trans_channel_cd_BCK',
                                            'fncl_trans_dt_tm_tz_pacific_LPC', 'NUM_TRANSACTIONS_LPC',
                                            'LPC_AMOUNT', 'fncl_trans_method_cd_LPC',
                                            'fncl_trans_channel_cd_LPC',
                                            'fncl_trans_dt_tm_tz_pacific_OTHER', 'NUM_TRANSACTIONS_OTHER',
                                            'OTHER_AMOUNT', 'fncl_trans_method_cd_OTHER',
                                            'fncl_trans_channel_cd_OTHER']]

# COMMAND ----------

dup = df_pmt_tz_cut_dlnqcyc_NODUP_FL[df_pmt_tz_cut_dlnqcyc_NODUP_FL.duplicated(['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])]

dup.shape

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_FL = df_pmt_tz_cut_dlnqcyc_NODUP_FL.sort_values(['BAN','TRANS_DTTM_PACIFIC_END']).drop_duplicates(['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'], keep='first')

# COMMAND ----------

dup = df_pmt_tz_cut_dlnqcyc_NODUP_FL[df_pmt_tz_cut_dlnqcyc_NODUP_FL.duplicated(['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])]

dup.shape

# COMMAND ----------

df_pmt_tz_cut[(df_pmt_tz_cut["BAN"] == '100184538') & (df_pmt_tz_cut['fncl_trans_dt_tm_tz_pacific'].dt.month >= 8)
         & (df_pmt_tz_cut['fncl_trans_dt_tm_tz_pacific'].dt.month <= 10) 
              & (df_pmt_tz_cut['fncl_trans_dt_tm_tz_pacific'].dt.year == 2019)]

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_FL[df_pmt_tz_cut_dlnqcyc_NODUP_FL['BAN']=='100184538']

# COMMAND ----------

df_pmt_tz[(df_pmt_tz["acct_nbr"] == '100184538')]

# COMMAND ----------

merged_df2_cut_tm[(merged_df2_cut_tm["BAN"] == '100184538')]

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_FL[df_pmt_tz_cut_dlnqcyc_NODUP_FL['BAN']=='100057601']

# COMMAND ----------

merged_df2_cut_tm.shape

# COMMAND ----------

merged_df2_cut_tm.BAN.unique()

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_FL.shape

# COMMAND ----------

df_pmt_tz_cut_dlnqcyc_NODUP_FL.columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Add the payments to data frame

# COMMAND ----------

merged_df2_cut_tm_pymt = pd.merge(merged_df2_cut_tm, df_pmt_tz_cut_dlnqcyc_NODUP_FL, how='left',
                                     left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                     right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

merged_df2_cut_tm_pymt.shape

# COMMAND ----------

merged_df2_cut_tm_pymt[(merged_df2_cut_tm_pymt["BAN"] == '100184538')]

# COMMAND ----------

merged_df2_cut_tm_pymt[(merged_df2_cut_tm_pymt["BAN"] == '100057601')] 

# COMMAND ----------

#merged_df2_cut_tm_pymt.to_csv("merged_df2_cut_tm_pymt_6_30_21.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC # Join Paymet Arrangements

# COMMAND ----------

#AQUI

# COMMAND ----------

df_pa.head(1)

# COMMAND ----------

df_ptp_tz = df_pa[['acct_nbr', 'ptp_taken_dt', 'Pmt_Pln_Crtn_Dt_Tm', 'Trans_Dt_Tm', 'Prms_Take_Dt', 'Prms_Take_Tm',
                   'ptp_final_disposition_dt', 'ptp_init_prms_dt', 'ptp_sbsqt_prms_dt',
                   'ptp_initial_disposition_cd', 'ptp_final_disposition_cd',
                   'ptp_source_cd', 'ptp_tot_prms_amt',
                   'ptp_pln_type_cd', 'ptp_init_prms_amt',
                   'ptp_init_prms_method_cd', 'ptp_sbsqt_prms_amt',
                   'ptp_sbsqt_prms_method_cd']]

df_ptp_tz.columns = ['BAN', 'PTP_TAKEN_DT', 'PTP_PMT_PLN_CRTN_DTTM', 'PTP_TRANS_DTTM', 'PRMS_TAKE_DT', 'PRMS_TAKE_TIME', 
                     'PTP_FINAL_DISPOSITION_DT', 'PTP_INIT_PRMS_DT', 'PTP_SUBSQT_PRMS_DT',
                     'PTP_INI_CD', 'PTP_FIN_CD',
                     'PTP_SOURCE', 'PTP_TOT_PRMS_AMT', 'PTP_PLN_TYPE', 'PTP_INI_PRMS_AMT',
                     'PTP_INI_PRMS_METHOD', 'PTP_SUBSQT_PRMS_AMT',
                     'PTP_SUBSQT_PRMS_METHOD']

df_ptp_tz.BAN = df_ptp_tz.BAN.astype(str)

df_ptp_tz['PTP_TAKEN_DT'] = pd.to_datetime(df_ptp_tz['PTP_TAKEN_DT'], errors = 'coerce')


df_ptp_tz['PTP_PMT_PLN_CRTN_DTTM'] = pd.to_datetime(df_ptp_tz['PTP_PMT_PLN_CRTN_DTTM'], errors = 'coerce')
df_ptp_tz['PTP_TRANS_DTTM'] = pd.to_datetime(df_ptp_tz['PTP_TRANS_DTTM'], errors = 'coerce')
df_ptp_tz['PTP_FINAL_DISPOSITION_DT'] =  pd.to_datetime(df_ptp_tz['PTP_FINAL_DISPOSITION_DT'], errors = 'coerce')
df_ptp_tz['PTP_INIT_PRMS_DT'] =  pd.to_datetime(df_ptp_tz['PTP_INIT_PRMS_DT'], errors = 'coerce')
df_ptp_tz['PTP_SUBSQT_PRMS_DT'] =  pd.to_datetime(df_ptp_tz['PTP_SUBSQT_PRMS_DT'], errors = 'coerce')
df_ptp_tz['PRMS_TAKE_DT'] =  pd.to_datetime(df_ptp_tz['PRMS_TAKE_DT'], errors = 'coerce')

df_ptp_tz = df_ptp_tz.sort_values(by = ['BAN', 'PTP_TAKEN_DT'])

df_ptp_tz = df_ptp_tz.reset_index(drop=True)

# COMMAND ----------

df_ptp_tz.head(58)

# COMMAND ----------

df_ptp_tz.dtypes

# COMMAND ----------

df_ptp_tz.shape

# COMMAND ----------

df_ptp_tz.isna().sum() # PTP_PMT_PLN_CRTN_DTTM and PTP_TRANS_DTTM all missing

# COMMAND ----------

spike_cols = [col for col in df_ptp_tz.select_dtypes(include='object').columns if 'BAN' not in col]
spike_cols

# COMMAND ----------

for i in spike_cols:
    
    print(str(i)+': ')
    print(df_ptp_tz[i].value_counts(dropna=False))

# COMMAND ----------

df_ptp_tz['PRMS_TAKE_DT'].equals(df_ptp_tz['PTP_TAKEN_DT'])

# COMMAND ----------

df_ptp_tz.columns

# COMMAND ----------

df_ptp_tz = df_ptp_tz[['BAN', 'PTP_TAKEN_DT', 'PRMS_TAKE_TIME', 'PTP_FINAL_DISPOSITION_DT',
       'PTP_INIT_PRMS_DT', 'PTP_SUBSQT_PRMS_DT', 'PTP_INI_CD', 'PTP_FIN_CD',
       'PTP_SOURCE', 'PTP_TOT_PRMS_AMT', 'PTP_INI_PRMS_AMT',
       'PTP_INI_PRMS_METHOD', 'PTP_SUBSQT_PRMS_AMT', 'PTP_SUBSQT_PRMS_METHOD']]

# COMMAND ----------

df_ptp_tz['len_time'] = df_ptp_tz.apply(lambda x: len(str(x.PRMS_TAKE_TIME)), axis=1)
df_ptp_tz['PTP_TAKEN_DTTM'] = df_ptp_tz.apply(lambda x: new_datetime(x, 'PRMS_TAKE_TIME', 'PTP_TAKEN_DT', central_tz), axis=1)
df_ptp_tz['PTP_TIMEZONE'] = df_ptp_tz.apply(lambda x: x.PTP_TAKEN_DTTM.tzinfo, axis=1)
df_ptp_tz['PTP_TAKEN_DTTM_PACIFIC'] = df_ptp_tz.apply(lambda x: x['PTP_TAKEN_DTTM'].astimezone(pacific_tz), axis=1)

# COMMAND ----------

df_ptp_tz = df_ptp_tz[['BAN', 'PTP_TAKEN_DT', 'PRMS_TAKE_TIME', 'PTP_TAKEN_DTTM','PTP_TIMEZONE', 'PTP_TAKEN_DTTM_PACIFIC',
                       'PTP_SOURCE', 'PTP_TOT_PRMS_AMT',
                       'PTP_INIT_PRMS_DT', 'PTP_INI_PRMS_AMT', 'PTP_INI_PRMS_METHOD', 'PTP_INI_CD',
                       'PTP_SUBSQT_PRMS_DT', 'PTP_SUBSQT_PRMS_AMT', 'PTP_SUBSQT_PRMS_METHOD',
                       'PTP_FINAL_DISPOSITION_DT', 'PTP_FIN_CD']]

# COMMAND ----------

df_ptp_tz.PTP_SUBSQT_PRMS_METHOD = df_ptp_tz.PTP_SUBSQT_PRMS_METHOD.fillna('NONE')
df_ptp_tz.PTP_SUBSQT_PRMS_METHOD = df_ptp_tz.PTP_SUBSQT_PRMS_METHOD.replace('?', 'NONE')

# COMMAND ----------

dup = df_ptp_tz[df_ptp_tz.duplicated(['BAN', 'PTP_TAKEN_DT', 'PRMS_TAKE_TIME', 'PTP_TAKEN_DTTM','PTP_TIMEZONE', 'PTP_TAKEN_DTTM_PACIFIC',
                       'PTP_SOURCE', 'PTP_TOT_PRMS_AMT',
                       'PTP_INIT_PRMS_DT', 'PTP_INI_PRMS_AMT', 'PTP_INI_PRMS_METHOD', 'PTP_INI_CD',
                       'PTP_SUBSQT_PRMS_DT', 'PTP_SUBSQT_PRMS_AMT', 'PTP_SUBSQT_PRMS_METHOD',
                       'PTP_FINAL_DISPOSITION_DT', 'PTP_FIN_CD'])]

dup.shape

# COMMAND ----------

df_ptp_tz = df_ptp_tz.drop_duplicates(['BAN', 'PTP_TAKEN_DT', 'PRMS_TAKE_TIME', 'PTP_TAKEN_DTTM','PTP_TIMEZONE', 'PTP_TAKEN_DTTM_PACIFIC',
                       'PTP_SOURCE', 'PTP_TOT_PRMS_AMT',
                       'PTP_INIT_PRMS_DT', 'PTP_INI_PRMS_AMT', 'PTP_INI_PRMS_METHOD', 'PTP_INI_CD',
                       'PTP_SUBSQT_PRMS_DT', 'PTP_SUBSQT_PRMS_AMT', 'PTP_SUBSQT_PRMS_METHOD',
                       'PTP_FINAL_DISPOSITION_DT', 'PTP_FIN_CD'], keep='first')

# COMMAND ----------

dup = df_ptp_tz[df_ptp_tz.duplicated(['BAN', 'PTP_TAKEN_DT', 'PRMS_TAKE_TIME', 'PTP_TAKEN_DTTM','PTP_TIMEZONE', 'PTP_TAKEN_DTTM_PACIFIC',
                       'PTP_SOURCE', 'PTP_TOT_PRMS_AMT',
                       'PTP_INIT_PRMS_DT', 'PTP_INI_PRMS_AMT', 'PTP_INI_PRMS_METHOD', 'PTP_INI_CD',
                       'PTP_SUBSQT_PRMS_DT', 'PTP_SUBSQT_PRMS_AMT', 'PTP_SUBSQT_PRMS_METHOD',
                       'PTP_FINAL_DISPOSITION_DT', 'PTP_FIN_CD'])]

dup.shape

# COMMAND ----------

df_ptp_tz.shape

# COMMAND ----------

df_ptp_tz.head()

# COMMAND ----------

ptp_match = pd.merge(merged_df2_cut_tm_pymt[['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END']],
                     df_ptp_tz, 
                     how='left',
                     left_on=['BAN'],
                     right_on=['BAN'])

# COMMAND ----------

ptp_match = ptp_match[(ptp_match['PTP_TAKEN_DTTM_PACIFIC'] > 
                       ptp_match['TRANS_DTTM_PACIFIC_START']) &
                      (ptp_match['PTP_TAKEN_DTTM_PACIFIC'] <=
                       ptp_match['TRANS_DTTM_PACIFIC_END'])]

# COMMAND ----------

ptp_match.head()

# COMMAND ----------

ptp_match.shape

# COMMAND ----------

dup_bans = ptp_match[ptp_match.duplicated(['BAN', 'TRANS_DTTM_PACIFIC_END'])]['BAN'].unique()

# COMMAND ----------

ptp_match[ptp_match.BAN.isin(dup_bans)].sort_values(by=['BAN', 'TRANS_DTTM_PACIFIC_END', 'PTP_TAKEN_DTTM'],
                                                   ascending=[False, False, False])

# COMMAND ----------

dup = ptp_match[ptp_match.duplicated(['BAN', 'TRANS_DTTM_PACIFIC_END'])]

dup.shape

# COMMAND ----------

ptp_match = ptp_match.drop_duplicates(['BAN', 'TRANS_DTTM_PACIFIC_END'], keep='first')

# COMMAND ----------

dup = ptp_match[ptp_match.duplicated(['BAN', 'TRANS_DTTM_PACIFIC_END'])]

dup.shape

# COMMAND ----------

merged_df2_cut_tm_pymt.shape

# COMMAND ----------

len(merged_df2_cut_tm_pymt.BAN.unique())

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr = pd.merge(merged_df2_cut_tm_pymt,
                                         ptp_match, 
                                         how='left',
                                         left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                         right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr.shape

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr.head()

# COMMAND ----------

#df_ptp_tz[(df_ptp_tz["BAN"] == '100184538')]

# COMMAND ----------

#merged_df2_cut_tm_pymt_pyarr[(merged_df2_cut_tm_pymt_pyarr["BAN"] == '100184538')] 

# COMMAND ----------

#df_ptp_tz[(df_ptp_tz["BAN"] == '100057601')]

# COMMAND ----------

#merged_df2_cut_tm_pymt_pyarr[(merged_df2_cut_tm_pymt_pyarr["BAN"] == '100057601')] 

# COMMAND ----------

# '''df_pmt_timestamp[(df_pmt_timestamp["acct_nbr"] == 100057601) & (df_pmt_timestamp['fncl_trans_dt'].dt.month >= 8)
#          & (df_pmt_timestamp['fncl_trans_dt'].dt.month <= 11)
#               & (df_pmt_timestamp['fncl_trans_dt'].dt.year == 2019)].sort_values(by=['fncl_trans_dt'], ascending=True)

# COMMAND ----------

# '''df_act[(df_act["acct_nbr"] == 100057601) & (df_act['critical_action_dt'].dt.month >= 8)
#          & (df_act['critical_action_dt'].dt.month <= 11)
#               & (df_act['critical_action_dt'].dt.year == 2019)].sort_values(by=['critical_action_dt'], ascending=True)

# COMMAND ----------

#df_ptp_tz[(df_ptp_tz["BAN"] == '100142994')]

# COMMAND ----------

#merged_df2_cut_tm_pymt_pyarr[(merged_df2_cut_tm_pymt_pyarr["BAN"] == '100142994')] 

# COMMAND ----------

#merged_df2_cut_tm_pymt_pyarr.to_csv("merged_df2_cut_tm_pymt_pyarr_7_1_21.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC # Join Billing Info

# COMMAND ----------

#AQUI

# COMMAND ----------

df_billing.head(1)

# COMMAND ----------

df_bill = df_billing[['acct_nbr', 'bill_dt', 'bill_due_dt', 'bill_tot_due_amt', 'bill_curr_chrgs_amt', 'bill_past_due_amt', 'first_final_bill_ind']]

df_bill.columns = ['BAN', 'BILL_DT', 'BILL_DUE_DT', 'BILL_TOT_DUE_AMT', 'BILL_CURR_CHRGS_AMT', 'BILL_PAST_DUE_AMT', 'FIRST_FINAL_BILL_IND']

df_bill.BAN = df_bill.BAN.astype(str)

df_bill['BILL_DT'] = df_bill['BILL_DT'].astype('datetime64[D]')
df_bill['BILL_DUE_DT'] = df_bill['BILL_DUE_DT'].astype('datetime64[D]')

df_bill = df_bill.sort_values(by = ['BAN', 'BILL_DUE_DT'], ascending=False)
df_bill = df_bill.reset_index(drop=True)

# COMMAND ----------

bill_match = pd.merge(merged_df2_cut_tm_pymt_pyarr[['BAN', 'DLNQ_DT_STR', 'RISK_CD']],
                     df_bill, 
                     how='left',
                     left_on=['BAN'],
                     right_on=['BAN'])

# COMMAND ----------

bill_match = bill_match[(bill_match['BILL_DUE_DT'] <= bill_match['DLNQ_DT_STR'])]

# COMMAND ----------

dup = bill_match[bill_match.duplicated(['BAN', 'DLNQ_DT_STR', 'RISK_CD'])]

dup.shape

# COMMAND ----------

bill_match = bill_match.drop_duplicates(['BAN', 'DLNQ_DT_STR', 'RISK_CD'], keep='first')

# COMMAND ----------

dup = bill_match[bill_match.duplicated(['BAN', 'DLNQ_DT_STR', 'RISK_CD'])]

dup.shape

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill = pd.merge(merged_df2_cut_tm_pymt_pyarr,
                                         bill_match, 
                                         how='left',
                                         left_on=['BAN', 'DLNQ_DT_STR', 'RISK_CD'],
                                         right_on=['BAN', 'DLNQ_DT_STR', 'RISK_CD'])

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill = merged_df2_cut_tm_pymt_pyarr_bill[['TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END', 'TRANS_DT', 'BAN',
                                                                       'RISK_CD', 'INSTR_CD', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR',
                                                                       'CURE_DT_PACIFIC', 'WO_DT', 'OCA_DT', 'LST_EVENT_DT', 'LG_SUSP_DT',
                                                                       'RCMD_PROJ_SUSP_DT', 'PAR_PROJ_SUSP_DT', 'PROJ_SUSP_DT',
                                                                       'INIT_REQ_DTTM', 'PAR_RCMND_DTTM', 'DAYS_TO_EVENT', 'BILL_DT', 'BILL_DUE_DT',
                                                                       'BILL_TOT_DUE_AMT', 'BILL_CURR_CHRGS_AMT', 'BILL_PAST_DUE_AMT',
                                                                       'FIRST_FINAL_BILL_IND', 'TOT_DUE_AMT',
                                                                       'TOT_DLNQ_AMT', 'T1', 'T2', 'T3', 'T4', 'T5', 'TRANS_TYPE',
                                                                       'TRANS_SUB_TYPE', 'TREAT_MSG_CD', 'TREAT_MSG_CD_TIMECUTOFF',
                                                                       'TREAT_DT_PACIFIC_CUTOFF', 'TREAT_DTTM_PACIFIC_CUTOFF', 'TRANS_DT_FAKE',
                                                                       'DAYS_IN_DLNQ', 'fncl_trans_dt_tm_tz_pacific_PYMT',
                                                                       'NUM_TRANSACTIONS_PYMT', 'PYMT_AMOUNT', 'fncl_trans_method_cd_PYMT',
                                                                       'fncl_trans_channel_cd_PYMT', 'fncl_trans_dt_tm_tz_pacific_ADJ',
                                                                       'NUM_TRANSACTIONS_ADJ', 'ADJ_AMOUNT', 'fncl_trans_method_cd_ADJ',
                                                                       'fncl_trans_channel_cd_ADJ', 'fncl_trans_dt_tm_tz_pacific_BCK',
                                                                       'NUM_TRANSACTIONS_BCK', 'BCK_AMOUNT', 'fncl_trans_method_cd_BCK',
                                                                       'fncl_trans_channel_cd_BCK', 'fncl_trans_dt_tm_tz_pacific_LPC',
                                                                       'NUM_TRANSACTIONS_LPC', 'LPC_AMOUNT', 'fncl_trans_method_cd_LPC',
                                                                       'fncl_trans_channel_cd_LPC', 'fncl_trans_dt_tm_tz_pacific_OTHER',
                                                                       'NUM_TRANSACTIONS_OTHER', 'OTHER_AMOUNT', 'fncl_trans_method_cd_OTHER',
                                                                       'fncl_trans_channel_cd_OTHER', 'PTP_TAKEN_DT', 'PRMS_TAKE_TIME',
                                                                       'PTP_TAKEN_DTTM', 'PTP_TIMEZONE', 'PTP_TAKEN_DTTM_PACIFIC',
                                                                       'PTP_SOURCE', 'PTP_TOT_PRMS_AMT', 'PTP_INIT_PRMS_DT',
                                                                       'PTP_INI_PRMS_AMT', 'PTP_INI_PRMS_METHOD', 'PTP_INI_CD',
                                                                       'PTP_SUBSQT_PRMS_DT', 'PTP_SUBSQT_PRMS_AMT', 'PTP_SUBSQT_PRMS_METHOD',
                                                                       'PTP_FINAL_DISPOSITION_DT', 'PTP_FIN_CD']]

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill.shape

# COMMAND ----------

len(merged_df2_cut_tm_pymt_pyarr_bill.BAN.unique())

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr.shape

# COMMAND ----------

len(merged_df2_cut_tm_pymt_pyarr.BAN.unique())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join Survey data

# COMMAND ----------

#ACA

# COMMAND ----------

df_survey.head(1)

# COMMAND ----------

df_survey.survey_call_center_cd.head()

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill[merged_df2_cut_tm_pymt_pyarr_bill['BAN']=='100184538']

# COMMAND ----------

df_svy = df_survey[['acct_nbr', 'survey_dt', 'survey_rep_sat_score', 'survey_wtr_score', 'survey_call_center_cd']]

df_svy.columns = ['BAN', 'TRANS_DT', 'SVY_SAT_SCORE', 'SVY_WTR_SCORE', 'SVY_CALL_CENTER_CD']

df_svy.BAN = df_svy.BAN.astype(str)

df_svy['TRANS_DT'] = df_svy['TRANS_DT'].astype('datetime64[D]')
df_svy = df_svy.sort_values(by = ['BAN', 'TRANS_DT'], ascending=False)

df_svy = df_svy.reset_index(drop=True)

# COMMAND ----------

df_svy.head()

# COMMAND ----------

df_svy.shape

# COMMAND ----------

dup = df_svy[df_svy.duplicated(['BAN', 'TRANS_DT', 'SVY_SAT_SCORE', 'SVY_WTR_SCORE', 'SVY_CALL_CENTER_CD'])]

dup.shape

# COMMAND ----------

df_svy = df_svy.drop_duplicates(['BAN', 'TRANS_DT', 'SVY_SAT_SCORE', 'SVY_WTR_SCORE', 'SVY_CALL_CENTER_CD'], keep='first')

# COMMAND ----------

dup = df_svy[df_svy.duplicated(['BAN', 'TRANS_DT', 'SVY_SAT_SCORE', 'SVY_WTR_SCORE', 'SVY_CALL_CENTER_CD'])]

dup.shape

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill_svy = pd.merge(merged_df2_cut_tm_pymt_pyarr_bill,
                                                 df_svy,
                                                 how='left',
                                                 left_on=['BAN', 'TRANS_DT'],
                                                 right_on=['BAN', 'TRANS_DT'])

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill_svy.SVY_SAT_SCORE.isna().sum()

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill_svy.SVY_WTR_SCORE.isna().sum()

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill_svy.shape

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join IB Calls

# COMMAND ----------

#ACA

# COMMAND ----------

df_calls.head(1)

# COMMAND ----------

df_ib_call_tz = df_calls[['acct_nbr', 'call_dt', 'call_time', 'call_type_cd', 'call_center_cd']]
df_ib_call_tz.columns = ['BAN', 'IB_CALL_DT', 'IB_CALL_TIME', 'IB_CALL_TRANS_TYPE', 'IB_CALL_TRANS_SUB_TYPE']

df_ib_call_tz['BAN'] = df_ib_call_tz['BAN'].astype(str)
df_ib_call_tz['IB_CALL_DT'] = df_ib_call_tz['IB_CALL_DT'].astype('datetime64[D]')

df_ib_call_tz = df_ib_call_tz[df_ib_call_tz['IB_CALL_TRANS_TYPE']=='IB']

df_ib_call_tz = df_ib_call_tz.sort_values(by =['BAN','IB_CALL_DT'], ascending=True)
df_ib_call_tz = df_ib_call_tz.reset_index(drop=True)

#New datetime

df_ib_call_tz['len_time'] = df_ib_call_tz.apply(lambda x: len(str(x.IB_CALL_TIME)), axis=1)
df_ib_call_tz['IB_CALL_DT_TZ'] = df_ib_call_tz.apply(lambda x: new_datetime(x, 'IB_CALL_TIME', 'IB_CALL_DT', central_tz), axis=1)
df_ib_call_tz['IB_CALL_TIMEZONE'] = df_ib_call_tz.apply(lambda x: x.IB_CALL_DT_TZ.tzinfo, axis=1)
df_ib_call_tz['IB_CALL_DT_TZ_PACIFIC'] = df_ib_call_tz.apply(lambda x: x['IB_CALL_DT_TZ'].astimezone(pacific_tz), axis=1)

df_ib_call_tz = df_ib_call_tz.sort_values(by =['BAN','IB_CALL_DT_TZ_PACIFIC'], ascending=True)
df_ib_call_tz = df_ib_call_tz.reset_index(drop=True)


df_ib_call_tz['IB_CALL_DT_PACIFIC'] = df_ib_call_tz.apply(lambda x: x['IB_CALL_DT_TZ_PACIFIC'].date(), axis=1)

# COMMAND ----------

df_ib_call_tz.IB_CALL_TRANS_TYPE.unique()

# COMMAND ----------

df_ib_call_tz.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### How many changes in day we would have

# COMMAND ----------

df_ib_call_tz['IND_TIME_CHANGED_DAY'] = df_ib_call_tz.apply(lambda x: 0 if x['IB_CALL_DT_TZ'].date() ==  x['IB_CALL_DT_TZ_PACIFIC'].date() else 1, axis=1)

# COMMAND ----------

df_ib_call_tz['IND_TIME_CHANGED_DAY'].sum() #There are 70725 changes of day because of time zone difference

# COMMAND ----------

df_ib_call_tz.shape

# COMMAND ----------

df_ib_call_tz.head()

# COMMAND ----------

df_ib_call_tz['IB_CALL_DT_PACIFIC'] = df_ib_call_tz['IB_CALL_DT_PACIFIC'].astype('datetime64[D]')

# COMMAND ----------

df_ib_call_tz_cut = df_ib_call_tz[['BAN', 'IB_CALL_DT', 'IB_CALL_DT_TZ', 'IB_CALL_TRANS_TYPE', 'IB_CALL_TRANS_SUB_TYPE', 'IB_CALL_DT_PACIFIC', 'IB_CALL_DT_TZ_PACIFIC']]

# COMMAND ----------

df_ib_call_tz_cut[(df_ib_call_tz_cut["BAN"] == '100142994') & (df_ib_call_tz_cut['IB_CALL_DT_TZ_PACIFIC'].dt.month == 8)]

# COMMAND ----------

df_ib_call_tz_cut.dtypes

# COMMAND ----------

dup = df_ib_call_tz_cut[df_ib_call_tz_cut.duplicated(['BAN', 'IB_CALL_DT', 'IB_CALL_DT_TZ', 'IB_CALL_TRANS_TYPE', 'IB_CALL_TRANS_SUB_TYPE', 'IB_CALL_DT_PACIFIC', 'IB_CALL_DT_TZ_PACIFIC'])]

print(len(dup))

dup.head()

# COMMAND ----------

df_ib_call_tz_cut = df_ib_call_tz_cut.drop_duplicates(['BAN', 'IB_CALL_DT', 'IB_CALL_DT_TZ', 'IB_CALL_TRANS_TYPE', 'IB_CALL_TRANS_SUB_TYPE', 'IB_CALL_DT_PACIFIC', 'IB_CALL_DT_TZ_PACIFIC'], keep='first')

# COMMAND ----------

dup = df_ib_call_tz_cut[df_ib_call_tz_cut.duplicated(['BAN', 'IB_CALL_DT', 'IB_CALL_DT_TZ', 'IB_CALL_TRANS_TYPE', 'IB_CALL_TRANS_SUB_TYPE', 'IB_CALL_DT_PACIFIC', 'IB_CALL_DT_TZ_PACIFIC'])]

print(len(dup))

dup.head()

# COMMAND ----------

df_ib_call_tz_cut.shape

# COMMAND ----------

df_ib_call_tz_cut.head()

# COMMAND ----------

IB_CALL_match = pd.merge(merged_df2_cut_tm_pymt_pyarr_bill[['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END']],
                         df_ib_call_tz_cut, 
                         how='left',
                         left_on=['BAN'],
                         right_on=['BAN'])

# COMMAND ----------

IB_CALL_match = IB_CALL_match[(IB_CALL_match['IB_CALL_DT_TZ_PACIFIC'] > 
                       IB_CALL_match['TRANS_DTTM_PACIFIC_START']) &
                      (IB_CALL_match['IB_CALL_DT_TZ_PACIFIC'] <=
                       IB_CALL_match['TRANS_DTTM_PACIFIC_END'])]

# COMMAND ----------

IB_CALL_match.shape

# COMMAND ----------

IB_CALL_match.head()

# COMMAND ----------

dup = IB_CALL_match[IB_CALL_match.duplicated(['BAN', 'TRANS_DTTM_PACIFIC_END'])]

dup.shape

# COMMAND ----------

dup = IB_CALL_match[IB_CALL_match.duplicated(['BAN', 'TRANS_DTTM_PACIFIC_START'])]

dup.shape

# COMMAND ----------

dup = IB_CALL_match[IB_CALL_match.duplicated(['BAN', 'IB_CALL_DT_PACIFIC'])]

dup.shape

# COMMAND ----------

IB_CALL_match.IB_CALL_TRANS_TYPE.isna().sum()

# COMMAND ----------

IB_CALL_match.IB_CALL_TRANS_SUB_TYPE.isna().sum()

# COMMAND ----------

IB_CALL_match['IB_CALL_IND']=np.where(IB_CALL_match['IB_CALL_TRANS_TYPE'].isna(), 0, 1)

# COMMAND ----------

IB_CALL_match.head()

# COMMAND ----------

IB_CALL_match_gp = IB_CALL_match.copy()

IB_CALL_match_gp.IB_CALL_DT_TZ_PACIFIC = IB_CALL_match_gp.IB_CALL_DT_TZ_PACIFIC.astype('str')

IB_CALL_match_gp = IB_CALL_match_gp.groupby(['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'], as_index = False).agg({'IB_CALL_TRANS_SUB_TYPE': ' | '.join, 'IB_CALL_DT_TZ_PACIFIC': ' | '.join, 'IB_CALL_IND': sum})

IB_CALL_match_gp.head()

# COMMAND ----------

dup = IB_CALL_match_gp[IB_CALL_match_gp.duplicated(['BAN', 'TRANS_DTTM_PACIFIC_END'])]

dup.shape

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill_svy_IB = pd.merge(merged_df2_cut_tm_pymt_pyarr_bill_svy,
                                                    IB_CALL_match_gp, 
                                                    how='left',
                                                    left_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'],
                                                    right_on=['BAN', 'TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END'])

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill_svy_IB.shape

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill_svy.shape

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill_svy_IB['IB_CALL_IND'] = merged_df2_cut_tm_pymt_pyarr_bill_svy_IB['IB_CALL_IND'].fillna(0)

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill_svy_IB['TRANS_MONYR']=pd.to_datetime(merged_df2_cut_tm_pymt_pyarr_bill_svy_IB['DLNQ_DT_PACIFIC']).dt.strftime('%Y%m')
merged_df2_cut_tm_pymt_pyarr_bill_svy_IB['TRANS_MONYR']=merged_df2_cut_tm_pymt_pyarr_bill_svy_IB['TRANS_MONYR'].astype('int')

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill_svy_IB.TRANS_MONYR.unique()

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill_svy_IB[merged_df2_cut_tm_pymt_pyarr_bill_svy_IB.DLNQ_DT_PACIFIC.dt.month != 8]['BAN'].unique()

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill_svy_IB['CUM_IB_CALL_IND'] = merged_df2_cut_tm_pymt_pyarr_bill_svy_IB.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])['IB_CALL_IND'].transform(lambda x: x.cumsum())
merged_df2_cut_tm_pymt_pyarr_bill_svy_IB['CUM_IB_CALL_IND'] = merged_df2_cut_tm_pymt_pyarr_bill_svy_IB['CUM_IB_CALL_IND'].fillna(0)

# COMMAND ----------

merged_df2_cut_tm_pymt_pyarr_bill_svy_IB.shape

# COMMAND ----------

#merged_df2_cut_tm_pymt_pyarr_bill_svy_IB[merged_df2_cut_tm_pymt_pyarr_bill_svy_IB['BAN']=='120653395']

# COMMAND ----------

#merged_df2_cut_tm_pymt_pyarr_bill_svy_IB.to_csv("merged_df2_cut_tm_pymt_pyarr_bill_svy_IB_7_7_21.csv")

# COMMAND ----------

df_iask_enabler = merged_df2_cut_tm_pymt_pyarr_bill_svy_IB.copy()

# COMMAND ----------

#df_iask_enabler[df_iask_enabler['BAN']=='100142994']

# COMMAND ----------

# MAGIC %md
# MAGIC #### Input days to event if does not cure before susp or never cures

# COMMAND ----------

df_iask_enabler['DAYS_TO_EVENT']=df_iask_enabler.apply(lambda x: (x['LST_EVENT_DT']-x['TRANS_DT']).days, axis=1)

# COMMAND ----------

#df_iask_enabler_100142994 = df_iask_enabler[df_iask_enabler['BAN']=='100142994']
#df_iask_enabler_100142994.to_csv('df_iask_enabler_100142994.csv')

# COMMAND ----------

#df_iask_enabler_100057601 = df_iask_enabler[df_iask_enabler['BAN']=='100057601']
#df_iask_enabler_100057601.to_csv('df_iask_enabler_100057601.csv')

# COMMAND ----------

#df_iask_enabler_100184538 = df_iask_enabler[df_iask_enabler['BAN']=='100184538']
#df_iask_enabler_100184538.to_csv('df_iask_enabler_100184538.csv')

# COMMAND ----------

#df_iask_enabler_120653395 = df_iask_enabler[df_iask_enabler['BAN']=='120653395']
#df_iask_enabler_120653395.to_csv('df_iask_enabler_120653395.csv')

# COMMAND ----------

# [100142994, 100057601, 100184538, 120653395]

# COMMAND ----------

df_iask_enabler.DAYS_TO_EVENT.max()

# COMMAND ----------

#df_iask_enabler['DAYS_TO_EVENT']=df_iask_enabler.apply(lambda x: x.DAYS_TO_EVENT if x.DAYS_TO_EVENT<=100 else 100, axis=1)

# COMMAND ----------

#'DAYS_TO_EVENT'
df_Freq = pd.Series(df_iask_enabler.DAYS_TO_EVENT).groupby(df_iask_enabler.DAYS_TO_EVENT).count().reset_index(name='FREQ')
plt.figure(1,figsize=(20,10))
#Plot Frequencies:
df_Freq.FREQ.plot(kind='bar',label='DAYS_TO_EVENT')
plt.legend()
plt.title('DAYS_TO_EVENT FREQ')

# COMMAND ----------

#df_iask_enabler[df_iask_enabler['DAYS_TO_EVENT']>=100].shape

# COMMAND ----------

df_iask_enabler.TRANS_MONYR.unique()

# COMMAND ----------

#df_iask_enabler.DAYS_TO_EVENT = df_iask_enabler.DAYS_TO_EVENT.astype('str')
#df_iask_enabler_GROUP = df_iask_enabler.groupby(['BAN', 'TRANS_MONYR'])['DAYS_TO_EVENT'].agg(', '.join).reset_index()
#df_iask_enabler_GROUP.head()

# COMMAND ----------

#df_iask_enabler_GROUP.shape

# COMMAND ----------

#df_iask_enabler_GROUP['CURED'] = df_iask_enabler_GROUP.apply(lambda x: 1 if x.DAYS_TO_EVENT.find(', 0') != -1 else 0, axis=1)

# COMMAND ----------

#df_iask_enabler_GROUP.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refreshing Billing derived vars

# COMMAND ----------

df_iask_enabler.head()

# COMMAND ----------

#df_iask_enabler['BILL_DUE_DT'].isna().sum()

# COMMAND ----------

#df_iask_enabler[df_iask_enabler['BILL_DUE_DT'].isna()].head()

# COMMAND ----------

df_iask_enabler['PASSED_DAYS_FROM_BILL_DUE_DATE_TO_TRANS_DT']=(df_iask_enabler['TRANS_DT'] - df_iask_enabler['BILL_DUE_DT']).dt.days

# COMMAND ----------

df_iask_enabler['PASSED_DAYS_FROM_BILL_DUE_DATE_TO_TRANS_DT'] = df_iask_enabler['PASSED_DAYS_FROM_BILL_DUE_DATE_TO_TRANS_DT'].fillna(90)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refreshing Payment derived vars

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cumulative financial transaction amounts to today

# COMMAND ----------

df_iask_enabler['ADJ_AMOUNT']=df_iask_enabler['ADJ_AMOUNT'].fillna(0)

df_iask_enabler['CUM_ADJ_AMT_TO_TODAY'] = df_iask_enabler.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])['ADJ_AMOUNT'].transform(lambda x: x.cumsum())

# COMMAND ----------

df_iask_enabler['BCK_AMOUNT']=df_iask_enabler['BCK_AMOUNT'].fillna(0)

df_iask_enabler['CUM_BCK_AMT_TO_TODAY'] = df_iask_enabler.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])['BCK_AMOUNT'].transform(lambda x: x.cumsum())

# COMMAND ----------

df_iask_enabler['PYMT_AMOUNT']=df_iask_enabler['PYMT_AMOUNT'].fillna(0)

df_iask_enabler['CUM_PYMT_AMT_TO_TODAY'] = df_iask_enabler.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])['PYMT_AMOUNT'].transform(lambda x: x.cumsum())

# COMMAND ----------

df_iask_enabler['LPC_AMOUNT']=df_iask_enabler['LPC_AMOUNT'].fillna(0)

df_iask_enabler['CUM_LPC_AMT_TO_TODAY'] = df_iask_enabler.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])['LPC_AMOUNT'].transform(lambda x: x.cumsum())

# COMMAND ----------

df_iask_enabler['OTHER_AMOUNT']=df_iask_enabler['OTHER_AMOUNT'].fillna(0)

df_iask_enabler['CUM_OTHER_AMT_TO_TODAY'] = df_iask_enabler.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])['OTHER_AMOUNT'].transform(lambda x: x.cumsum())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Days since last payment within dlnq cycle

# COMMAND ----------

df_iask_enabler['LAST_PYMT_DT'] = np.where(df_iask_enabler.PYMT_AMOUNT == 0, np.datetime64('NaT'), df_iask_enabler.TRANS_DT)
df_iask_enabler['LAST_PYMT_DT'] = df_iask_enabler.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])['LAST_PYMT_DT'].transform(lambda x: x.fillna(method = 'ffill'))

df_iask_enabler['PASSED_DAYS_FROM_LAST_PYMT_TO_TODAY'] = (df_iask_enabler['TRANS_DT'] - df_iask_enabler['LAST_PYMT_DT']).dt.days
df_iask_enabler['PASSED_DAYS_FROM_LAST_PYMT_TO_TODAY'] = df_iask_enabler['PASSED_DAYS_FROM_LAST_PYMT_TO_TODAY'].fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### TOT_DUE_AMT_TO_CUM_PYMT_TO_TODAY_RTO

# COMMAND ----------

df_iask_enabler['TOT_DUE_AMT_TO_CUM_PYMT_TO_TODAY_RTO'] = df_iask_enabler.apply(lambda x: x['CUM_PYMT_AMT_TO_TODAY']/x['TOT_DUE_AMT']*100 if (x['TOT_DUE_AMT']!=0) else 0, axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### BILL_TOT_DUE_AMT_TO_CUM_PYMT_TO_TODAY_RTO

# COMMAND ----------

df_iask_enabler['BILL_TOT_DUE_AMT_TO_CUM_PYMT_TO_TODAY_RTO'] = df_iask_enabler.apply(lambda x: x['CUM_PYMT_AMT_TO_TODAY']/x['BILL_TOT_DUE_AMT']*100 if (x['BILL_TOT_DUE_AMT']!=0) else 0, axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Daily TOT_DUE_AMT / TOT_DLNQ_AMT / BILL_TOT_DUE_AMT Update

# COMMAND ----------

df_iask_enabler['UP_TO_TODAY_TOT_DUE_AMT'] = df_iask_enabler['TOT_DUE_AMT']-df_iask_enabler['CUM_PYMT_AMT_TO_TODAY']
df_iask_enabler['UP_TO_TODAY_TOT_DLNQ_AMT'] = df_iask_enabler['TOT_DLNQ_AMT']-df_iask_enabler['CUM_PYMT_AMT_TO_TODAY']
df_iask_enabler['UP_TO_TODAY_BILL_TOT_DUE_AMT'] = df_iask_enabler['BILL_TOT_DUE_AMT']-df_iask_enabler['CUM_PYMT_AMT_TO_TODAY']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refreshing Payment Arrangement derived vars

# COMMAND ----------

# MAGIC %md
# MAGIC #### Passed days from last pyarr to today

# COMMAND ----------

df_iask_enabler['DRAGGED_PTP_TAKEN_DT'] = df_iask_enabler.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])['PTP_TAKEN_DT'].transform(lambda x: x.fillna(method = 'ffill'))

# COMMAND ----------

df_iask_enabler['PASSED_DAYS_FROM_LAST_PYARR_TAKEN_TO_TODAY'] = (df_iask_enabler['TRANS_DT'] - df_iask_enabler['DRAGGED_PTP_TAKEN_DT']).dt.days
df_iask_enabler['PASSED_DAYS_FROM_LAST_PYARR_TAKEN_TO_TODAY'] = df_iask_enabler['PASSED_DAYS_FROM_LAST_PYARR_TAKEN_TO_TODAY'].fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pyarr indicator and cum indicator per delnq cycle

# COMMAND ----------

df_iask_enabler['PYARR_TAKEN_IND']=np.where(df_iask_enabler['PTP_TAKEN_DT'].isna(), 0, 1)

# COMMAND ----------

df_iask_enabler['CUM_PYARR_TAKEN_IND'] = df_iask_enabler.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])['PYARR_TAKEN_IND'].transform(lambda x: x.cumsum())
df_iask_enabler['CUM_PYARR_TAKEN_IND'] = df_iask_enabler['CUM_PYARR_TAKEN_IND'].fillna(0)

# COMMAND ----------

#print(df_iask_enabler.columns.tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Dragging last payment arrangements attributes

# COMMAND ----------

ptp_vars_cd = ['PTP_INI_CD', 'PTP_FIN_CD', 'PTP_SOURCE', 'PTP_INI_PRMS_METHOD', 'PTP_SUBSQT_PRMS_METHOD']
ptp_vars_dt = ['PTP_FINAL_DISPOSITION_DT','PTP_INIT_PRMS_DT','PTP_SUBSQT_PRMS_DT']
ptp_vars_num = ['PTP_TOT_PRMS_AMT','PTP_INI_PRMS_AMT','PTP_SUBSQT_PRMS_AMT']

# COMMAND ----------

for i in ptp_vars_cd:

    df_iask_enabler['DRAGGED_'+i] = df_iask_enabler.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])[i].transform(lambda x: x.fillna(method = 'ffill'))
    df_iask_enabler['DRAGGED_'+i] = df_iask_enabler['DRAGGED_'+i].fillna('NONE')

# COMMAND ----------

for i in ptp_vars_num:

    df_iask_enabler['DRAGGED_'+i] = df_iask_enabler.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])[i].transform(lambda x: x.fillna(method = 'ffill'))
    df_iask_enabler['DRAGGED_'+i] = df_iask_enabler['DRAGGED_'+i].fillna(0)

# COMMAND ----------

for i in ptp_vars_dt:

    df_iask_enabler['DRAGGED_'+i] = df_iask_enabler.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])[i].transform(lambda x: x.fillna(method = 'ffill'))

# COMMAND ----------

for i in ptp_vars_dt:
    
    df_iask_enabler['DAYS_TO_'+i] =  np.where(((df_iask_enabler['TRANS_DT'] <= df_iask_enabler['DRAGGED_'+i]) & (df_iask_enabler['DRAGGED_'+i].isna() == False)),
                                         (df_iask_enabler['DRAGGED_'+i]-df_iask_enabler['TRANS_DT']).dt.days,
                                            0)

# COMMAND ----------

#print(sorted(set(df_iask_enabler.columns.tolist()).difference(set(merged_df2_cut_tm_pymt_pyarr_bill_svy_IB.columns.tolist())), key=str.lower))

# COMMAND ----------

#merged_df2_cut_tm_pymt_pyarr_bill_svy_IB.columns

# COMMAND ----------

df_iask_enabler=df_iask_enabler[['TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END', #Critical actions
                                   'TRANS_DT',  'TRANS_DT_FAKE', 'BAN', 'DLNQ_DT_PACIFIC', 'TRANS_MONYR', 'DLNQ_DT_STR',
                                   'CURE_DT_PACIFIC', 'WO_DT', 'OCA_DT', 'LST_EVENT_DT',
                                   'DAYS_TO_EVENT', #Target
                                   'DAYS_IN_DLNQ',
                                   'LG_SUSP_DT', 'PROJ_SUSP_DT',
                                   'RCMD_PROJ_SUSP_DT', 'PAR_PROJ_SUSP_DT', 'INIT_REQ_DTTM', 'PAR_RCMND_DTTM', #PAR vars
                                   'BILL_DT', 'BILL_DUE_DT', 'PASSED_DAYS_FROM_BILL_DUE_DATE_TO_TRANS_DT', #Billing and related vars
                                   'BILL_TOT_DUE_AMT', 'UP_TO_TODAY_BILL_TOT_DUE_AMT', 'BILL_CURR_CHRGS_AMT', 'BILL_PAST_DUE_AMT', 'FIRST_FINAL_BILL_IND',
                                   'RISK_CD', 'INSTR_CD', # Strata and related vars
                                   'TOT_DUE_AMT', 'UP_TO_TODAY_TOT_DUE_AMT', 
                                   'TOT_DLNQ_AMT', 'UP_TO_TODAY_TOT_DLNQ_AMT',
                                   'T1', 'T2', 'T3', 'T4', 'T5', #Hold days BAU
                                   'TRANS_TYPE', 'TRANS_SUB_TYPE', 'TREAT_MSG_CD', #Messages and calls treatments
                                   'TREAT_MSG_CD_TIMECUTOFF', 'TREAT_DT_PACIFIC_CUTOFF', 'TREAT_DTTM_PACIFIC_CUTOFF', 
                                   'IB_CALL_TRANS_SUB_TYPE', 'IB_CALL_DT_TZ_PACIFIC', 'IB_CALL_IND', 'CUM_IB_CALL_IND', #IB calls 
                                   'fncl_trans_dt_tm_tz_pacific_PYMT', #Payments and realted vars
                                   'NUM_TRANSACTIONS_PYMT', 'PYMT_AMOUNT', 'CUM_PYMT_AMT_TO_TODAY', 
                                   'fncl_trans_method_cd_PYMT', 'fncl_trans_channel_cd_PYMT',
                                   'LAST_PYMT_DT', 'PASSED_DAYS_FROM_LAST_PYMT_TO_TODAY',
                                   'TOT_DUE_AMT_TO_CUM_PYMT_TO_TODAY_RTO', 'BILL_TOT_DUE_AMT_TO_CUM_PYMT_TO_TODAY_RTO',
                                   'fncl_trans_dt_tm_tz_pacific_ADJ',   #Adjustments and related vars                                    
                                   'NUM_TRANSACTIONS_ADJ', 'ADJ_AMOUNT', 'CUM_ADJ_AMT_TO_TODAY', 
                                   'fncl_trans_method_cd_ADJ', 'fncl_trans_channel_cd_ADJ', 
                                   'fncl_trans_dt_tm_tz_pacific_BCK', #BCK and related vars  
                                   'NUM_TRANSACTIONS_BCK', 'BCK_AMOUNT', 'CUM_BCK_AMT_TO_TODAY', 
                                   'fncl_trans_method_cd_BCK', 'fncl_trans_channel_cd_BCK', 
                                   'fncl_trans_dt_tm_tz_pacific_LPC', #last payment charge and related vars
                                   'NUM_TRANSACTIONS_LPC', 'LPC_AMOUNT', 'CUM_LPC_AMT_TO_TODAY', 
                                   'fncl_trans_method_cd_LPC', 'fncl_trans_channel_cd_LPC',
                                   'fncl_trans_dt_tm_tz_pacific_OTHER', #other charges and related vars
                                   'NUM_TRANSACTIONS_OTHER', 'OTHER_AMOUNT', 'CUM_OTHER_AMT_TO_TODAY',
                                   'fncl_trans_method_cd_OTHER', 'fncl_trans_channel_cd_OTHER',
                                   'PTP_TAKEN_DT', 'DRAGGED_PTP_TAKEN_DT', 'PRMS_TAKE_TIME', #pyarr and related vars                                   
                                   'PTP_TAKEN_DTTM', 'PTP_TIMEZONE', 'PTP_TAKEN_DTTM_PACIFIC',
                                   'PYARR_TAKEN_IND', 'CUM_PYARR_TAKEN_IND', 'PASSED_DAYS_FROM_LAST_PYARR_TAKEN_TO_TODAY',
                                   'PTP_SOURCE', 'DRAGGED_PTP_SOURCE', 'PTP_TOT_PRMS_AMT', 'DRAGGED_PTP_TOT_PRMS_AMT',
                                   'PTP_INIT_PRMS_DT', 'DRAGGED_PTP_INIT_PRMS_DT', 'DAYS_TO_PTP_INIT_PRMS_DT', 'PTP_INI_PRMS_AMT', 'DRAGGED_PTP_INI_PRMS_AMT', 'PTP_INI_PRMS_METHOD', 'DRAGGED_PTP_INI_PRMS_METHOD', 'PTP_INI_CD', 'DRAGGED_PTP_INI_CD',
                                   'PTP_SUBSQT_PRMS_DT', 'DRAGGED_PTP_SUBSQT_PRMS_DT', 'DAYS_TO_PTP_SUBSQT_PRMS_DT', 'PTP_SUBSQT_PRMS_AMT', 'DRAGGED_PTP_SUBSQT_PRMS_AMT', 'PTP_SUBSQT_PRMS_METHOD', 'DRAGGED_PTP_SUBSQT_PRMS_METHOD',
                                   'PTP_FINAL_DISPOSITION_DT', 'DRAGGED_PTP_FINAL_DISPOSITION_DT', 'DAYS_TO_PTP_FINAL_DISPOSITION_DT', 'PTP_FIN_CD', 'DRAGGED_PTP_FIN_CD',
                                   'SVY_SAT_SCORE', 'SVY_WTR_SCORE', 'SVY_CALL_CENTER_CD']] #Survey vars

# COMMAND ----------

#df_iask_enabler[df_iask_enabler['BAN']=='100142994']

# COMMAND ----------

df_iask_enabler.shape

# COMMAND ----------

#AQUI

# COMMAND ----------

df_iask_enabler.TREAT_MSG_CD = df_iask_enabler.TREAT_MSG_CD.replace(regex=[' '], value='_')

# COMMAND ----------

df_iask_enabler[df_iask_enabler['BAN']=='101586080']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refreshing Grid Treat Msg Code variables (Calls and Mesagge codes)

# COMMAND ----------

TREAT_CD_COMBO_DF = df_iask_enabler[['TRANS_DT', 'TRANS_MONYR', 'BAN', 'RISK_CD', 'DLNQ_DT_PACIFIC', 'TREAT_MSG_CD']]

# COMMAND ----------

#TREAT_CD_COMBO_DF.head(10)

# COMMAND ----------

#TREAT_CD_COMBO_DF['LAG_TREAT_MSG_CD'] = TREAT_CD_COMBO_DF.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])['TREAT_MSG_CD'].shift(1)

# COMMAND ----------

#TREAT_CD_COMBO_DF['LAG_TREAT_MSG_CD'] = TREAT_CD_COMBO_DF['LAG_TREAT_MSG_CD'].fillna('NONE')

# COMMAND ----------

#TREAT_CD_COMBO_DF['LAG_TREAT_MSG_CD'] = TREAT_CD_COMBO_DF['LAG_TREAT_MSG_CD'].replace('NONE', 'A')
TREAT_CD_COMBO_DF['TREAT_MSG_CD'] = TREAT_CD_COMBO_DF['TREAT_MSG_CD'].replace('NONE', 'A')

# COMMAND ----------

#TREAT_CD_COMBO_DF['JOINED_LAG_TREAT_MSG_CD'] = TREAT_CD_COMBO_DF.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])['LAG_TREAT_MSG_CD'].transform(lambda x : ' & '.join(x))
TREAT_CD_COMBO_DF['JOINED_TREAT_MSG_CD'] = TREAT_CD_COMBO_DF.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])['TREAT_MSG_CD'].transform(lambda x : ' & '.join(x))

# COMMAND ----------

TREAT_CD_COMBO_DF.head()

# COMMAND ----------

TREAT_CD_COMBO_DF[TREAT_CD_COMBO_DF['BAN']=='101586080']

# COMMAND ----------

TREAT_CD_COMBO_DF.shape

# COMMAND ----------

dup = TREAT_CD_COMBO_DF[TREAT_CD_COMBO_DF.duplicated(['TRANS_DT', 'BAN'])]
len(dup) #Revisar pyarrs duplicates

# COMMAND ----------

df_iask_cut_tm_07 = pd.merge(df_iask_enabler,
                             TREAT_CD_COMBO_DF[['TRANS_DT', 'BAN', 'RISK_CD', 'DLNQ_DT_PACIFIC', 'JOINED_TREAT_MSG_CD']],
                             how='left',
                             left_on=['TRANS_DT', 'BAN', 'RISK_CD', 'DLNQ_DT_PACIFIC',],
                             right_on=['TRANS_DT', 'BAN', 'RISK_CD', 'DLNQ_DT_PACIFIC',])

# COMMAND ----------

df_iask_cut_tm_07.shape

# COMMAND ----------

df_iask_cut_tm_07_c = df_iask_cut_tm_07[['TRANS_DT', 'BAN', 'RISK_CD', 'DLNQ_DT_PACIFIC', 'DAYS_IN_DLNQ', 'CURE_DT_PACIFIC',
                                         'TREAT_MSG_CD', 'JOINED_TREAT_MSG_CD']]

# COMMAND ----------

df_iask_cut_tm_07_c['CUT_JOINED_TREAT_MSG_CD'] = df_iask_cut_tm_07_c.apply(lambda x: " & ".join([i for i in x.JOINED_TREAT_MSG_CD.split() if i != '&'][0:(x.DAYS_IN_DLNQ+1)]), axis=1)

# COMMAND ----------

df_iask_cut_tm_07_c['CUT_JOINED_TREAT_MSG_CD'] = df_iask_cut_tm_07_c['CUT_JOINED_TREAT_MSG_CD'].replace("", 'A')

# COMMAND ----------

df_iask_cut_tm_07_c['STRIP_CUT_JOINED_TREAT_MSG_CD'] = df_iask_cut_tm_07_c.apply(lambda x: x['CUT_JOINED_TREAT_MSG_CD'].strip('A & '), axis=1)
df_iask_cut_tm_07_c['STRIP_CUT_JOINED_TREAT_MSG_CD'] = df_iask_cut_tm_07_c.apply(lambda x: x['STRIP_CUT_JOINED_TREAT_MSG_CD'].strip(' & A'), axis=1)
df_iask_cut_tm_07_c['STRIP_CUT_JOINED_TREAT_MSG_CD'] = df_iask_cut_tm_07_c['STRIP_CUT_JOINED_TREAT_MSG_CD'].str.replace("& A ", "")

# COMMAND ----------

df_iask_cut_tm_07_c['STRIP_CUT_JOINED_TREAT_MSG_CD'] = df_iask_cut_tm_07_c['STRIP_CUT_JOINED_TREAT_MSG_CD'].replace("", 'A')

# COMMAND ----------

df_iask_cut_tm_07_c.columns

# COMMAND ----------

df_iask_cut_tm_07_c.columns = ['TRANS_DT', 'BAN', 'RISK_CD', 'DLNQ_DT_PACIFIC', 'DAYS_IN_DLNQ', 'CURE_DT_PACIFIC',
       'TREAT_MSG_CD', 'TREAT_MSG_CD_COMBO', 'TREAT_MSG_CD_COMBO_FROM_DLNQ_TO_TDY', 'TREAT_MSG_CD_COMBO_STRIPPED_FROM_DLNQ_TO_TDY']

# COMMAND ----------

#df_iask_cut_tm_07_c[df_iask_cut_tm_07_c['BAN']=='100142994']

# COMMAND ----------

df_iask_cut_tm_07_vf = df_iask_cut_tm_07_c.copy()

# COMMAND ----------

df_iask_cut_tm_07_vf = df_iask_cut_tm_07_vf[['TRANS_DT', 'BAN', 'RISK_CD', 'DLNQ_DT_PACIFIC', 'TREAT_MSG_CD_COMBO',
                                             'TREAT_MSG_CD_COMBO_FROM_DLNQ_TO_TDY', 'TREAT_MSG_CD_COMBO_STRIPPED_FROM_DLNQ_TO_TDY']]

# COMMAND ----------

df_iask_cut_tm_07_vf.shape

# COMMAND ----------

df_iask_enabler.shape

# COMMAND ----------

df_iask_enabler = pd.merge(df_iask_enabler,
                             df_iask_cut_tm_07_vf,
                             how='left',
                             left_on=['TRANS_DT', 'BAN', 'RISK_CD', 'DLNQ_DT_PACIFIC'],
                             right_on=['TRANS_DT', 'BAN', 'RISK_CD', 'DLNQ_DT_PACIFIC'])

# COMMAND ----------

df_iask_enabler.shape

# COMMAND ----------

# MAGIC %md
# MAGIC #### Remaining vars

# COMMAND ----------

for i in range(1,8):
    df_iask_enabler['TREAT_MSG_CD_LAG_D'+str(i)] = df_iask_enabler.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])['TREAT_MSG_CD'].shift(i)
    df_iask_enabler['TREAT_MSG_CD_LAG_D'+str(i)] = df_iask_enabler['TREAT_MSG_CD_LAG_D'+str(i)].fillna('NONE')

# COMMAND ----------

cols_treat = ['TREAT_MSG_CD_LAG_D7','TREAT_MSG_CD_LAG_D6','TREAT_MSG_CD_LAG_D5','TREAT_MSG_CD_LAG_D4','TREAT_MSG_CD_LAG_D3','TREAT_MSG_CD_LAG_D2','TREAT_MSG_CD_LAG_D1']
     
df_iask_enabler['FULL_PATH_COMBO_FROM_D7_TO_YTD'] = df_iask_enabler[cols_treat].apply(lambda x: '|'.join(x.values.astype(str)), axis=1)

# COMMAND ----------

df_iask_enabler.columns[:-5]

# COMMAND ----------



# COMMAND ----------

df_iask_enabler['STP_PATH_COMBO_FROM_D7_TO_YTD'] = df_iask_enabler.apply(lambda x: x.FULL_PATH_COMBO_FROM_D7_TO_YTD.replace('NONE|', ''), axis=1)

# COMMAND ----------

df_iask_enabler['STP_PATH_COMBO_FROM_D7_TO_YTD'] = df_iask_enabler.apply(lambda x: x.STP_PATH_COMBO_FROM_D7_TO_YTD.replace('|NONE', ''), axis=1)

# COMMAND ----------

df_iask_enabler['TREAT_MSG_CD_COMBO_FROM_DLNQ_TO_TDY'].head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Days to susp

# COMMAND ----------

df_iask_enabler['DAYS_TO_SUSP']=(df_iask_enabler['LG_SUSP_DT'] - df_iask_enabler['TRANS_DT']).dt.days

# COMMAND ----------

print(df_iask_enabler.columns.tolist())

# COMMAND ----------

df_iask_enabler.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # <span style='color:Yellow'>NOTE: TBD derived variables for message codes</span> 

# COMMAND ----------

df_iask_enabler = df_iask_enabler[['TRANS_DTTM_PACIFIC_START', 'TRANS_DTTM_PACIFIC_END', #Critical actions
                                   'TRANS_DT',  'TRANS_DT_FAKE', 'BAN', 'DLNQ_DT_PACIFIC', 'TRANS_MONYR', 'DLNQ_DT_STR',
                                   'CURE_DT_PACIFIC', 'WO_DT', 'OCA_DT', 'LST_EVENT_DT',
                                   'DAYS_TO_EVENT', #Target
                                   'DAYS_IN_DLNQ', 'DAYS_TO_SUSP',
                                   'LG_SUSP_DT', 'PROJ_SUSP_DT',
                                   'RCMD_PROJ_SUSP_DT', 'PAR_PROJ_SUSP_DT', 'INIT_REQ_DTTM', 'PAR_RCMND_DTTM', #PAR vars
                                   'BILL_DT', 'BILL_DUE_DT', 'PASSED_DAYS_FROM_BILL_DUE_DATE_TO_TRANS_DT', #Billing and related vars
                                   'BILL_TOT_DUE_AMT', 'UP_TO_TODAY_BILL_TOT_DUE_AMT', 'BILL_CURR_CHRGS_AMT', 'BILL_PAST_DUE_AMT', 'FIRST_FINAL_BILL_IND',
                                   'RISK_CD', 'INSTR_CD', # Strata and related vars
                                   'TOT_DUE_AMT', 'UP_TO_TODAY_TOT_DUE_AMT', 
                                   'TOT_DLNQ_AMT', 'UP_TO_TODAY_TOT_DLNQ_AMT',
                                   'T1', 'T2', 'T3', 'T4', 'T5', #Hold days BAU
                                   'TRANS_TYPE', 'TRANS_SUB_TYPE', 'TREAT_MSG_CD', #Messages and calls treatments
                                   'TREAT_MSG_CD_COMBO', 'TREAT_MSG_CD_COMBO_FROM_DLNQ_TO_TDY', 'TREAT_MSG_CD_COMBO_STRIPPED_FROM_DLNQ_TO_TDY',
                                   'TREAT_MSG_CD_LAG_D1', 'TREAT_MSG_CD_LAG_D2', 'TREAT_MSG_CD_LAG_D3', 'TREAT_MSG_CD_LAG_D4',
                                   'TREAT_MSG_CD_LAG_D5', 'TREAT_MSG_CD_LAG_D6', 'TREAT_MSG_CD_LAG_D7',
                                   'STP_PATH_COMBO_FROM_D7_TO_YTD', 'FULL_PATH_COMBO_FROM_D7_TO_YTD',
                                   'TREAT_MSG_CD_TIMECUTOFF', 'TREAT_DT_PACIFIC_CUTOFF', 'TREAT_DTTM_PACIFIC_CUTOFF', 
                                   'IB_CALL_TRANS_SUB_TYPE', 'IB_CALL_DT_TZ_PACIFIC', 'IB_CALL_IND', 'CUM_IB_CALL_IND', #IB calls 
                                   'fncl_trans_dt_tm_tz_pacific_PYMT', #Payments and realted vars
                                   'NUM_TRANSACTIONS_PYMT', 'PYMT_AMOUNT', 'CUM_PYMT_AMT_TO_TODAY', 
                                   'fncl_trans_method_cd_PYMT', 'fncl_trans_channel_cd_PYMT',
                                   'LAST_PYMT_DT', 'PASSED_DAYS_FROM_LAST_PYMT_TO_TODAY',
                                   'TOT_DUE_AMT_TO_CUM_PYMT_TO_TODAY_RTO', 'BILL_TOT_DUE_AMT_TO_CUM_PYMT_TO_TODAY_RTO',
                                   'fncl_trans_dt_tm_tz_pacific_ADJ',   #Adjustments and related vars                                    
                                   'NUM_TRANSACTIONS_ADJ', 'ADJ_AMOUNT', 'CUM_ADJ_AMT_TO_TODAY', 
                                   'fncl_trans_method_cd_ADJ', 'fncl_trans_channel_cd_ADJ', 
                                   'fncl_trans_dt_tm_tz_pacific_BCK', #BCK and related vars  
                                   'NUM_TRANSACTIONS_BCK', 'BCK_AMOUNT', 'CUM_BCK_AMT_TO_TODAY', 
                                   'fncl_trans_method_cd_BCK', 'fncl_trans_channel_cd_BCK', 
                                   'fncl_trans_dt_tm_tz_pacific_LPC', #last payment charge and related vars
                                   'NUM_TRANSACTIONS_LPC', 'LPC_AMOUNT', 'CUM_LPC_AMT_TO_TODAY', 
                                   'fncl_trans_method_cd_LPC', 'fncl_trans_channel_cd_LPC',
                                   'fncl_trans_dt_tm_tz_pacific_OTHER', #other charges and related vars
                                   'NUM_TRANSACTIONS_OTHER', 'OTHER_AMOUNT', 'CUM_OTHER_AMT_TO_TODAY',
                                   'fncl_trans_method_cd_OTHER', 'fncl_trans_channel_cd_OTHER',
                                   'PTP_TAKEN_DT', 'DRAGGED_PTP_TAKEN_DT', 'PRMS_TAKE_TIME', #pyarr and related vars                                   
                                   'PTP_TAKEN_DTTM', 'PTP_TIMEZONE', 'PTP_TAKEN_DTTM_PACIFIC',
                                   'PYARR_TAKEN_IND', 'CUM_PYARR_TAKEN_IND', 'PASSED_DAYS_FROM_LAST_PYARR_TAKEN_TO_TODAY',
                                   'PTP_SOURCE', 'DRAGGED_PTP_SOURCE', 'PTP_TOT_PRMS_AMT', 'DRAGGED_PTP_TOT_PRMS_AMT',
                                   'PTP_INIT_PRMS_DT', 'DRAGGED_PTP_INIT_PRMS_DT', 'DAYS_TO_PTP_INIT_PRMS_DT', 'PTP_INI_PRMS_AMT', 'DRAGGED_PTP_INI_PRMS_AMT', 'PTP_INI_PRMS_METHOD', 'DRAGGED_PTP_INI_PRMS_METHOD', 'PTP_INI_CD', 'DRAGGED_PTP_INI_CD',
                                   'PTP_SUBSQT_PRMS_DT', 'DRAGGED_PTP_SUBSQT_PRMS_DT', 'DAYS_TO_PTP_SUBSQT_PRMS_DT', 'PTP_SUBSQT_PRMS_AMT', 'DRAGGED_PTP_SUBSQT_PRMS_AMT', 'PTP_SUBSQT_PRMS_METHOD', 'DRAGGED_PTP_SUBSQT_PRMS_METHOD',
                                   'PTP_FINAL_DISPOSITION_DT', 'DRAGGED_PTP_FINAL_DISPOSITION_DT', 'DAYS_TO_PTP_FINAL_DISPOSITION_DT', 'PTP_FIN_CD', 'DRAGGED_PTP_FIN_CD',
                                   'SVY_SAT_SCORE', 'SVY_WTR_SCORE', 'SVY_CALL_CENTER_CD']] #Survey vars]]

# COMMAND ----------

df_iask_enabler.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # Join Cohort remaining Info

# COMMAND ----------

df_coh.shape

# COMMAND ----------

df_coh2 = df_coh.copy()

# COMMAND ----------

df_coh2.rename(columns={'acct_nbr':'BAN', 'cohort_data_capture_dt':'DLNQ_DT_STR'}, inplace=True)

# COMMAND ----------

df_coh2.BAN = df_coh2.BAN.astype('str')

# COMMAND ----------

df_coh2 = df_coh2[['BAN', 'DLNQ_DT_STR', 'acct_orgnl_srv_dt', 'account_type', 'ism_ind',
       'paperless_bill_ind', 'auto_bill_pay_ind', 'email_addr_avail_ind',
       'msg_device_avail_ind', 'credit_risk_segment', 'state_cd',
       'telco_region_cd', 'mobility_market_cd', 'mobility_region_name',
       'mobility_cluster_name', 'segment_cd', 'wrls_unfd_standalone_ind',
       'NBR_OF_LN_CNT', 'lblty_type_cd', 'nbi_ind', 'digital_life_ind',
       'firstnet_ind', 'connected_car_ind', 'next_plan_active_ind',
       'acc_plan_active_ind', 'Device_Mobile_Voice_Ind', 'Device_Tablet_Ind',
       'Device_Watch_Ind', 'Device_Other_Ind', 'dtv_standalone_ind',
       'video_type_grp', 'voice_type_grp', 'internet_type_grp',
       'wireless_type_grp', 'pots_service_ind', 'dry_loop_ind', 'PREV_3_MO_AVG_tot_due_amt', 'PREV_3_MO_AVG_curr_chrgs_amt',
       'PREV_3_MO_AVG_past_due_amt', 'PREV_3_MO_NBR_PAYMENT',
       'PREV_3_MO_NBR_ADJUSTMENT', 'PREV_3_MO_NBR_CREDIT',
       'PREV_3_MO_TOT_PAYMENT_AMT', 'PREV_3_MO_TOT_ADJUSTMENT_AMT',
       'PREV_3_MO_TOT_CREDIT_AMT', 'PREV_3_MO_NBR_CNC_INB_CALL',
       'PREV_3_MO_NBR_CNC_OTB_CALL', 'PREV_3_MO_NBR_CNC_AOB_CALL',
       'PREV_3_MO_NBR_CARE_INB_CALL', 'PREV_3_MO_NBR_LETTER',
       'PREV_3_MO_NBR_EMAIL', 'PREV_3_MO_NBR_SMS']]

# COMMAND ----------

df_coh2.shape

# COMMAND ----------

df_iask_enabler[df_iask_enabler["BAN"].isin(['100142994', '100057601', '100184538', '120653395'])].groupby(by=['BAN', 'DLNQ_DT_STR']).size()

# COMMAND ----------

df_iask_enabler.shape

# COMMAND ----------

# check by Wen:
df_iask_enabler['DLNQ_DT_STR'].dtype, df_coh2['DLNQ_DT_STR'].dtype

# COMMAND ----------

# update by Wen, to convert the same datatype before merging, otherwise it would error out
df_coh2['DLNQ_DT_STR'] = df_coh2['DLNQ_DT_STR'].astype('datetime64[D]')

# COMMAND ----------

df_iask_enabler = pd.merge(df_iask_enabler,
                      df_coh2, 
                      how='left',
                      left_on=['BAN', 'DLNQ_DT_STR'],
                      right_on=['BAN', 'DLNQ_DT_STR'])

# COMMAND ----------

df_iask_enabler.shape

# COMMAND ----------

df_iask_enabler.head(100)

# COMMAND ----------

#df_iask_enabler.to_csv('df_iask_enabler_7_9_2021_WenCheck.csv')

# Note df_iask_enabler is the resulting df for current cohort data, which will be used in the following session below

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Session 2: Generate hist treatment df and create features for hist treatment <a name="session2"></a>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Session 2 Part 1: run part of Daan's code with 6 hist month cohorts to get hist treatment df <a name="subsession21"></a>

# COMMAND ----------

#Move this to the beginning of Session 2, df_model is the current cohort df from Daan's first Session code

# update with Daan's latest df, which can be replaced with resulting df from Daan's session 1 code after combined with Session 1
#df_model =  pd.read_csv('df_iask_enabler_7_9_2021.csv', index_col=0)

df_model = df_iask_enabler.copy()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reading input files *(none special order)*

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC # # 01-09 Data Frames WITH timestamp
# MAGIC # df_str = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File9_Strata_10k.csv', index_col=0) 
# MAGIC # df_coh = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File1_MainCohort_10k.csv', index_col=0) #(not used)
# MAGIC # df_act = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File5_CriticalActvt_10k.csv', index_col=0)
# MAGIC # df_calls = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File3_Calls_10k.csv', index_col=0)
# MAGIC # df_treatment_timestamp = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File4_Messages_10k.csv', index_col=0)
# MAGIC # df_pmt_timestamp = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File2_PayAdj_10k.csv', index_col=0)
# MAGIC # df_pa = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File8_Promises_10k.csv', index_col=0)
# MAGIC # df_billing = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File6_Bills_10k.csv', index_col=0)
# MAGIC # df_survey = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File7_Surveys_10k.csv', index_col=0)
# MAGIC 
# MAGIC # # PAR Files to get recommended suspension date if available
# MAGIC # #df_PAR_2019 = pd.read_csv('Enabler/ENBLR_iASCE Trial_Aug19_PAR_Recommender_10k.csv', index_col=0)
# MAGIC # #df_PAR_2020 = pd.read_csv('Enabler/ENBLR_iASCE Trial_Aug20_PAR_Recommender_10k.csv', index_col=0)
# MAGIC 
# MAGIC # #File with complete Strara Info, not used this time for Enabler
# MAGIC # #df_Full_Strata_2019 = pd.read_csv('.csv', index_col=0)
# MAGIC # #df_Full_Strata_2020 = pd.read_csv('.csv', index_col=0)
# MAGIC 
# MAGIC df_DATA01_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File9_Strata_10k.csv', header=True)
# MAGIC df_DATA02_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File1_MainCohort_10k.csv', header=True)
# MAGIC df_DATA03_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File5_CriticalActvt_10k.csv', header=True)
# MAGIC df_DATA04_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File3_Calls_10k.csv', header=True)
# MAGIC df_DATA05_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File4_Messages_10k.csv', header=True)
# MAGIC df_DATA06_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File2_PayAdj_10k.csv', header=True)
# MAGIC df_DATA07_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File8_Promises_10k.csv', header=True)
# MAGIC df_DATA08_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File6_Bills_10k.csv', header=True)
# MAGIC df_DATA09_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File7_Surveys_10k.csv', header=True)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC df_str = df_DATA01_Aug2020.toPandas()
# MAGIC df_coh = df_DATA02_Aug2020.toPandas()
# MAGIC df_act = df_DATA03_Aug2020.toPandas()
# MAGIC df_calls = df_DATA04_Aug2020.toPandas()
# MAGIC df_treatment_timestamp = df_DATA05_Aug2020.toPandas()
# MAGIC df_pmt_timestamp = df_DATA06_Aug2020.toPandas()
# MAGIC df_pa = df_DATA07_Aug2020.toPandas()
# MAGIC df_billing = df_DATA08_Aug2020.toPandas()
# MAGIC df_survey = df_DATA09_Aug2020.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## <span style='color:Yellow'> **Current Functions** <span/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### <span style='color:Brown'> **Function 1** <span/>

# COMMAND ----------

# Wen's function
# define a function to detect CLTN and Cure/Write-off/OCA pair per BAN, 
# output a list of row index that need to be dropped from dataframe CURES

ls_valid_crit_act_cd = ['CLTN', 'CURE', 'OCAE', 'WOFF']

def detect_rows_to_be_dropped(index_comb, crit_act_comb, ls_valid_crit_act_cd):
    ls_index_comb =[int(i.strip()) for i in index_comb.split(', ')]
    ls_crit_act_comb = crit_act_comb.split(', ')
    
    list_rows_to_be_dropped = []  #list to store the row index that needs to be removed from the dataframe
    list_valid_act_cd = []
    cltn_stack = [] # create a stack to pair up CLTN and its nearest CURE/Write-off/OCA
    for i, act_cd in enumerate(ls_crit_act_comb):
        if act_cd not in ls_valid_crit_act_cd: 
            list_rows_to_be_dropped.append(ls_index_comb[i])
        elif act_cd == 'CLTN' and not cltn_stack:
            list_valid_act_cd.append('CLTN')
            cltn_stack.append(('CLTN', ls_index_comb[i])) #find the first 'CLTN' code as well as its row index
        elif act_cd == 'CLTN' and cltn_stack: #another 'CLTN' comes after
            list_rows_to_be_dropped.append(cltn_stack.pop()[1]) #put the ealier CLTN index into drop list
            cltn_stack.append(('CLTN', ls_index_comb[i]))  # update the stack with the latest CLTN           
        elif act_cd !='CLTN' and cltn_stack:
            cltn_stack.pop()  # clear this stack once a matched Cure/Write-off/OCA is found
            list_valid_act_cd.append(act_cd)
        elif act_cd !='CLTN' and not cltn_stack:  #no 'CLTN' to be paired with
            list_rows_to_be_dropped.append(ls_index_comb[i])
     
    if cltn_stack: # still have 'CLTN' left in the stack, without any Cure/Write-off/OCA to be matched with
        # remove this row since no valid code found to pair with this CLTN
        list_rows_to_be_dropped.append((cltn_stack.pop()[1]))
        list_valid_act_cd.pop()
    
    return (list_rows_to_be_dropped, list_valid_act_cd)
            
            
# test an example:
detect_rows_to_be_dropped('0, 1, 2', 'CLTN, CURE, CURE', ls_valid_crit_act_cd)


# COMMAND ----------

# MAGIC %md
# MAGIC ### <span style='color:Brown'> **Function 2** <span/>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Function  for critical actions, messages and calls to get datetime with time in proper format from original integer time variable

# COMMAND ----------

#Getting the date with timestamp and timezone
def new_datetime(df, time_variable, date_variable, timezone):
    
    time_variable = str(time_variable)
    date_variable = str(date_variable)
    
    timestamp = str(df[time_variable])
    
    if df['len_time'] == 6:
        timestamp = timestamp
    elif df['len_time'] == 5:
        timestamp = '0'+timestamp
    elif df['len_time'] == 4:
        timestamp = '00'+timestamp
    elif df['len_time'] == 3:
        timestamp = '000'+timestamp
    elif df['len_time'] == 2:
        timestamp = '0000'+timestamp
    else:
        timestamp = '00000'+timestamp
    
    a = int(timestamp[0:2])
    b = int(timestamp[2:4])
    c = int(timestamp[4:6])
    
    tm1_op2 = time(a, b, c)    
    new_date = datetime.combine(df[date_variable], tm1_op2)    
    new_date_tz = timezone.localize(new_date, is_dst=True)
    
    return new_date_tz

# COMMAND ----------

# MAGIC %md
# MAGIC ### <span style='color:Brown'> **Function 3** <span/>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Function for payments to add different timezones and convert to pacific timezone

# COMMAND ----------

#Based on timezone add to the datetime variable
def add_timezone(df, timezone_variable, date_variable):
    
    tm_fake = time(0, 0, 0)
    
    timezone_variable = str(timezone_variable)
    date_variable = str(date_variable)
    date_variable_tz = str(date_variable+'_tz')
    date_variable_tz_pacific = str(date_variable_tz+'_pacific')
    
    df[date_variable_tz] = df.apply(lambda x: pd.to_datetime(x[date_variable], format="%m/%d/%Y %H:%M:%S")
                                    if (x['fncl_trans_type_cd'] == 'PYM')
                                    else datetime.combine(x.fncl_trans_dt, tm_fake), axis=1)


    def assess_tz(row, row2, value):
        if row == 'EST':
            value = eastern_tz.localize(value, is_dst=True)
            return value
        elif row == 'PST':
            value = pacific_tz.localize(value, is_dst=True)  
            return value
        elif ((row == 'CST') | (row2 == 'ADJ')):
            value = central_tz.localize(value, is_dst=True) 
            return value
        elif row == 'MST':
            value = mountain_tz.localize(value, is_dst=True)
            return value
        else:
            value = pd.NaT
            return value

    df[date_variable_tz] = df.apply(lambda x: assess_tz(x[timezone_variable], x['fncl_trans_type_cd'], x[date_variable_tz]), axis=1)
    df['pmt_timezone'] = df.apply(lambda x: x[date_variable_tz].tzinfo, axis=1)    
    df[date_variable_tz_pacific] = df.apply(lambda x: x[date_variable_tz].astimezone(pacific_tz), axis=1)
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ### <span style='color:Brown'> **Function 4** <span/>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Function to add laboral days to calculate projected suspension date

# COMMAND ----------

def adding_business_days(from_date, add_days):
    business_days_to_add = add_days
    current_date = from_date
    while business_days_to_add > 0:
        current_date += timedelta(days=1)
        weekday = current_date.weekday()
        if weekday >= 5: # sunday = 6
            continue
        if current_date in us_holidays:
            continue
        business_days_to_add -= 1
    return current_date

#demo:
print('10 business days from today:')
print(date.today())
print(adding_business_days(date.today(), 5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### <span style='color:Brown'> **Function 5** <span/>

# COMMAND ----------

#Adding days to suspension and projected suspension date to Strata  OR TREATMENTS BEFOR SUSP
def timeline_days(risk, stage):
    if stage == 1:
        return df_timeline_susp.loc[risk]['Frie_SMS_Email_Days']
    elif stage == 2:
        return df_timeline_susp.loc[risk]['Ent_Dlnq_SMS_Days']
    elif stage == 3:
        return df_timeline_susp.loc[risk]['PreSusp_SMS_Email_Days']
    elif stage == 4:
        return df_timeline_susp.loc[risk]['PreSusp_Lett_Days']
    else:
        return df_timeline_susp.loc[risk]['Susp_Days']

# COMMAND ----------

# MAGIC %md
# MAGIC ### <span style='color:Brown'> **Function 6** <span/>

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>  Below cell of code has three functions that will be used later by the module
# MAGIC </h3>    <ol>  
# MAGIC        <li> create_date_range  : takes 2 inputs "start_dt and end_dt"
# MAGIC              The function will create a date range wrt the start and end dates </li>
# MAGIC        <li> create_new_df      : takes input as a basic dataframe with  
# MAGIC            <ol>
# MAGIC            <li> BAN numbers </li>
# MAGIC            <li> SUSP_DT </li>
# MAGIC            <li> DLNQ_DT </li>
# MAGIC            <li> MONTHYR </li>
# MAGIC            </ol>
# MAGIC         and creates a new dataframe for all the sample BAN's based on months the customer transacted on.
# MAGIC         </li>
# MAGIC         <li> fit_prev_data :: The function takes the older dataframe from input step and does a merge with 
# MAGIC              the newer data with multiple date rows (date range from DLNQ till SUSP)
# MAGIC         </li>
# MAGIC                    
# MAGIC                

# COMMAND ----------

## Function to generate missing date rows between on START_DT and END_DT

def create_date_range(start_dt,end_dt):
    date_range_datetime=0
    date_index=0
    date_range_datetime = pd.date_range(start= start_dt, end=end_dt, freq='D', )
    date_index = list(date_range_datetime.strftime('%Y-%m-%d').values)
#     print("create date range function")
    return date_index
    

## Function to create dataframe with n rows [TRANS_DT range returned  from above function call] for BAN number with additional features like CRITICAL ACTION fields 
## ['BAN','DLNQ_DT','SUSP_DT','CRIT_ACT_DT','CRIT_ACT_TYPE_CD','TOTAL_DUE_AMT', 'last_cltn_risk_segment_cd','MONTHYR']
## Will try to get the hardcoding of column names automated
## The input to the below function will be a dataframe with sample BAN's
## DF structure :   BAN | DLNQ_DT | SUSP_DT | CRIT_ACT_DT | CRIT_ACT_TYPE_CD | TOTAL_DUE_AMT | last_cltn_risk_segment_cd |TRANS_MONY

def create_new_df(df):
    
    new_df = pd.DataFrame()
    df_group=df.groupby(['BAN', 'RISK_CD', 'INSTR_CD', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR', 'CURE_DT_PACIFIC', 'LG_SUSP_DT',
                       'LST_EVENT_DT', 'DAYS_TO_EVENT', 'TOT_DUE_AMT', 'TOT_DLNQ_AMT', 'T1', 'T2', 'T3', 'T4', 'T5']
                        , as_index = False).agg({'TRANS_MONYR': ' '.join})
    df_group.columns=['BAN', 'RISK_CD', 'INSTR_CD', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR', 'CURE_DT_PACIFIC', 'LG_SUSP_DT',
                       'LST_EVENT_DT', 'DAYS_TO_EVENT', 'TOT_DUE_AMT', 'TOT_DLNQ_AMT',
                      'T1', 'T2', 'T3', 'T4', 'T5','MONTHYR']
    first_time_counter=0
    
    for idx in tqdm(range(0, len(df_group.index))):
            temp_df=pd.DataFrame()
            start_of_mon=df_group.DLNQ_DT_PACIFIC[idx] 
            end_of_mon = min(df_group.LST_EVENT_DT[idx], df_group.LG_SUSP_DT[idx])
            temp_df['TRANS_DT'] = create_date_range(start_of_mon,end_of_mon)
            temp_df['BAN']=df_group.BAN[idx]
            temp_df['RISK_CD']=df_group.RISK_CD[idx]
            temp_df['INSTR_CD']=df_group.INSTR_CD[idx]
            temp_df['DLNQ_DT_PACIFIC']=df_group.DLNQ_DT_PACIFIC[idx]
            temp_df['DLNQ_DT_STR']=df_group.DLNQ_DT_STR[idx]
            temp_df['CURE_DT_PACIFIC']=df_group.CURE_DT_PACIFIC[idx]
            temp_df['LG_SUSP_DT']=df_group.LG_SUSP_DT[idx]
            temp_df['LST_EVENT_DT']=df_group.LST_EVENT_DT[idx]
            temp_df['DAYS_TO_EVENT']=df_group.DAYS_TO_EVENT[idx] 
            temp_df['TOT_DUE_AMT']=df_group.TOT_DUE_AMT[idx]
            temp_df['TOT_DLNQ_AMT']=df_group.TOT_DLNQ_AMT[idx]            
            temp_df['T1']=df_group.T1[idx]
            temp_df['T2']=df_group.T2[idx]
            temp_df['T3']=df_group.T3[idx]            
            temp_df['T4']=df_group.T4[idx]
            temp_df['T5']=df_group.T5[idx]            
            
            if(first_time_counter==0):
                new_df=temp_df
                first_time_counter+=1
            else:
                new_df = new_df.append(temp_df)            
    return new_df

## Function to help merge the 
##     a) newly created dataset with continuous TRANS_DT between the delinquent date and suspension date 
## and b) old dataframe with additional features [dummy based on TRANS_TYPE and TRANS_SUB_TYPE] 

def fit_prev_data(df_old, df_new):
    
    print(df_old.columns)
    print(df_new.columns)
    
    df_analysis=pd.DataFrame()
    
    df_analysis = df_new.merge(df_old,
                               how='left',
                               on=['BAN', 'DLNQ_DT_PACIFIC', 'TRANS_DT'],
                               indicator = True)

    df_analysis.sort_values(by=['BAN', 'DLNQ_DT_PACIFIC'],inplace=True)

    #df_analysis = df_analysis[(df_analysis["TRANS_DT"] >= df_analysis["DLNQ_DT"]) & (df_analysis["TRANS_DT"] <= df_analysis["DLNQ_DT"]+timedelta(days=int(29)))]
    
    df_analysis.reset_index(drop=True, inplace=True)
    #df_analysis.drop(['TRANS_MONYR','_merge'],inplace=True, axis=1)
    return df_analysis


# COMMAND ----------

# MAGIC %md
# MAGIC ## Part l. Get historical account cycles from Feb-Jul 19/20

# COMMAND ----------

df_action = df_act[['acct_nbr', 'critical_action_dt', 'critical_action_type_cd', 'critical_action_ar_tot_due_amt']]
df_action.columns = ['BAN', 'CRIT_ACT_DT', 'CRIT_ACT_TYPE_CD', 'AR_TOTAL_DUE_AMT']

df_action['BAN'] = df_action['BAN'].astype(str)
df_action['TRANS_MONYR']=pd.to_datetime(df_action['CRIT_ACT_DT']).dt.strftime('%Y%m')
df_action['TRANS_MONYR'] = df_action['TRANS_MONYR'].astype(int)
df_action['CRIT_ACT_DT'] = df_action['CRIT_ACT_DT'].astype('datetime64[D]')
df_action['CRIT_ACT_TYPE_CD'] = df_action['CRIT_ACT_TYPE_CD'].str.strip()

df_action = df_action.reset_index(drop=True)

df_action = df_action.sort_values(by=['BAN','CRIT_ACT_DT','TRANS_MONYR'])

# COMMAND ----------

#%%time

s = df_action.groupby('TRANS_MONYR').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s.pivot_table(index = ['TRANS_MONYR'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Keep cohort periods with 3 months performance window

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC cohort_months = [201902,201903,201904,201905,201906, 201907, 201908, 201909, 201910,
# MAGIC                  202002,202003,202004,202005,202006, 202007, 202008, 202009, 202010]
# MAGIC 
# MAGIC df_action = df_action[df_action["TRANS_MONYR"].isin(cohort_months)]

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC s = df_action.groupby('TRANS_MONYR').size().reset_index(name='FREQ')
# MAGIC s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
# MAGIC s.pivot_table(index = ['TRANS_MONYR'],
# MAGIC              margins = True, 
# MAGIC              margins_name='Total',
# MAGIC              aggfunc=sum)

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC cohort_aug = [201908, 202008]
# MAGIC 
# MAGIC Accts_201908 = df_action.BAN[df_action["TRANS_MONYR"] == cohort_aug[0]].unique().tolist()
# MAGIC 
# MAGIC Cohort_201908_hist = df_action[(df_action["TRANS_MONYR"].isin(cohort_months[0:9])) &
# MAGIC                           (df_action["BAN"].isin(Accts_201908))]
# MAGIC 
# MAGIC Accts_202008 = df_action.BAN[df_action["TRANS_MONYR"] == cohort_aug[1]].unique().tolist()
# MAGIC 
# MAGIC Cohort_202008_hist = df_action[(df_action["TRANS_MONYR"].isin(cohort_months[9:18])) &
# MAGIC                           (df_action["BAN"].isin(Accts_202008))]
# MAGIC 
# MAGIC print(len(Accts_201908))
# MAGIC print(len(Cohort_201908_hist))
# MAGIC print(len(Accts_202008))
# MAGIC print(len(Cohort_202008_hist))

# COMMAND ----------

Cohort_Aug_H = pd.concat([Cohort_201908_hist, Cohort_202008_hist], axis=0)

# COMMAND ----------

s = Cohort_Aug_H.groupby('TRANS_MONYR').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s.pivot_table(index = ['TRANS_MONYR'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

s = Cohort_Aug_H.groupby('CRIT_ACT_TYPE_CD').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s.pivot_table(index = ['CRIT_ACT_TYPE_CD'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

Cohort_Aug_H["CRIT_ACT_TYPE_CD"].unique()

# COMMAND ----------

Cohort_Aug_H = Cohort_Aug_H.sort_values(['BAN','CRIT_ACT_DT'])

Cohort_Aug_H = Cohort_Aug_H.reset_index(drop=True)

# COMMAND ----------

Cohort_Aug_H['AR_TOTAL_DUE_AMT'] = pd.to_numeric(Cohort_Aug_H['AR_TOTAL_DUE_AMT'], errors='coerce')
Cohort_Aug_H['AR_TOTAL_DUE_AMT'].dtype

# COMMAND ----------

Cohort_Aug_H = Cohort_Aug_H.drop_duplicates()
print(Cohort_Aug_H.shape)

# COMMAND ----------

Cohort_Aug_H = Cohort_Aug_H.reset_index(drop=True)
Cohort_Aug_H.head()

# COMMAND ----------

Cohort_Aug_H.CRIT_ACT_DT = Cohort_Aug_H.CRIT_ACT_DT.astype('datetime64[D]')
Cohort_Aug_H.CRIT_ACT_DT.dtype

# COMMAND ----------

Cohort_Aug_H['TRANS_YEAR'] = Cohort_Aug_H['CRIT_ACT_DT'].dt.year

# COMMAND ----------

Cohort_Aug_H_02 = Cohort_Aug_H.reset_index()
Cohort_Aug_H_02.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pair first delinquent indicator (CLTN) with nearest event  between (WO, OCA, CURE)

# COMMAND ----------

Cohort_Aug_H_02['index'] = Cohort_Aug_H_02['index'].astype(str)
Cohort_Aug_H_GROUP = Cohort_Aug_H_02.groupby(['BAN', 'TRANS_YEAR'])[['index', 'CRIT_ACT_TYPE_CD']].agg(', '.join).reset_index()
Cohort_Aug_H_GROUP.columns = ['BAN', 'TRANS_YEAR', 'index_comb', 'CRIT_ACT_TYPE_CD_comb']
Cohort_Aug_H_GROUP.head()

# COMMAND ----------

Cohort_Aug_H_GROUP['ROW_CLEAN_UP'] = Cohort_Aug_H_GROUP.apply(lambda x: detect_rows_to_be_dropped(x['index_comb'], \
                                                            x['CRIT_ACT_TYPE_CD_comb'], ls_valid_crit_act_cd), axis=1)

# COMMAND ----------

Cohort_Aug_H_GROUP.head()

# COMMAND ----------

total_index_dropped = Cohort_Aug_H_GROUP['ROW_CLEAN_UP'].apply(lambda x: x[0]).tolist()

# COMMAND ----------

list_total_index_dropped = [item for sublist in total_index_dropped for item in sublist]

# COMMAND ----------

Cohort_Aug_H_NEW = Cohort_Aug_H.drop(list_total_index_dropped)
Cohort_Aug_H_NEW.shape

# COMMAND ----------

pd.set_option('display.max_rows', None)
Cohort_Aug_H_NEW['FLAG_CLTN'] = np.where(Cohort_Aug_H_NEW['CRIT_ACT_TYPE_CD'] == 'CLTN', 1, 0)  #add flag for CLTN

Cohort_Aug_H_NEW = Cohort_Aug_H_NEW.reset_index(drop=True)

# COMMAND ----------

s = Cohort_Aug_H_NEW.groupby('CRIT_ACT_TYPE_CD').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s.pivot_table(index = ['CRIT_ACT_TYPE_CD'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

Cohort_Aug_H_NEW['FIRST_EVENT'] = np.where(Cohort_Aug_H_NEW['CRIT_ACT_TYPE_CD'] != 'CLTN', 1, 0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cycle indicator

# COMMAND ----------

Cohort_Aug_H_NEW["FLAG_CYC"]=0

Cohort_Aug_H_NEW.loc[(Cohort_Aug_H_NEW['FLAG_CLTN']== 1) | (Cohort_Aug_H_NEW['FIRST_EVENT'] == 1),"FLAG_CYC"]=1

# COMMAND ----------

Cohort_Aug_H_NEW.head()

# COMMAND ----------

Cohort_Aug_H_NEW.shape

# COMMAND ----------

CYC_CURE = Cohort_Aug_H_NEW[Cohort_Aug_H_NEW["FLAG_CYC"]==1]

CYC_CURE = CYC_CURE.reset_index(drop=True)

# COMMAND ----------

CYC_CURE.shape[0]

# COMMAND ----------

s = CYC_CURE.groupby('CRIT_ACT_TYPE_CD').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s.pivot_table(index = ['CRIT_ACT_TYPE_CD'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

CYC_CURE.shape

# COMMAND ----------

# MAGIC %md
# MAGIC #### Assign an index for each pair

# COMMAND ----------

num = int(len(CYC_CURE.index)/2)+1

num

# COMMAND ----------

import itertools

lst = range(1,num)

xlist =  list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in lst))


# COMMAND ----------

len(xlist)

# COMMAND ----------

CYC_CURE['COMODIN'] = xlist


# COMMAND ----------

CYC_CURE['BANCOMODIN'] = CYC_CURE['BAN'].astype(str) + ' '+ CYC_CURE['COMODIN'].astype(str)


# COMMAND ----------

CYC_CURE_H = CYC_CURE.pivot_table(index=['BANCOMODIN'],
                                          columns = 'CRIT_ACT_TYPE_CD',
                                          aggfunc=min,
                                          values='CRIT_ACT_DT')


# COMMAND ----------

CYC_CURE_H.shape

# COMMAND ----------

CYC_CURE_H = CYC_CURE_H.reset_index()

# COMMAND ----------

CYC_CURE_H.info()

# COMMAND ----------

CYC_CURE_H['BAN'] = CYC_CURE_H.apply(lambda x: x['BANCOMODIN'].split()[0], axis=1)

# COMMAND ----------

CYC_CURE_H.head()

# COMMAND ----------

CYC_CURE_H = CYC_CURE_H.drop('BANCOMODIN', axis=1)

CYC_CURE_H = CYC_CURE_H[['BAN', 'CLTN', 'CURE',  'WOFF', 'OCAE']]

CYC_CURE_H.columns = ['BAN', 'DLNQ_DT', 'CURE_DT', 'WO_DT', 'OCA_DT']

# COMMAND ----------

CYC_CURE_H['CURE_DT'] = CYC_CURE_H.apply(lambda x: x['CURE_DT']+timedelta(days=int(1)), axis=1)

# COMMAND ----------

CYC_CURE_H.head(12)

# COMMAND ----------

CYC_CURE_H.tail(12)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Verify duplicates and na's

# COMMAND ----------

dup = CYC_CURE_H[CYC_CURE_H.duplicated(['BAN', 'DLNQ_DT'])]

dup

# COMMAND ----------

CYC_CURE_H.CURE_DT.isna().sum()

# COMMAND ----------

CYC_CURE_H.DLNQ_DT.isna().sum()

# COMMAND ----------

CYC_CURE_H.WO_DT.isna().sum()

# COMMAND ----------

CYC_CURE_H.OCA_DT.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Take those from august at dlnq dt since we could have more dlnq cohorts because window expansion

# COMMAND ----------

CYC_CURE_H.shape

# COMMAND ----------

CYC_CURE_H = CYC_CURE_H[CYC_CURE_H.DLNQ_DT.dt.month < 8]

# COMMAND ----------

CYC_CURE_H.shape

# COMMAND ----------

len(CYC_CURE_H.BAN.unique())

# COMMAND ----------

dup = CYC_CURE_H[CYC_CURE_H.duplicated(['BAN', 'DLNQ_DT'])]

dup

# COMMAND ----------

CYC_CURE_H.head()

# COMMAND ----------

CYC_CURE_H.groupby(CYC_CURE_H.DLNQ_DT.dt.month).size()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insert time

# COMMAND ----------

df_action_tm = df_act[['acct_nbr', 'critical_action_dt', 'critical_action_time', 'critical_action_type_cd']]
df_action_tm.columns = ['BAN', 'CRIT_ACT_DT', 'CRIT_ACT_TIME', 'CRIT_ACT_TYPE_CD']

df_action_tm['BAN'] = df_action_tm['BAN'].astype(str)

df_action_tm['CRIT_ACT_TIME'] = pd.to_numeric(df_action_tm['CRIT_ACT_TIME'], errors='coerce')

df_action_tm['CRIT_ACT_TIME'] = df_action_tm['CRIT_ACT_TIME'].fillna(0)
df_action_tm['CRIT_ACT_TIME'] = df_action_tm.apply(lambda x: round(x['CRIT_ACT_TIME'], 0), axis=1)
df_action_tm['CRIT_ACT_TIME'] = df_action_tm['CRIT_ACT_TIME'].astype(int)
df_action_tm['CRIT_ACT_TIME'] = df_action_tm['CRIT_ACT_TIME'].astype(str)

df_action_tm['CRIT_ACT_DT'] = df_action_tm['CRIT_ACT_DT'].astype('datetime64[D]')
df_action_tm.CRIT_ACT_TIME = df_action_tm.apply(lambda x: x.CRIT_ACT_TIME.strip(), axis=1)

df_action_tm['CRIT_ACT_DT'] = df_action_tm.apply(lambda x: x['CRIT_ACT_DT']+timedelta(days=int(1)) 
                                                 if x.CRIT_ACT_TYPE_CD == 'CURE' else x['CRIT_ACT_DT'], axis=1)

df_action_tm = df_action_tm.sort_values(by=['BAN','CRIT_ACT_DT'])
df_action_tm = df_action_tm.reset_index(drop=True)

df_action_tm['len_time'] = df_action_tm.apply(lambda x: len(x.CRIT_ACT_TIME), axis=1)

df_action_tm.head()

# COMMAND ----------

df_action_tm['ACTION_DTTM_CLTN'] = df_action_tm.apply(lambda x: new_datetime(x, 'CRIT_ACT_TIME', 'CRIT_ACT_DT', central_tz), axis=1)
df_action_tm['ACTION_DTTM_TIMEZONE_CLTN'] = df_action_tm.apply(lambda x: x.ACTION_DTTM_CLTN.tzinfo, axis=1)
df_action_tm['ACTTION_DTTM_PACIFIC_TZ_CLTN'] = df_action_tm.apply(lambda x: x['ACTION_DTTM_CLTN'].astimezone(pacific_tz), axis=1)

# COMMAND ----------

df_action_tm.columns

# COMMAND ----------

df_action_tm = df_action_tm[['BAN', 'CRIT_ACT_DT', 'ACTION_DTTM_CLTN','ACTION_DTTM_TIMEZONE_CLTN', 'ACTTION_DTTM_PACIFIC_TZ_CLTN']]

# COMMAND ----------

df_action_tm[df_action_tm['BAN'] == '100057601']

# COMMAND ----------

dup = df_action_tm[df_action_tm.duplicated(['BAN', 'CRIT_ACT_DT'])]

dup.shape

# COMMAND ----------

df_action_tm = df_action_tm.drop_duplicates(['BAN', 'CRIT_ACT_DT'], keep='first')

# COMMAND ----------

dup = df_action_tm[df_action_tm.duplicated(['BAN', 'CRIT_ACT_DT'])]

dup.shape

# COMMAND ----------

CYC_CURE_H.shape

# COMMAND ----------

df_action_tm.columns = ['BAN', 'DLNQ_DT', 'ACTION_DTTM_CLTN','ACTION_DTTM_TIMEZONE_CLTN', 'ACTION_DTTM_PACIFIC_TZ_CLTN']

# COMMAND ----------

CYC_CURE_H = pd.merge(CYC_CURE_H,
                      df_action_tm, 
                      how='left',
                      left_on=['BAN', 'DLNQ_DT'],
                      right_on=['BAN', 'DLNQ_DT'])

# COMMAND ----------

df_action_tm.columns = ['BAN', 'CURE_DT', 'ACTION_DTTM_CURE','ACTION_DTTM_TIMEZONE_CURE', 'ACTION_DTTM_PACIFIC_TZ_CURE']

# COMMAND ----------

CYC_CURE_H = pd.merge(CYC_CURE_H,
                      df_action_tm, 
                      how='left',
                      left_on=['BAN', 'CURE_DT'],
                      right_on=['BAN', 'CURE_DT'])

# COMMAND ----------

CYC_CURE_H.shape

# COMMAND ----------

CYC_CURE_H.head()

# COMMAND ----------

CYC_CURE_H.dtypes

# COMMAND ----------

CYC_CURE_H.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part ll. Set the boundaries of dlnq cycles and prepare treatments

# COMMAND ----------

# MAGIC %md
# MAGIC ### New DF Call <span style='color:Pink'> (Central to Pacific Time) </span>

# COMMAND ----------

df_calls.head(1)

# COMMAND ----------

df_calls.call_type_cd.unique()

# COMMAND ----------

df_call_tz = df_calls[['acct_nbr', 'call_dt', 'call_time', 'call_type_cd', 'call_center_cd']]
df_call_tz.columns = ['BAN', 'CALL_DT', 'CALL_TIME', 'TRANS_TYPE', 'TRANS_SUB_TYPE']

df_call_tz['BAN'] = df_call_tz['BAN'].astype(str)
df_call_tz['CALL_DT'] = df_call_tz['CALL_DT'].astype('datetime64[D]')

df_call_tz = df_call_tz[df_call_tz['TRANS_TYPE'] != 'IB']

df_call_tz = df_call_tz.sort_values(by =['BAN','CALL_DT'], ascending=True)
df_call_tz = df_call_tz.reset_index(drop=True)

df_call_tz['TREAT_MSG_CD'] = df_call_tz.apply(lambda x: 'OUTBOUNDCALL' if x.TRANS_TYPE == 'OB' else 
                                              ('AD_CALL' if x.TRANS_TYPE == 'AD' else 'INBOUNDCALL'), axis=1)

print('Estas son las llamadas incluidas:'+str(df_call_tz.TREAT_MSG_CD.unique()))

#New datetime

df_call_tz['len_time'] = df_call_tz.apply(lambda x: len(str(x.CALL_TIME)), axis=1)
df_call_tz['CALL_DT_TZ'] = df_call_tz.apply(lambda x: new_datetime(x, 'CALL_TIME', 'CALL_DT', central_tz), axis=1)
df_call_tz['CALL_TIMEZONE'] = df_call_tz.apply(lambda x: x.CALL_DT_TZ.tzinfo, axis=1)
df_call_tz['CALL_DT_TZ_PACIFIC'] = df_call_tz.apply(lambda x: x['CALL_DT_TZ'].astimezone(pacific_tz), axis=1)

df_call_tz = df_call_tz.sort_values(by =['BAN','CALL_DT_TZ_PACIFIC'], ascending=True)
df_call_tz = df_call_tz.reset_index(drop=True)


df_call_tz['CALL_DT_PACIFIC'] = df_call_tz.apply(lambda x: x['CALL_DT_TZ_PACIFIC'].date(), axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### How many changes in day we would have

# COMMAND ----------

df_call_tz['IND_TIME_CHANGED_DAY'] = df_call_tz.apply(lambda x: 0 if x['CALL_DT_TZ'].date() ==  x['CALL_DT_TZ_PACIFIC'].date() else 1, axis=1)

# COMMAND ----------

df_call_tz['IND_TIME_CHANGED_DAY'].sum() #There are 70725 changes of day because of time zone difference

# COMMAND ----------

df_call_tz.shape

# COMMAND ----------

df_call_tz.head()

# COMMAND ----------

df_call_tz['CALL_DT_PACIFIC'] = df_call_tz['CALL_DT_PACIFIC'].astype('datetime64[D]')

# COMMAND ----------

df_call_tz_cut = df_call_tz[['BAN', 'CALL_DT', 'CALL_DT_TZ', 'TRANS_TYPE', 'TRANS_SUB_TYPE', 'TREAT_MSG_CD', 'CALL_DT_PACIFIC', 'CALL_DT_TZ_PACIFIC']]

# COMMAND ----------

df_call_tz_cut[(df_call_tz_cut["BAN"] == '100142994') & (df_call_tz_cut['CALL_DT_TZ_PACIFIC'].dt.month == 8)]

# COMMAND ----------

df_call_tz_cut.dtypes

# COMMAND ----------

dup = df_call_tz_cut[df_call_tz_cut.duplicated(['BAN', 'CALL_DT', 'TREAT_MSG_CD', 'CALL_DT_PACIFIC', 'CALL_DT_TZ_PACIFIC'])]

print(len(dup))

dup.head()

# COMMAND ----------

df_call_tz_cut = df_call_tz_cut.drop_duplicates(['BAN', 'CALL_DT', 'TREAT_MSG_CD', 'CALL_DT_PACIFIC', 'CALL_DT_TZ_PACIFIC'], keep='first')

# COMMAND ----------

dup = df_call_tz_cut[df_call_tz_cut.duplicated(['BAN', 'CALL_DT', 'TREAT_MSG_CD', 'CALL_DT_PACIFIC', 'CALL_DT_TZ_PACIFIC'])]

print(len(dup))

dup.head()

# COMMAND ----------

df_call_tz_cut.shape

# COMMAND ----------

df_call_tz_cut.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ### DF MESSAGE <span style='color:Pink'> (Central to Pacific Time) </span>

# COMMAND ----------

df_treatment_timestamp.head(1)

# COMMAND ----------

# df_treatment_timestamp.timezone_cd.unique()

# COMMAND ----------

df_message_tz = df_treatment_timestamp[['acct_nbr', 'message_dt', 'message_type_cd', 'message_subtype_cd',
                                'message_cltn_sys_letter_cd', 'Message_Dt_Tm', 'timezone_cd']]

df_message_tz.columns = ['BAN', 'MESSAGE_DT', 'TRANS_TYPE', 'TRANS_SUB_TYPE', 'TREAT_MSG_CD',
                         'MESSAGE_DT_TM', 'TIMEZONE_CD']

df_message_tz['BAN'] = df_message_tz['BAN'].astype(str)
df_message_tz['MESSAGE_DT'] = df_message_tz['MESSAGE_DT'].astype('datetime64[D]')
df_message_tz['MESSAGE_DT_TM'] = pd.to_datetime(df_message_tz['MESSAGE_DT_TM'], errors = 'coerce')

df_message_tz = df_message_tz.sort_values(by =['BAN','MESSAGE_DT'], ascending=True)
df_message_tz = df_message_tz.reset_index(drop=True)

# COMMAND ----------

df_message_tz['tm_msg_trans'] = df_message_tz.apply(lambda x: time(x.MESSAGE_DT_TM.hour, 
                                                                   x.MESSAGE_DT_TM.minute,
                                                                   x.MESSAGE_DT_TM.second), axis=1)

# COMMAND ----------

df_message_tz['MESSAGE_DT']= pd.to_datetime(df_message_tz['MESSAGE_DT'], format="%m/%d/%Y %H:%M:%S")
df_message_tz['MESSAGE_DATETIME'] = df_message_tz.apply(lambda x: datetime.combine(x.MESSAGE_DT, x.tm_msg_trans), axis=1)
df_message_tz['MESSAGE_DATETIME'] = df_message_tz.apply(lambda x: central_tz.localize(x['MESSAGE_DATETIME'], is_dst=True), axis=1)
df_message_tz['MESSAGE_DATETIME_PACIFIC'] = df_message_tz.apply(lambda x: x['MESSAGE_DATETIME'].astimezone(pacific_tz), axis=1)

# COMMAND ----------

df_message_tz['MESSAGE_DT_PACIFIC'] = df_message_tz.apply(lambda x: x.MESSAGE_DATETIME_PACIFIC.date(), axis=1)

# COMMAND ----------

df_message_tz['MESSAGE_DT_PACIFIC'] = df_message_tz['MESSAGE_DT_PACIFIC'].astype('datetime64[D]')

# COMMAND ----------

df_message_tz = df_message_tz.sort_values(by=['BAN', 'MESSAGE_DATETIME_PACIFIC', 'MESSAGE_DT_PACIFIC', 'TREAT_MSG_CD'], 
                                               ascending=[True, True, True, True])
df_message_tz = df_message_tz.reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Dismiss messages codes

# COMMAND ----------

df_message_tz_cut = df_message_tz[['BAN', 'MESSAGE_DT', 'MESSAGE_DATETIME', 'TRANS_TYPE', 'TRANS_SUB_TYPE', 
                                   'TREAT_MSG_CD', 'MESSAGE_DT_PACIFIC', 'MESSAGE_DATETIME_PACIFIC']]   

# COMMAND ----------

dup = df_message_tz_cut[df_message_tz_cut.duplicated(['BAN', 'MESSAGE_DT', 'TREAT_MSG_CD', 'MESSAGE_DT_PACIFIC', 'MESSAGE_DATETIME_PACIFIC'])]

print(len(dup))

dup.head()


# COMMAND ----------

df_message_tz_cut = df_message_tz_cut.drop_duplicates(['BAN', 'MESSAGE_DT', 'TREAT_MSG_CD', 'MESSAGE_DT_PACIFIC', 'MESSAGE_DATETIME_PACIFIC'], keep='first')

# COMMAND ----------

dup = df_message_tz_cut[df_message_tz_cut.duplicated(['BAN', 'MESSAGE_DT', 'TREAT_MSG_CD', 'MESSAGE_DT_PACIFIC', 'MESSAGE_DATETIME_PACIFIC'])]

print(len(dup))

dup.head()



# COMMAND ----------

df_message_tz_cut.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC # <span style='color:Orange'> Approach </span>
# MAGIC - Mix calls and messages to replace TRANS_DT with the earliest time treatment from MESSAGE_DT_TM_PACIFIC or CALL_DT_TZ_PACIFIC ocurred the same day using Pacif time zone
# MAGIC - If no treatment then let same TRANS_DT with 00:00:00 hour in pacific time zone

# COMMAND ----------

df_message_tz_cut.columns = ['BAN', 'TREAT_DT', 'TREAT_DTTM', 'TRANS_TYPE', 'TRANS_SUB_TYPE',
                             'TREAT_MSG_CD', 'TREAT_DT_PACIFIC', 'TREAT_DTTM_PACIFIC']

# COMMAND ----------

df_call_tz_cut.columns = ['BAN', 'TREAT_DT', 'TREAT_DTTM', 'TRANS_TYPE', 'TRANS_SUB_TYPE',
                             'TREAT_MSG_CD', 'TREAT_DT_PACIFIC', 'TREAT_DTTM_PACIFIC']

# COMMAND ----------

df_treatment_tz = pd.concat([df_message_tz_cut, df_call_tz_cut], axis=0)

# COMMAND ----------

df_treatment_tz = df_treatment_tz.sort_values(by =['BAN','TREAT_DTTM_PACIFIC'], ascending=True)
df_treatment_tz = df_treatment_tz.reset_index(drop=True)

# COMMAND ----------

df_treatment_tz.dtypes

# COMMAND ----------

df_treatment_tz.TRANS_TYPE = df_treatment_tz.apply(lambda x: x.TRANS_TYPE.upper(), axis=1)

# COMMAND ----------

df_treatment_tz.TRANS_TYPE.unique()

# COMMAND ----------

df_treatment_tz.TRANS_SUB_TYPE.unique()

# COMMAND ----------

df_treatment_tz.TREAT_MSG_CD.unique()

# COMMAND ----------

df_treatment_tz.shape

# COMMAND ----------

df_treatment_tz.head()

# COMMAND ----------

df_treatment_tz_first = df_treatment_tz.drop_duplicates(['BAN','TREAT_DT_PACIFIC'], keep='first')

# COMMAND ----------

df_treatment_tz_first.shape

# COMMAND ----------

df_treatment_tz_first = df_treatment_tz_first.sort_values(by =['BAN','TREAT_DTTM_PACIFIC'], ascending=True)
df_treatment_tz_first = df_treatment_tz_first.reset_index(drop=True)

# COMMAND ----------

df_treatment_tz_first[(df_treatment_tz_first["BAN"] == '100142994') & (df_treatment_tz_first['TREAT_DTTM_PACIFIC'].dt.month >= 8) & (df_treatment_tz_first['TREAT_DTTM_PACIFIC'].dt.year >= 2019)]

# COMMAND ----------

df_treatment_tz_first[(df_treatment_tz_first["BAN"] == '100057601') & (df_treatment_tz_first['TREAT_DTTM_PACIFIC'].dt.month >= 8) & (df_treatment_tz_first['TREAT_DTTM_PACIFIC'].dt.year >= 2019)]

# COMMAND ----------

df_treatment_tz_first.columns

# COMMAND ----------

df_treatment_tz_first.columns = ['BAN', 'TREAT_DT', 'TREAT_DTTM',
                                 'TRANS_TYPE_TIMECUTOFF', 'TRANS_SUB_TYPE_TIMECUTOFF',
                                 'TREAT_MSG_CD_TIMECUTOFF',
                                 'TREAT_DT_PACIFIC_CUTOFF', 'TREAT_DTTM_PACIFIC_CUTOFF']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part lll. Calculate proj susp dt

# COMMAND ----------

# MAGIC %md
# MAGIC #### Merge with Strata before

# COMMAND ----------

Dict_risk = {"NT": 0, "LL": 1, "LM": 2, "LH": 3, "ML": 4, "MM": 5,
             "MH": 6, "HL": 7, "HM": 8, "HH": 9, "FP": 10, "SH": 11,
             "LT":12, "NF":13, "CR":14}

# COMMAND ----------

# MAGIC %md
# MAGIC ###  DF Strata 
# MAGIC 
# MAGIC    Matching strata by decision date (dlnq date) with critical action file with corresponging delinquent date. It has the information we need instead of Cohort DF, we get from this df next variables:
# MAGIC    1. Risk code
# MAGIC    2. Strata Instruction code
# MAGIC    3. Total due amount and total delinquent amount

# COMMAND ----------

# Extract BAN, DLNQ_DT, RISK, DUE AMT:
df_strata = df_str[["acct_nbr", "strata_decision_dt", "strata_cltn_risk_segment_cd", "strata_instruction_cd", "strata_tot_due_amt", "strata_tot_dlnq_amt"]]
df_strata.columns = ['BAN', 'DLNQ_DT_STR', 'RISK_CD', "INSTR_CD", 'TOT_DUE_AMT', 'TOT_DLNQ_AMT']

df_strata['BAN'] = df_strata['BAN'].astype(str)
df_strata['DLNQ_DT_STR'] = df_strata['DLNQ_DT_STR'].astype('datetime64[D]')
df_strata['TRANS_MONYR']=pd.to_datetime(df_strata['DLNQ_DT_STR']).dt.strftime('%Y%m')

df_strata = df_strata.sort_values(by=['BAN','DLNQ_DT_STR','TRANS_MONYR'])
df_strata = df_strata.reset_index(drop=True)

# COMMAND ----------

s = df_strata.groupby('RISK_CD').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s["index"] = s['RISK_CD'].map(Dict_risk)
s.set_index("index").sort_values("index")
s.pivot_table(index = ['index', 'RISK_CD'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merging delinquet cycles with STRATA DF

# COMMAND ----------

CYC_CURE_H_STR = pd.merge(CYC_CURE_H, df_strata, how='left', left_on=['BAN'], right_on=['BAN'])

# COMMAND ----------

CYC_CURE_H_STR.shape

# COMMAND ----------

CYC_CURE_H_STR.head()

# COMMAND ----------

CYC_CURE_H_STR.dtypes

# COMMAND ----------

len(CYC_CURE_H_STR.BAN.unique())

# COMMAND ----------

CYC_CURE_H_STR2 = CYC_CURE_H_STR[(CYC_CURE_H_STR['DLNQ_DT_STR'] <= CYC_CURE_H_STR['DLNQ_DT'])]

# COMMAND ----------

len(CYC_CURE_H_STR2.BAN.unique())

# COMMAND ----------

2150-1914

# COMMAND ----------

CYC_CURE_H_STR2.shape

# COMMAND ----------

# MAGIC %md
# MAGIC #### Drop duplicates with same dlnq dt but different risk

# COMMAND ----------

dup = CYC_CURE_H_STR2[CYC_CURE_H_STR2.duplicated(['BAN', 'DLNQ_DT'])]

len(dup)

# COMMAND ----------

CYC_CURE_H_STR2 = CYC_CURE_H_STR2.sort_values(by=['BAN', 'DLNQ_DT_STR', 'RISK_CD'], ascending=False)

CYC_CURE_H_STR2 = CYC_CURE_H_STR2.reset_index(drop = True)

# COMMAND ----------

CYC_CURE_H_STR2 = CYC_CURE_H_STR2.sort_values(['BAN','DLNQ_DT']).drop_duplicates(['BAN', 'DLNQ_DT'], keep='first')

# COMMAND ----------

CYC_CURE_H_STR2.columns

# COMMAND ----------

CYC_CURE_H_STR2.shape

# COMMAND ----------

CYC_CURE_H.shape

# COMMAND ----------

CYC_CURE_H_STR2.shape[0]-CYC_CURE_H.shape[0]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Replace NAN risk cd with NF = not found

# COMMAND ----------

CYC_CURE_H_STR2.RISK_CD.isna().sum()

# COMMAND ----------

CYC_CURE_H_STR2[CYC_CURE_H_STR2.RISK_CD.isna() == True].head()

# COMMAND ----------

CYC_CURE_H_STR2.RISK_CD = CYC_CURE_H_STR2.RISK_CD.replace(np.nan, 'NF')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Merge cycles with first treatment by dlnq dt without standarize

# COMMAND ----------

CYC_CURE_H_STR2.shape

# COMMAND ----------

CYC_CURE_H_STR2.ACTION_DTTM_PACIFIC_TZ_CLTN.dt.month.unique()

# COMMAND ----------

CYC_CURE_H_STR2.ACTION_DTTM_PACIFIC_TZ_CLTN.dt.year.unique()

# COMMAND ----------

CYC_CURE_H_STR2.head()

# COMMAND ----------

CYC_CURE_H_STR2['CURE_DT_PACIFIC'] = CYC_CURE_H_STR2.apply(lambda x: x.ACTION_DTTM_PACIFIC_TZ_CURE.date(), axis=1)

# COMMAND ----------

CYC_CURE_H_STR2.shape

# COMMAND ----------

CYC_CURE_H_STR2.head()

# COMMAND ----------

CYC_CURE_H_STR2['DLNQ_DT_PACIFIC'] = CYC_CURE_H_STR2.apply(lambda x: x.ACTION_DTTM_PACIFIC_TZ_CLTN.date(), axis=1)

# COMMAND ----------

CYC_CURE_H_STR2.head()

# COMMAND ----------

CYC_CURE_H_STR2.columns

# COMMAND ----------

CYC_CURE_H_TR2 = CYC_CURE_H_STR2[['BAN', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR', 'CURE_DT_PACIFIC', 'WO_DT', 'OCA_DT',
       'RISK_CD', 'INSTR_CD', 'TOT_DUE_AMT', 'TOT_DLNQ_AMT', 'TRANS_MONYR']]

# COMMAND ----------

CYC_CURE_H_TR2.head()

# COMMAND ----------

CYC_CURE_H_TR2.DLNQ_DT_PACIFIC = pd.to_datetime(CYC_CURE_H_TR2.DLNQ_DT_PACIFIC, errors = 'coerce')
CYC_CURE_H_TR2.CURE_DT_PACIFIC = pd.to_datetime(CYC_CURE_H_TR2.CURE_DT_PACIFIC, errors = 'coerce')

# COMMAND ----------

CYC_CURE_H_TR2.dtypes

# COMMAND ----------

CYC_CURE_H_TR2.DLNQ_DT_PACIFIC.dt.month.unique()

# COMMAND ----------

CYC_CURE_H_TR2['TRANS_MONYR']=pd.to_datetime(CYC_CURE_H_TR2['DLNQ_DT_PACIFIC']).dt.strftime('%Y%m')
CYC_CURE_H_TR2['TRANS_MONYR'] = CYC_CURE_H_TR2['TRANS_MONYR'].astype(str)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get suspension date 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Treatment Timeline by Risk

# COMMAND ----------

CYC_CURE_H_TR2.RISK_CD.unique() # There are some blanks

# COMMAND ----------

CYC_CURE_H_TR2.RISK_CD = CYC_CURE_H_TR2.RISK_CD.replace('  ', 'NF')

# COMMAND ----------

s = CYC_CURE_H_STR2.groupby('RISK_CD').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s["index"] = s['RISK_CD'].map(Dict_risk)
s.set_index("index").sort_values("index")
s.pivot_table(index = ['index', 'RISK_CD'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

#### Updatad for EG

data = [['NT',0,0,0,0,32],  #Same as LL?
        ['LL',0,0,0,0,32], 
        ['LM',0,0,0,0,30], 
        ['LH',0,0,0,0,28], 
        ['ML',0,0,0,0,25], 
        ['MM',0,0,0,0,19], 
        ['MH',0,0,0,0,13], 
        ['HL',0,0,0,0,10], 
        ['HM',0,0,0,0,9], 
        ['HH',0,0,0,0,8], 
        ['LT',0,0,0,0,15], 
        ['FP',0,0,0,0,15],
        ['SH',0,0,0,0,15], #Same as LT?
        ['NF',0,0,0,0,15],  #Same as LT if Not Found / Missing
        ['CR',0,0,0,0,15],  #Same as LT?
       ]

df_timeline_susp = pd.DataFrame(data, columns=['Risk', 'Frie_SMS_Email_Days', 'Ent_Dlnq_SMS_Days', 'PreSusp_SMS_Email_Days', 'PreSusp_Lett_Days', 'Susp_Days'])
df_timeline_susp["index"] = df_timeline_susp['Risk'].map(Dict_risk)
df_timeline_susp = df_timeline_susp.set_index('Risk')
df_timeline_susp = df_timeline_susp.sort_values("index")
df_timeline_susp

# COMMAND ----------

CYC_CURE_H_TR2['T1'] = CYC_CURE_H_TR2.apply(lambda x: timeline_days(x['RISK_CD'], 1), axis=1)
CYC_CURE_H_TR2['T2'] = CYC_CURE_H_TR2.apply(lambda x: timeline_days(x['RISK_CD'], 2), axis=1)
CYC_CURE_H_TR2['T3'] = CYC_CURE_H_TR2.apply(lambda x: timeline_days(x['RISK_CD'], 3), axis=1)
CYC_CURE_H_TR2['T4'] = CYC_CURE_H_TR2.apply(lambda x: timeline_days(x['RISK_CD'], 4), axis=1)
CYC_CURE_H_TR2['T5'] = CYC_CURE_H_TR2.apply(lambda x: timeline_days(x['RISK_CD'], 5), axis=1)
CYC_CURE_H_TR2['PROJ_SUSP_DT'] = CYC_CURE_H_TR2.apply(lambda x: (adding_business_days(x['DLNQ_DT_PACIFIC'], x['T5'])), axis=1)

# COMMAND ----------

CYC_CURE_H_TR2.columns

# COMMAND ----------

CYC_CURE_H_TR2.head(15)

# COMMAND ----------

s = CYC_CURE_H_TR2.groupby('TRANS_MONYR').size().reset_index(name='FREQ')
s["PC_FREQ"] = round(s['FREQ']/sum( s['FREQ'])*100, 1)
s.pivot_table(index = ['TRANS_MONYR'],
             margins = True, 
             margins_name='Total',
             aggfunc=sum)

# COMMAND ----------

CYC_CURE_H_TR2.shape

# COMMAND ----------

CYC_CURE_H_TR2.columns

# COMMAND ----------

CYC_CURE_H_TR2.dtypes

# COMMAND ----------

#After create all the T's
df_base = CYC_CURE_H_TR2.copy()

df_base = df_base.reset_index(drop = True)

# COMMAND ----------

df_base.head()

# COMMAND ----------

df_base.columns

# COMMAND ----------

df_base["DAYS_TO_CURE"] = (df_base["CURE_DT_PACIFIC"]-df_base["DLNQ_DT_PACIFIC"]).dt.days


df_base = df_base[['BAN', 'DLNQ_DT_PACIFIC', 'DLNQ_DT_STR', 'CURE_DT_PACIFIC', 'WO_DT',
       'OCA_DT', 'DAYS_TO_CURE', 'RISK_CD', 'INSTR_CD', 'TOT_DUE_AMT', 'TOT_DLNQ_AMT',
       'TRANS_MONYR', 'T1', 'T2', 'T3', 'T4', 'T5', 'PROJ_SUSP_DT']]

df_base = df_base.sort_values(by=["BAN", "DLNQ_DT_PACIFIC"])

# COMMAND ----------

df_base.shape

# COMMAND ----------

df_base.head()

# COMMAND ----------

df_base['LST_EVENT_DT']=np.where((df_base['CURE_DT_PACIFIC'].isna() & df_base['WO_DT'].isna()),
                                  df_base['OCA_DT'], 
                                    np.where((df_base['CURE_DT_PACIFIC'].isna() & df_base['OCA_DT'].isna()),
                                      df_base['WO_DT'], 
                                        df_base['CURE_DT_PACIFIC']))

# COMMAND ----------

df_base["DAYS_TO_EVENT"] = (df_base["LST_EVENT_DT"]-df_base["DLNQ_DT_PACIFIC"]).dt.days

# COMMAND ----------

df_base.head()

# COMMAND ----------

df_base.shape

# COMMAND ----------

len(df_base.BAN.unique())

# COMMAND ----------

df_base = df_base.reset_index(drop=True)

# COMMAND ----------

df_base=df_base.rename(columns = {'PROJ_SUSP_DT':'LG_SUSP_DT'})

# COMMAND ----------

df_base.LG_SUSP_DT = pd.to_datetime(df_base.LG_SUSP_DT, errors = 'coerce')

# COMMAND ----------

df_base.dtypes

# COMMAND ----------

df_base.isna().sum()

# COMMAND ----------

final_df = create_new_df(df_base)

final_df.head()

# COMMAND ----------

final_df.shape

# COMMAND ----------

final_df.info()

# COMMAND ----------

len(final_df.BAN.unique())

# COMMAND ----------

final_df['TRANS_DT'] = final_df['TRANS_DT'].astype('datetime64[D]')

final_df['DAYS_TO_EVENT'] = final_df.apply(lambda x: (x['LST_EVENT_DT']-x['TRANS_DT']).days, axis=1)

# COMMAND ----------

final_df.columns

# COMMAND ----------

final_df.shape

# COMMAND ----------

final_df.head()

# COMMAND ----------

df_treatment_tz.TRANS_TYPE.unique()

# COMMAND ----------

df_treatment_tz.head(1)

# COMMAND ----------

df_treatment_tz.columns

# COMMAND ----------

df_treatment_mrg = df_treatment_tz[['BAN', 'TREAT_DT_PACIFIC', 'TRANS_TYPE', 'TRANS_SUB_TYPE',
       'TREAT_MSG_CD']] 

# COMMAND ----------

df_treatment_mrg.columns = ['BAN', 'TRANS_DT', 'TRANS_TYPE_H', 'TRANS_SUB_TYPE_H',
       'TREAT_MSG_CD_H']

# COMMAND ----------

final_hist_treat = pd.merge(final_df,
                    df_treatment_mrg,
                    how='left',
                    on=['TRANS_DT', 'BAN'])

final_hist_treat = final_hist_treat.sort_values(['BAN','DLNQ_DT_PACIFIC'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### FILL 'NA' for features where data is not available

# COMMAND ----------

final_hist_treat['TRANS_TYPE_H']=final_hist_treat['TRANS_TYPE_H'].fillna('NONE')
final_hist_treat['TRANS_SUB_TYPE_H']=final_hist_treat['TRANS_SUB_TYPE_H'].fillna('NONE')
final_hist_treat['TREAT_MSG_CD_H']=final_hist_treat['TREAT_MSG_CD_H'].fillna('NONE')

# COMMAND ----------

final_hist_treat['TRANS_MONYR']=pd.to_datetime(final_hist_treat['DLNQ_DT_PACIFIC']).dt.strftime('%Y%m')
final_hist_treat['TRANS_MONYR'] = final_hist_treat['TRANS_MONYR'].astype(str)

# COMMAND ----------

final_hist_treat.head(1)

# COMMAND ----------

final_hist_treat.shape

# COMMAND ----------

len(final_hist_treat.BAN.unique())

# COMMAND ----------

dup = final_hist_treat[final_hist_treat.duplicated(['TRANS_DT', 'BAN', 'RISK_CD', 'TRANS_MONYR'])]
dup.head()

# COMMAND ----------

final_hist_treat[final_hist_treat['BAN']=='101611986']

# COMMAND ----------

final_hist_treat_GP = final_hist_treat.groupby(['TRANS_DT', 'BAN', 'DLNQ_DT_PACIFIC', 'CURE_DT_PACIFIC', 'LG_SUSP_DT', 'TRANS_MONYR', 'RISK_CD']
                        , as_index = False).agg({'TREAT_MSG_CD_H': '|'.join})

# COMMAND ----------

dup = final_hist_treat_GP[final_hist_treat_GP.duplicated(['TRANS_DT', 'BAN', 'RISK_CD', 'TRANS_MONYR'])]
dup.head()

# COMMAND ----------

final_hist_treat_GP[final_hist_treat_GP['BAN']=='101611986']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refreshing Grid Treat Msg Code variables (Calls and Mesagge codes)

# COMMAND ----------

TREAT_CD_COMBO_DF = final_hist_treat_GP[['BAN', 'DLNQ_DT_PACIFIC', 'CURE_DT_PACIFIC', 'LG_SUSP_DT', 'TRANS_MONYR', 'RISK_CD', 'TREAT_MSG_CD_H']]

# COMMAND ----------

TREAT_CD_COMBO_DF['TREAT_MSG_CD'] = TREAT_CD_COMBO_DF['TREAT_MSG_CD_H'].replace('NONE', 'A')

# COMMAND ----------

TREAT_CD_COMBO_DF['JOINED_TREAT_MSG_CD'] = TREAT_CD_COMBO_DF.groupby(['BAN', 'TRANS_MONYR', 'RISK_CD'])['TREAT_MSG_CD'].transform(lambda x : ' & '.join(x))

# COMMAND ----------

#TREAT_CD_COMBO_DF.head()

# COMMAND ----------

#TREAT_CD_COMBO_DF.shape

# COMMAND ----------

dup = TREAT_CD_COMBO_DF[TREAT_CD_COMBO_DF.duplicated(['DLNQ_DT_PACIFIC', 'BAN', 'RISK_CD', 'TRANS_MONYR'])]
len(dup)

# COMMAND ----------

TREAT_CD_COMBO_DF = TREAT_CD_COMBO_DF.drop_duplicates(['DLNQ_DT_PACIFIC', 'BAN', 'RISK_CD', 'TRANS_MONYR'], keep='first')

# COMMAND ----------

dup = TREAT_CD_COMBO_DF[TREAT_CD_COMBO_DF.duplicated(['DLNQ_DT_PACIFIC', 'BAN', 'RISK_CD', 'TRANS_MONYR'])]
len(dup)

# COMMAND ----------

TREAT_CD_COMBO_DF = TREAT_CD_COMBO_DF[['BAN', 'DLNQ_DT_PACIFIC', 'CURE_DT_PACIFIC', 'LG_SUSP_DT', 'TRANS_MONYR', 'RISK_CD', 'JOINED_TREAT_MSG_CD']]

# COMMAND ----------

TREAT_CD_COMBO_DF.head()

# COMMAND ----------

len(TREAT_CD_COMBO_DF.BAN.unique())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Session 2 Part 2: create historical treatment features by Wen <a name="subsession22"></a>
# MAGIC - 1.Based on hist treatment df, 3 features generated:  NBR_OF_HIST_DLNQ, HIST_SHORTEST_CURE, TREAT_DAILY_FROM_HIST_SHORTEST_CURE
# MAGIC - 2.Join the extracted hist treatment features back to current cohort dataframe

# COMMAND ----------

ls_current_cohort_months = ['201908', '202008']  #list of current cohort months

# COMMAND ----------

# read below csv of historical treatment dataframe
#df_hist = pd.read_csv('df_iask_hist_treatments_tested_for_Wen.csv', index_col = 0)
df_hist = TREAT_CD_COMBO_DF.copy()
df_hist.shape

# COMMAND ----------

# move this part to the beginning of Session 2, since this is key input to run Session 2

#Importing dataframe of current cohorts: update with Daan's latest df with timestamps
#df_model =  pd.read_csv('C:/Users/ww173k/Project_Wen/iASCE_project/combine_feature_creation_code/df_iask_timestamp_final_aug_06_10_2021_Daan.csv', index_col=0)

# update with Daan's latest df
#df_model =  pd.read_csv('df_iask_enabler_7_9_2021.csv', index_col=0)

# COMMAND ----------

df_hist.head()

# COMMAND ----------

df_check_hist01 = df_hist[['BAN','DLNQ_DT_PACIFIC','CURE_DT_PACIFIC', 'LG_SUSP_DT', 'JOINED_TREAT_MSG_CD']].drop_duplicates().sort_values(by=['BAN', 'DLNQ_DT_PACIFIC'])
df_check_hist01.shape

# COMMAND ----------

df_check_hist01[['CURE_DT_PACIFIC', 'DLNQ_DT_PACIFIC']] = df_check_hist01[['CURE_DT_PACIFIC', 'DLNQ_DT_PACIFIC']].astype('datetime64[D]')

df_check_hist01['Cure_Days'] = (df_check_hist01['CURE_DT_PACIFIC']-df_check_hist01['DLNQ_DT_PACIFIC']).dt.days

# COMMAND ----------

df_check_hist01['DLNQ_YEAR_MON'] = pd.to_datetime(df_check_hist01['DLNQ_DT_PACIFIC']).dt.strftime('%Y%m')

# COMMAND ----------

print(df_check_hist01.shape)
df_check_hist01.head()

# COMMAND ----------

# exclude current cohort months if any
df_check_hist01 = df_check_hist01.loc[~df_check_hist01['DLNQ_YEAR_MON'].isin(ls_current_cohort_months)] 

print(df_check_hist01.shape)

# COMMAND ----------

df_check_hist01['DLNQ_YEAR'] = pd.to_datetime(df_check_hist01['DLNQ_DT_PACIFIC']).dt.year
df_check_hist01.head()

# COMMAND ----------

df_check_hist01['rank_shortest_cure'] = df_check_hist01.groupby(['BAN', 'DLNQ_YEAR'])['Cure_Days'].rank(method='dense')
df_check_hist01.head()

# COMMAND ----------

# number of dlnq in the last 6 month
df_count_hist_dlnq = df_check_hist01.groupby(['BAN', 'DLNQ_YEAR']).size().reset_index(name='NBR_OF_HIST_DLNQ')
print(df_count_hist_dlnq.shape)
df_count_hist_dlnq.head()

# COMMAND ----------

df_check_hist01 = df_check_hist01.merge(df_count_hist_dlnq, how='left', on=['BAN', 'DLNQ_YEAR'])
print(df_check_hist01.shape)
df_check_hist01.head()

# COMMAND ----------

print(df_check_hist01.loc[df_check_hist01['rank_shortest_cure'] == 1].shape)
print(df_check_hist01.loc[df_check_hist01['rank_shortest_cure'] == 1].drop_duplicates(subset=['BAN', 'DLNQ_YEAR', 'Cure_Days'], keep='first').shape)

# COMMAND ----------

df_hist_short_cure = df_check_hist01.loc[df_check_hist01['rank_shortest_cure'] == 1].drop_duplicates(\
    subset=['BAN', 'DLNQ_YEAR', 'Cure_Days'], keep='first')

print(df_hist_short_cure.shape)
df_hist_short_cure.head()

# COMMAND ----------

df_hist_short_cure['Cure_Days'].describe()

# COMMAND ----------

# join the hist nbr of DLNQ, shortest cure and treat combo back to the current cohort dataframe

# COMMAND ----------

df_model.shape

# COMMAND ----------

df_model.head()

# COMMAND ----------

df_model['BAN'].nunique()

# COMMAND ----------

df_model['DLNQ_YEAR'] = pd.to_datetime(df_model['DLNQ_DT_PACIFIC']).dt.year
df_model.shape

# COMMAND ----------

# Make sure the types are the same before merging
df_hist_short_cure['BAN'] = df_hist_short_cure['BAN'].astype(str)
df_model['BAN'] = df_model['BAN'].astype(str)

# COMMAND ----------

df_hist_short_cure_output = df_hist_short_cure[['BAN', 'DLNQ_YEAR', 'JOINED_TREAT_MSG_CD', 'Cure_Days', 'NBR_OF_HIST_DLNQ']].\
rename(columns={'Cure_Days': 'HIST_SHORTEST_CURE', 'JOINED_TREAT_MSG_CD': 'HIST_SHORTEST_CURE_TREAT_COMBO'})

df_join_hist_trt = df_model.merge(df_hist_short_cure_output, how='left', on=['BAN', 'DLNQ_YEAR'])


# COMMAND ----------

df_join_hist_trt['HIST_SHORTEST_CURE_TREAT_COMBO'] = df_join_hist_trt['HIST_SHORTEST_CURE_TREAT_COMBO'].fillna('')
df_join_hist_trt['NBR_OF_HIST_DLNQ'] = df_join_hist_trt['NBR_OF_HIST_DLNQ'].fillna(0)

# COMMAND ----------

df_join_hist_trt.head()

# COMMAND ----------

# check unmatched rows:
df_join_hist_trt[df_join_hist_trt['HIST_SHORTEST_CURE'].isnull()].shape

# COMMAND ----------


df_join_hist_trt[df_join_hist_trt['HIST_SHORTEST_CURE_TREAT_COMBO'] == ''].shape

# COMMAND ----------

# Test a split example:
'A & EMLRE|SMSPD'.split(' & ')

# COMMAND ----------

# note: here we are going to use treat_msg_cd_combo instead trans_flag combo, need to split by delimiter ' &'

def get_treat_per_day_from_shortest_cure_hist(hist_shortest_cure_treat_combo, cur_cohort_days_in_dlnq):
    if hist_shortest_cure_treat_combo == '': return ''
    elif cur_cohort_days_in_dlnq >= len(hist_shortest_cure_treat_combo.split(' & ')): return ''
    else: return hist_shortest_cure_treat_combo.split(' & ')[int(cur_cohort_days_in_dlnq)]

# COMMAND ----------

# derive a new field per Sasha's suggestion: treatment per day based on shortest cure in last 6 months:
df_join_hist_trt['TREAT_DAILY_FROM_HIST_SHORTEST_CURE'] = df_join_hist_trt.apply(lambda x: \
                get_treat_per_day_from_shortest_cure_hist(x['HIST_SHORTEST_CURE_TREAT_COMBO'], x['DAYS_IN_DLNQ']), axis=1)

# COMMAND ----------

df_join_hist_trt.head()

# COMMAND ----------

# check one BAN for example:

df_hist_short_cure_output[df_hist_short_cure_output['BAN'] == '102671435']

# COMMAND ----------

df_hist[df_hist['BAN'] == '102671435']

# COMMAND ----------

df_join_hist_trt[df_join_hist_trt['BAN'] == '102671435']

# COMMAND ----------

df_join_hist_trt.shape

# COMMAND ----------

# clean up some columns, only keep those 3 features for hist treatment

df_join_hist_trt.drop(['HIST_SHORTEST_CURE_TREAT_COMBO', 'DLNQ_YEAR'], axis=1, inplace=True)

print(df_join_hist_trt.shape)


# COMMAND ----------

df_join_hist_trt.head()

# COMMAND ----------

# 3 features generated for hist treatments:  NBR_OF_HIST_DLNQ, HIST_SHORTEST_CURE, TREAT_DAILY_FROM_HIST_SHORTEST_CURE

# COMMAND ----------

# output joined dataframe
#df_join_hist_trt.to_csv('df_iask_with_hist_treat_feature_Wen.csv')

# COMMAND ----------

# plot the distribution of derived features

# COMMAND ----------

df_hist_short_cure_output['DLNQ_YEAR'].value_counts()

# COMMAND ----------

df_hist_short_cure_output['NBR_OF_HIST_DLNQ'].value_counts()

# COMMAND ----------

df_hist_short_cure_output['HIST_SHORTEST_CURE'].plot.hist(bins=30, title='shortest days_to_cure from hist treatments')

# COMMAND ----------

df_hist_short_cure_output['NBR_OF_HIST_DLNQ'].plot.hist(bins=30, title='number of hist DLNQ cycles')

# COMMAND ----------

# Note: df_join_hist_trt is the updated dataframe for current cohort, with added hist treatment features
# we can use it for following feature code session

# COMMAND ----------

# output joined dataframe
# df_join_hist_trt.to_csv('df_iask_with_hist_treat_feature_Wen_0715.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC # Session 3: Hist PA feature creation by Rishabh <a name="session3"></a>
# MAGIC - Create code logic to generate historical payment arrangement features 
# MAGIC - Use Files08 and 01 instead of PAR files as source data, updated by Wen

# COMMAND ----------

#Loading current cohort dataframe with all variables, for later data merging use
#df_original = pd.read_csv('df_iask_enabler_7_9_2021.csv', index_col = 0)

df_original = df_join_hist_trt.copy()
df_original.shape

# COMMAND ----------

#load enabler sampled data files 
#df_coh_file01 = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File1_MainCohort_10k.csv', index_col=0) 
#df_pa_file08 = pd.read_csv('Enabler/ENBLR_iASCE_Trial_File8_Promises_10k.csv', index_col=0)
# df_coh_file01=df_str
# df_pa_file08=df_pa


df_DATA02_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File1_MainCohort_10k.csv', header=True)

df_DATA07_Aug2020 = spark.read.csv('/FileStore/shared_uploads/jg585a@att.com/ENBLR_iASCE_Trial_File8_Promises_10k.csv', header=True)

df_coh = df_DATA02_Aug2020.toPandas()

df_pa = df_DATA07_Aug2020.toPandas()


# COMMAND ----------

# note that in File01, 'acct_nbr' is BAN, 'cohort_data_capture_dt' is DLNQ_DT, 'last_cltn_risk_segment_cd' is risk cd
print(df_coh_file01.shape)
df_coh_file01.head()

# COMMAND ----------

df_coh_file01['acct_nbr'].nunique()

# COMMAND ----------

df_pa_file08['acct_nbr'].nunique()

# COMMAND ----------

df_coh_file01['cohort_data_capture_dt'] = df_coh_file01['cohort_data_capture_dt'].astype('datetime64[D]')

# COMMAND ----------

# note that in File08, there are several date-related fields:
# ptp_taken_dt: promise to pay taken date (PA setup date)
# ptp_final_disposition_dt: promise to pay final payment date (actual payment date)
# ptp_init_prms_dt: promise to pay initial promise date
# ptp_sbsqt_prms_dt: promise to pay second promise date (second part of payment promise date)


# replace all 9999 year dates with 2099. as python throws error (out of bounds timestamp)
# first make sure they are all string types before doing str.replace
date_cols = ['ptp_taken_dt', 'ptp_final_disposition_dt', 'ptp_init_prms_dt', 'ptp_sbsqt_prms_dt']
for col in date_cols:
    df_pa_file08[col] = df_pa_file08[col].astype(str)

df_pa_file08.ptp_taken_dt = df_pa_file08.ptp_taken_dt.str.replace('9999', '2099', regex=True)
df_pa_file08.ptp_final_disposition_dt = df_pa_file08.ptp_final_disposition_dt.str.replace('9999', '2099', regex=True)
df_pa_file08.ptp_init_prms_dt = df_pa_file08.ptp_init_prms_dt.str.replace('9999', '2099', regex=True)
df_pa_file08.ptp_sbsqt_prms_dt = df_pa_file08.ptp_sbsqt_prms_dt.str.replace('9999', '2099', regex=True)


# COMMAND ----------

# Convert to date format
date_cols = ['ptp_taken_dt', 'ptp_final_disposition_dt', 'ptp_init_prms_dt', 'ptp_sbsqt_prms_dt']
for col in date_cols:
    df_pa_file08[col] = pd.to_datetime(df_pa_file08[col], errors = 'coerce')

print(df_pa_file08.shape)
df_pa_file08.head()

# COMMAND ----------

df_pa_combine =  pd.merge(df_pa_file08, df_coh_file01[['acct_nbr', 'cohort_data_capture_dt']], on=['acct_nbr'])
df_pa_combine.shape

# COMMAND ----------

# try two date fields 'ptp_taken_dt' and 'ptp_final_disposition_dt' respectively, following same logic as used in PAR files

# COMMAND ----------

# subsetting dataframe for Aug month only (we are training on aug,19 and aug,20 data)
df_pa_combine = df_pa_combine[df_pa_combine.cohort_data_capture_dt.dt.month == 8]
print(df_pa_combine.shape)

# COMMAND ----------

# Function to calculate Month based on the difference in days
def months_calculate(DT_DIFF):
    if DT_DIFF > 0 and DT_DIFF <= 30:
        return 1
    elif DT_DIFF >= 31 and DT_DIFF <= 60:
        return 2
    elif DT_DIFF >= 61 and DT_DIFF <= 90:
        return 3
    elif DT_DIFF >= 91 and DT_DIFF <= 120:
        return 4
    elif DT_DIFF >= 121 and DT_DIFF <= 150:
        return 5
    elif DT_DIFF >= 151 and DT_DIFF <= 180:
        return 6
    else :
        return 'None'

# COMMAND ----------

# 1. ptp_taken_dt

# COMMAND ----------

# Subset rows for historical data before capture date.
df_pa_combine_ptp_taken_dt = df_pa_combine.loc[df_pa_combine.cohort_data_capture_dt >= df_pa_combine.ptp_taken_dt]
df_pa_combine_ptp_taken_dt['DT_DIFF'] = (df_pa_combine_ptp_taken_dt['cohort_data_capture_dt'] -  \
                                  df_pa_combine_ptp_taken_dt['ptp_taken_dt']).dt.days
df_pa_combine_ptp_taken_dt['PA_MONTH'] = df_pa_combine_ptp_taken_dt['DT_DIFF'].apply(months_calculate)

df_with_month_dummies_ptp_taken_dt = pd.get_dummies(df_pa_combine_ptp_taken_dt, prefix='ptp_taken_dt_PA_MONTH', columns=['PA_MONTH'])

# Aggregating at BAN + DLNQ date
df_with_month_dummies_ptp_taken_dt = df_with_month_dummies_ptp_taken_dt.groupby(['acct_nbr',
                                                                                         'cohort_data_capture_dt'])[['ptp_taken_dt_PA_MONTH_1',
                                                                                             'ptp_taken_dt_PA_MONTH_2',
                                                                                             'ptp_taken_dt_PA_MONTH_3',
                                                                                             'ptp_taken_dt_PA_MONTH_4',
                                                                                             'ptp_taken_dt_PA_MONTH_5',
                                                                                             'ptp_taken_dt_PA_MONTH_6']].sum().reset_index()

PA_MONTHS = ['ptp_taken_dt_PA_MONTH_1', 'ptp_taken_dt_PA_MONTH_2', 'ptp_taken_dt_PA_MONTH_3',
             'ptp_taken_dt_PA_MONTH_4', 'ptp_taken_dt_PA_MONTH_5', 'ptp_taken_dt_PA_MONTH_6']

df_with_month_dummies_ptp_taken_dt['ptp_taken_dt_PA_MAX'] = df_with_month_dummies_ptp_taken_dt[PA_MONTHS].max(axis=1)
df_with_month_dummies_ptp_taken_dt['ptp_taken_dt_PA_MIN'] = df_with_month_dummies_ptp_taken_dt[PA_MONTHS].min(axis=1)
df_with_month_dummies_ptp_taken_dt['ptp_taken_dt_PA_AVG'] = df_with_month_dummies_ptp_taken_dt[PA_MONTHS].mean(axis=1)
df_with_month_dummies_ptp_taken_dt['ptp_taken_dt_PA_TOTAL'] = df_with_month_dummies_ptp_taken_dt[PA_MONTHS].sum(axis=1)

df_with_month_dummies_ptp_taken_dt['cohort_data_capture_dt'] = df_with_month_dummies_ptp_taken_dt['cohort_data_capture_dt'].astype('datetime64[D]')

# COMMAND ----------

df_with_month_dummies_ptp_taken_dt['acct_nbr'] = df_with_month_dummies_ptp_taken_dt['acct_nbr'].astype(str)
df_with_month_dummies_ptp_taken_dt.rename({"acct_nbr": "BAN", "cohort_data_capture_dt": "DLNQ_DT_PACIFIC"}, axis = "columns", inplace = True)
print(df_with_month_dummies_ptp_taken_dt.shape)
df_with_month_dummies_ptp_taken_dt.head()

# COMMAND ----------

df_with_month_dummies_ptp_taken_dt.hist(bins=30, figsize=(15,10))

# COMMAND ----------

# 2. ptp_final_disposition_dt

# COMMAND ----------

# Subset rows for historical data before capture date.
df_pa_combine_ptp_final_disposition_dt = df_pa_combine[df_pa_combine.cohort_data_capture_dt >= df_pa_combine.ptp_final_disposition_dt]
df_pa_combine_ptp_final_disposition_dt['DT_DIFF'] = (df_pa_combine_ptp_final_disposition_dt['cohort_data_capture_dt'] -  \
                                  df_pa_combine_ptp_final_disposition_dt['ptp_final_disposition_dt']).dt.days
df_pa_combine_ptp_final_disposition_dt['PA_MONTH'] = df_pa_combine_ptp_final_disposition_dt['DT_DIFF'].apply(months_calculate)

df_with_month_dummies_ptp_final_disposition_dt = pd.get_dummies(df_pa_combine_ptp_final_disposition_dt, prefix='ptp_final_disposition_dt_PA_MONTH', columns=['PA_MONTH'])

# Aggregating at BAN + DLNQ date
df_with_month_dummies_ptp_final_disposition_dt = df_with_month_dummies_ptp_final_disposition_dt.groupby(['acct_nbr',
                                                                                         'cohort_data_capture_dt'])[['ptp_final_disposition_dt_PA_MONTH_1',
                                                                                             'ptp_final_disposition_dt_PA_MONTH_2',
                                                                                             'ptp_final_disposition_dt_PA_MONTH_3',
                                                                                             'ptp_final_disposition_dt_PA_MONTH_4',
                                                                                             'ptp_final_disposition_dt_PA_MONTH_5',
                                                                                             'ptp_final_disposition_dt_PA_MONTH_6']].sum().reset_index()

PA_MONTHS = ['ptp_final_disposition_dt_PA_MONTH_1', 'ptp_final_disposition_dt_PA_MONTH_2', 'ptp_final_disposition_dt_PA_MONTH_3',
             'ptp_final_disposition_dt_PA_MONTH_4', 'ptp_final_disposition_dt_PA_MONTH_5', 'ptp_final_disposition_dt_PA_MONTH_6']

df_with_month_dummies_ptp_final_disposition_dt['ptp_final_disposition_dt_PA_MAX'] = df_with_month_dummies_ptp_final_disposition_dt[PA_MONTHS].max(axis=1)
df_with_month_dummies_ptp_final_disposition_dt['ptp_final_disposition_dt_PA_MIN'] = df_with_month_dummies_ptp_final_disposition_dt[PA_MONTHS].min(axis=1)
df_with_month_dummies_ptp_final_disposition_dt['ptp_final_disposition_dt_PA_AVG'] = df_with_month_dummies_ptp_final_disposition_dt[PA_MONTHS].mean(axis=1)
df_with_month_dummies_ptp_final_disposition_dt['ptp_final_disposition_dt_PA_TOTAL'] = df_with_month_dummies_ptp_final_disposition_dt[PA_MONTHS].sum(axis=1)

df_with_month_dummies_ptp_final_disposition_dt['cohort_data_capture_dt'] = df_with_month_dummies_ptp_final_disposition_dt['cohort_data_capture_dt'].astype('datetime64[D]')

# COMMAND ----------

df_with_month_dummies_ptp_final_disposition_dt['acct_nbr'] = df_with_month_dummies_ptp_final_disposition_dt['acct_nbr'].astype(str)
df_with_month_dummies_ptp_final_disposition_dt.rename({"acct_nbr": "BAN", "cohort_data_capture_dt": "DLNQ_DT_PACIFIC"}, axis = "columns", inplace = True)
print(df_with_month_dummies_ptp_final_disposition_dt.shape)
df_with_month_dummies_ptp_final_disposition_dt.head()

# COMMAND ----------

df_with_month_dummies_ptp_final_disposition_dt.hist(bins=30, figsize=(15,10))

# COMMAND ----------



# COMMAND ----------

#need to check with Rishabh on below code, commented out since we don't have 'payarr_scr_nbr' in file08

# Subset rows for historical data before capture date.
# df_pa_pyarr_scr_nbr = df_pa[df_pa.cohort_data_capture_dt >= df_pa.pyarr_scr_dttm]

# # Aggregating at BAN + DLNQ date
# df_pa_pyarr_scr_nbr = df_pa_pyarr_scr_nbr.groupby(['acct_nbr','cohort_data_capture_dt'])['pyarr_scr_nbr'].mean().reset_index()
# df_pa_pyarr_scr_nbr.head()

# df_pa_pyarr_scr_nbr['acct_nbr'] = df_pa_pyarr_scr_nbr['acct_nbr'].astype(str)
# df_pa_pyarr_scr_nbr.rename({"acct_nbr": "BAN", "cohort_data_capture_dt": "DLNQ_DT",
#                            "pyarr_scr_nbr": "avg_pyarr_scr_nbr"}, axis = "columns", inplace = True)
# print(df_pa_pyarr_scr_nbr.shape)
# df_pa_pyarr_scr_nbr.head()

# COMMAND ----------



# COMMAND ----------

# checking the dimensions of all dataframes.
print(df_with_month_dummies_ptp_taken_dt.shape)
print(df_with_month_dummies_ptp_final_disposition_dt.shape)


# COMMAND ----------

# below is to append the new features for PA hist to the current cohort dataframe, need to use latest dataframe

# COMMAND ----------

df_original.head()

# COMMAND ----------

# update by Wen, for EG dataframe, we have DLNQ_DT_PACIFIC instead of DLNQ_DT for mobility

df_original['BAN'] = df_original['BAN'].astype(str)
df_original['DLNQ_DT_PACIFIC'] = df_original['DLNQ_DT_PACIFIC'].astype('datetime64[D]')
print(df_original.shape)

# COMMAND ----------

df_original[['BAN', 'DLNQ_DT_PACIFIC']].dtypes

# COMMAND ----------

# dropping previously created historic PA variables.
# cols_to_drop = ['PA_MONTH_1', 'PA_MONTH_2', 'PA_MONTH_3', 'PA_MONTH_4', 'PA_MONTH_5', 'PA_MONTH_6',
#                 'PA_MAX', 'PA_MIN', 'PA_AVG', 'PA_TOTAL']

# # dropping columns from the dataframe.
# df_original.drop(cols_to_drop, axis = 1, inplace=True)



# updated by Wen: Add additional condition before doing drop here: check if those columns already exist
cols_to_drop = ['PA_MONTH_1', 'PA_MONTH_2', 'PA_MONTH_3', 'PA_MONTH_4', 'PA_MONTH_5', 'PA_MONTH_6',
                'PA_MAX', 'PA_MIN', 'PA_AVG', 'PA_TOTAL']

for col in cols_to_drop:
    if col in df_original.columns:
        df_original = df_original.drop([col], axis=1)
        
        
print(df_original.shape)

# COMMAND ----------

df_merged = pd.merge(left = df_original, right = df_with_month_dummies_ptp_taken_dt, how = 'left', on = ['BAN','DLNQ_DT_PACIFIC'])
df_merged = pd.merge(left = df_merged, right = df_with_month_dummies_ptp_final_disposition_dt, how = 'left', on = ['BAN','DLNQ_DT_PACIFIC'])

#df_merged = pd.merge(left = df_merged, right = df_pa_pyarr_scr_nbr, how = 'left', on = ['BAN','DLNQ_DT'])

# COMMAND ----------

df_merged.head(20)

# COMMAND ----------

print("Dimensions of the original dataframe :", df_original.shape)
print("Dimensions after adding variables to the original dataframe :", df_merged.shape)
print("New Columns added are :", set(df_merged.columns.tolist()) - set(df_original.columns.tolist()))

# COMMAND ----------

#df_merged.to_csv('df_iask_with_hist_pa_features.csv')

# COMMAND ----------

#note that df_merged is the updated cohort dataframe, with PA hist features added. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Session 4: Numerical feature creation for grid actions <a name="session4"></a>
# MAGIC - Create 7 numerical features for EG treatment actions by Wen
# MAGIC - Note for Enabler data, need to update the list of treatment actions based on business feedback
# MAGIC - Also update on 07/15, to create numerical feature for treatment path combo

# COMMAND ----------

# update with Daan's df on 07/14:
#df0 = pd.read_csv('df_iask_enabler_7_9_2021.csv', index_col=0)

# use the updated dataframe from previous code session
df0= df_merged.copy()

print(df0.shape)
df0.head()

# COMMAND ----------

df0['BAN'].nunique()

# COMMAND ----------

#update on 07/14, create numerical feature for treatment path combo:

# COMMAND ----------

df0['TREAT_MSG_CD_COMBO_FROM_DLNQ_TO_TDY'].nunique()

# COMMAND ----------

df0['TREAT_MSG_CD_COMBO_STRIPPED_FROM_DLNQ_TO_TDY'].nunique()

# COMMAND ----------

df0['TREAT_MSG_CD_COMBO_FROM_DLNQ_TO_TDY'].value_counts().reset_index(name='FREQ')

# COMMAND ----------

base_treat_path = 'EMLTY'

def get_treat_path_match_score(treat_combo_to_tdy):
    if str(treat_combo_to_tdy) in('A', 'NONE'): return fuzz.ratio(base_treat_path.lower(), '')
    action_list = treat_combo_to_tdy.split('&')
    #print(action_list)
    for i, act in enumerate(action_list):
        action_list[i] = act.strip()
        # use no-treat action name NONE to replace 'A' ?
        #if act.strip() == 'A': action_list[i] = 'NONE'  
    new_actions = ' '.join(action_list)
    #print(new_actions)
    return fuzz.ratio(base_treat_path.lower(), new_actions.lower())

# COMMAND ----------

get_treat_path_match_score('EMLPR|1PDTX & A')

# COMMAND ----------

df0['TREAT_COMBO_FROM_DLNQ_TO_TDY_NUM_SCORE'] = df0['TREAT_MSG_CD_COMBO_FROM_DLNQ_TO_TDY'].apply(get_treat_path_match_score)

# COMMAND ----------

df0['TREAT_COMBO_FROM_DLNQ_TO_TDY_NUM_SCORE'].hist()

# COMMAND ----------

df0[['TREAT_MSG_CD_COMBO_FROM_DLNQ_TO_TDY', 'TREAT_COMBO_FROM_DLNQ_TO_TDY_NUM_SCORE']].head(50)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

df0['TREAT_MSG_CD'] = df0['TREAT_MSG_CD'].astype(str)

print(df0['TREAT_MSG_CD'].nunique())
df0['TREAT_MSG_CD'].value_counts()    # note that there is no N/A with other treatments together, after Daan added OB and AC into actions

# COMMAND ----------

df0['TREAT_MSG_CD'].unique()

# COMMAND ----------

# hard code the list of pre-suspend actions for EG, based on notice list - updated on 07/06/2021
# just use a single code 'AOB' to represent AOB call on the grid

list_valid_presusp_actions = \
['EMLPRESN',
 'EMLPSN1',
 'EMLPSN2',
 'EMLRE',
 'EMLSN',
 'EMLTY',
 'PTPSN1',
 'PTPSN2',
 'SMSPD',
 'SMSPRESN',
 'UDASMSG1',
 'UDASMSG2',
 'VSN',
 'AOB']

print(len(list_valid_presusp_actions))

# COMMAND ----------

# hard code the dictionary map of action groups for enabler data - updated on 07/06/2021:
# (How does AOB fit here in hierarchical groups? (e.g., try to put it in group 2)

group_action_dict = {1: ['SMSPD', 'SMSPRESN', 'EMLTY', 'EMLRE', 'EMLPRESN', 'UDASMSG1'],
                     2: ['UDASMSG2', 'VSN', 'EMLSN', 'AOB'],
                     3: ['EMLPSN1', 'EMLPSN2', 'PTPSN1', 'PTPSN2']}  

# COMMAND ----------

## clean up the field 'TREAT_MSG_CD': if N/A appears with other treatments, remove N/A

# update on May 06: note that there is no N/A with other treatments together, after Daan added OB and AC into actions. 
# Need to consolidate 'AOB' call - update on July 06

def clean_treat_msg_cd(treat_msg_cd):
    ls_actions = treat_msg_cd.split('|')
    if len(ls_actions) > 1 and 'N/A' in ls_actions: ls_actions.remove('N/A')
    clean_ls_actions = []
    for i in ls_actions:
        if i in ['AD CALL', 'OUTBOUNDCALL'] and 'AOB' not in clean_ls_actions: clean_ls_actions.append('AOB')
        #only keep valid action and non-duplicate for a single day
        elif i in list_valid_presusp_actions and i not in clean_ls_actions: clean_ls_actions.append(i)       
        
    if not clean_ls_actions: return 'NONE'  # if no any valid action exists

    return '|'.join(clean_ls_actions)

clean_treat_msg_cd('AD CALL|CHAMPPRE|AD CALL|EMLRE|SMSPD|UDASMSG1')

# COMMAND ----------

clean_treat_msg_cd('CHAMPPRE')

# COMMAND ----------

clean_treat_msg_cd('NONE')

# COMMAND ----------

df0['TREAT_MSG_CD_CLEAN'] = df0['TREAT_MSG_CD'].apply(clean_treat_msg_cd)

print(df0['TREAT_MSG_CD_CLEAN'].nunique())
df0['TREAT_MSG_CD_CLEAN'].value_counts()

# COMMAND ----------

list_valid_act_from_df = list(df0['TREAT_MSG_CD_CLEAN'].unique())
print(len(list_valid_act_from_df))

# COMMAND ----------

# check any single daily action in the pre-susp list but not appearing in our dataframe:  
# (all CALLS from Una's list, need to refresh dataframe with those call-related action code!!)
print(len([i for i in list_valid_presusp_actions if i not in list_valid_act_from_df]))
[i for i in list_valid_presusp_actions if i not in list_valid_act_from_df]

# COMMAND ----------

# consolidate a df with all existing actions in cohort dataframe and those in valid pre-susp list
all_act_list0 = list(df0['TREAT_MSG_CD_CLEAN'].unique()) + list_valid_presusp_actions
print(len(all_act_list0))
all_act_list = list(set(all_act_list0))
print(len(all_act_list))

# COMMAND ----------

df_all_act = pd.DataFrame({'All_Treat_Act_Cd': all_act_list})
df_all_act

# COMMAND ----------

def get_group_nbr(treat_msg_cd):
    if not treat_msg_cd or treat_msg_cd == 'NONE': return 'NoTreatment'
    ls_actions = treat_msg_cd.split('|')
    if len(ls_actions) > 1 and 'N/A' in ls_actions: ls_actions.remove('N/A')
    #print(ls_actions)
    group_ls = []
    for action in ls_actions:
        action = action.strip()
        for key, val in group_action_dict.items():
            if action.upper() in val:
                if str(key) not in group_ls: group_ls.append(str(key))
                break
        else:
            if 'NoMatched' not in group_ls: group_ls.append('NoMatched')
        
    return '&'.join(group_ls)
    
    
df0['MATCH_GROUP'] = df0['TREAT_MSG_CD_CLEAN'].apply(get_group_nbr)


# COMMAND ----------

df0['MATCH_GROUP'].value_counts()

# COMMAND ----------

#df0.loc[df0['MATCH_GROUP'] == 'NoMatched', ['TREAT_MSG_CD', 'TREAT_MSG_CD_CLEAN', 'MATCH_GROUP']].head()

# COMMAND ----------

df0.shape

# COMMAND ----------

# Note below df is only based on all existing treat_msg_cd in the dataframe

df_rank_output = df0[['MATCH_GROUP', 'TREAT_MSG_CD_CLEAN']].groupby(by=['MATCH_GROUP', 'TREAT_MSG_CD_CLEAN']).size().reset_index(name = 'FREQ').sort_values(by=['MATCH_GROUP', 'FREQ'], ascending=[True, False])
df_rank_output = df_rank_output.reset_index(drop=True)

# COMMAND ----------

# update by 07/06/2021: per discussion with Sasha, we use pure preq_rank, instead of group_freq_rank, since there might be
# multiple actions from different groups

# df_rank_output['dense_rank'] = df_rank_output.groupby(by='MATCH_GROUP')['FREQ'].rank(method='dense', ascending=False)
# df_rank_output['GROUP_FREQ_RANK'] = df_rank_output.apply(lambda x: int(x['MATCH_GROUP']) + x['dense_rank'] * 0.1 if x['MATCH_GROUP'].isnumeric() else None, axis=1)
# df_rank_output

# COMMAND ----------

# merge with a more complete list of treat actions (in case there are some valid actions not existing in our cohort dataframe)

df_rank_output02 = df_all_act.rename(columns={'All_Treat_Act_Cd': 'TREAT_MSG_CD_CLEAN'}).merge(df_rank_output, how='left', on='TREAT_MSG_CD_CLEAN')
df_rank_output02 = df_rank_output02.reset_index(drop=True)
df_rank_output02.FREQ = df_rank_output02.FREQ.fillna(0)

# COMMAND ----------

# update on 07/06: per discussion with Sasha, we use pure preq_rank, instead of group_freq_rank, since there might be
# multiple actions from different groups
df_rank_output02['FREQ_RANK'] = df_rank_output02['FREQ'].rank(method='dense', ascending=False)
df_rank_output02.sort_values(by='FREQ', ascending=False)

# COMMAND ----------

# Note for any valid actions not existing in our cohort dataframe, no value for Group_freq_rank,
# but we can still derive numerical values for other remaining attributes like similarity scores and fast response score..

# COMMAND ----------

# update for enabler data - on 06/25/2021
# for numeric attribute 2: word similarity score compared with base word 'EMLTY'

# use empty string '' to represent no treatment

# COMMAND ----------

base_word = 'EMLTY' # choose activity code 'EMLTY' for enabler

def get_string_match_score(treat_msg_cd_clean):
    if str(treat_msg_cd_clean) == 'NONE': return fuzz.ratio(base_word.lower(), '')
    action_list = treat_msg_cd_clean.split('|')
    new_actions = ' '.join(action_list)
    return fuzz.ratio(base_word.lower(), new_actions.lower())

df_rank_output02['WORD_SIMILARITY_SCORE'] = df_rank_output02['TREAT_MSG_CD_CLEAN'].apply(get_string_match_score)

# COMMAND ----------

def get_text_cosine_similarity(treat_msg_cd_clean):
    if str(treat_msg_cd_clean) == 'NONE': return textdistance.cosine(base_word.lower(), '')
    action_list = treat_msg_cd_clean.split('|')
    new_actions = ' '.join(action_list)
    return textdistance.cosine(base_word.lower(), new_actions.lower())

df_rank_output02['WORD_COSINE_SIMILARITY'] = df_rank_output02['TREAT_MSG_CD_CLEAN'].apply(get_text_cosine_similarity)

# COMMAND ----------

# for numeric attribute: response scores of communication, SMS(5)/Push(4, app, tv)/Email(3)/AOB (2)/Mail (1)/nothing (0) 2 combined methods var1 (3) var2 (5) single var1(3) var2 (0)

# COMMAND ----------

# map action/activity to its type

# Hard coded the dictionary map for Enabler - update on 07/06/2021:

dict_presusp_action_type_enabler = {'EMLPRESN': 'Email',
                                   'EMLPSN1': 'Email',
                                   'EMLPSN2': 'Email',
                                   'EMLRE': 'Email',
                                   'EMLSN': 'Email',
                                   'EMLTY': 'Email',
                                   'PTPSN1': 'Letter',
                                   'PTPSN2': 'Letter',
                                   'SMSPD': 'SMS',
                                   'SMSPRESN': 'SMS',
                                   'UDASMSG1': 'TV',
                                   'UDASMSG2': 'TV',
                                   'VSN': 'Letter',
                                    'AOB':  'AOB'}



dict_type_attr_score_enabler = {'SMS': 5, 'TV': 4, 'Email': 3, 'AOB': 2, 'Letter': 1, 'Nothing': 0}

# COMMAND ----------

# 13 actions + 1 call actions = 14

print(len(dict_presusp_action_type_enabler)) 

# COMMAND ----------

max_mutliple_acts = 4  # according to the current cohort data, there could be up to 4 actions on a single day

def get_sync_communication_score(treat_msg_cd_clean, max_mutliple_acts):
    #if not treat_msg_cd_clean or str(treat_msg_cd_clean) == 'nan': return [0, 0]
    if not treat_msg_cd_clean or str(treat_msg_cd_clean) == 'NONE': return [0] * max_mutliple_acts
    action_list = treat_msg_cd_clean.split('|')
    output_scores = []
    for action in action_list:
        if action in dict_presusp_action_type_enabler.keys():
            match_type = dict_presusp_action_type_enabler[action]
            match_sync_score = dict_type_attr_score_enabler.get(match_type)
            output_scores.append(match_sync_score)
        else: 
            output_scores.append(None)
    if len(output_scores) < max_mutliple_acts: output_scores += [0] * (max_mutliple_acts - len(output_scores))
    if len(output_scores) > max_mutliple_acts: output_scores = output_scores[:max_mutliple_acts]
    return output_scores
        
    

# COMMAND ----------

df_rank_output02['TYPE_ATTR_VAL_COMB'] = df_rank_output02.apply(lambda x: get_sync_communication_score(x['TREAT_MSG_CD_CLEAN'], max_mutliple_acts), axis=1)
#df_rank_output02

# COMMAND ----------

# Note that for EG, now we have 4 columns for response time score attributes!

df_rank_output02[['TYPE_ATTR_VAL_01','TYPE_ATTR_VAL_02', 'TYPE_ATTR_VAL_03', 'TYPE_ATTR_VAL_04']] \
= pd.DataFrame(df_rank_output02['TYPE_ATTR_VAL_COMB'].tolist(), index= df_rank_output02.index)

# COMMAND ----------

df_rank_output02 = df_rank_output02.drop(['TYPE_ATTR_VAL_COMB'], axis=1)

# COMMAND ----------

df_rank_output02 = df_rank_output02.drop(['MATCH_GROUP', 'FREQ'], axis=1)   
#note: need to putput this dataframe for use of grid code (we need to include 7 numerical features with every grid action)

df_rank_output02

# COMMAND ----------

# quick check on treat_msg_cd_clean unique action values:
list_check = df_rank_output02['TREAT_MSG_CD_CLEAN'].unique().tolist()
print(len(list_check))

list_single_act = []
for act in list_check:
    list_single_act.extend(act.split('|'))
list_nondup_single_act = list(set(list_single_act))

print(len(list_nondup_single_act))   # 13 valid pre-susp actions plus NONE plus AOB = 15
list_nondup_single_act     

# COMMAND ----------

#output this file for grid coding use!
#df_rank_output02.to_csv('treat_action_features_dataframe_update0715.csv', index=False)  

# COMMAND ----------

# Finally, add the newly created numerical features to the current cohort dataframe:

# COMMAND ----------

df0.columns

# COMMAND ----------

df0.shape

# COMMAND ----------

# Add additional condition before doing drop here: if those columns already exist in df0.columns, then drop them before merging with df_rank_output
list_of_numerical_attrs = ['FREQ_RANK', 'WORD_SIMILARITY_SCORE', 'WORD_COSINE_SIMILARITY', 
                             'TYPE_ATTR_VAL_01', 'TYPE_ATTR_VAL_02', 'TYPE_ATTR_VAL_03', 'TYPE_ATTR_VAL_04']

for col in list_of_numerical_attrs:
    if col in df0.columns:
        df0 = df0.drop([col], axis=1)
        
print(df0.shape)

# COMMAND ----------

df_new = df0.merge(df_rank_output02, how='left', on = 'TREAT_MSG_CD_CLEAN')

# COMMAND ----------

df_new.shape

# COMMAND ----------

df_new.columns

# COMMAND ----------

# remove unnecessary columns and output the new dataframe:
df_new = df_new.drop(['MATCH_GROUP'], axis=1)
df_new.shape

# COMMAND ----------

df_new.head()

# COMMAND ----------

df_new.to_csv('df_iask_enabler_0720_Wen.csv')

# COMMAND ----------

# note in the updated df, we add 7 numerical grid features, 1 'TREAT_MSG_CD_CLEAN', 
# and 1 numerical feature for treat path combo 'TREAT_COMBO_FROM_DLNQ_TO_TDY_NUM_SCORE'

# COMMAND ----------

# df_new is the final dataframe with all the features we have created
