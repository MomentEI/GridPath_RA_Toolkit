# -*- coding: utf-8 -*-
"""
Copyright 2022 Moment Energy Insights LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

"""
Processes unserved energy results from GridPath simulation and prints loss of load metrics and other information

To call in command line:
python process_results.py [subscenario_name]
if adjusting for imports based on a prior regional run (regional_scenario):
python process_results.py [subscenario_name] --imports [regional_scenario]

Notes:
Call after consolidating unserved results using consolidate_results.py

"""



import sys
import os
import csv
import numpy as np
import pandas as pd


scenario_name = sys.argv[1]
if '--imports' in sys.argv:
    import_ind = sys.argv.index('--imports')
    import_flag = 1
    import_case = sys.argv[import_ind+1]
else:
    import_flag = 0


# determine if the scenario is a subscenario
with open(os.path.join('settings','subscenarios.csv')) as csvfile:
    csvreader = csv.reader(csvfile)
    subscenario_names = csvreader.__next__()
    base_scenarios = csvreader.__next__()
if scenario_name in subscenario_names:
    subscenario_ind = subscenario_names.index(scenario_name)
    base_scenario = base_scenarios[subscenario_ind]
else:
    base_scenario = scenario_name

# read in scenario information
with open('settings/scenario_settings.csv') as csvfile:
    file_reader = csv.reader(csvfile)
    scenarios = file_reader.__next__()
    if base_scenario not in scenarios:
        print('Error - scenario not listed in scenario_settings.csv')
    else:
        scenario_ind = scenarios.index(base_scenario)
        study_year = int(file_reader.__next__()[scenario_ind])
        weather_mode = file_reader.__next__()[scenario_ind]
        opt_window = file_reader.__next__()[scenario_ind]
        iterations = int(file_reader.__next__()[scenario_ind])
        parameters_file = file_reader.__next__()[scenario_ind]
        aggregations_file = file_reader.__next__()[scenario_ind]
        timeseries_file = file_reader.__next__()[scenario_ind]
        vartypes_file = file_reader.__next__()[scenario_ind]
        common_files = file_reader.__next__()[scenario_ind]



first_weather_year= 1991
last_weather_year = 2020
z_unc = 1.96

# Import day draw information
print('Importing draw information...')
draw_horizon = []
draw_dayofweek = []
draw_hydroyear = []
draw_month = []
draw_weatherbin = []
with open(os.path.join('Simulations',scenario_name+'_log','draw_data.csv')) as csvfile:
    csvreader = csv.reader(csvfile)
    csvreader.__next__()
    for row in csvreader:
        draw_horizon.append(float(row[0]))
        draw_dayofweek.append(int(np.mod(int(float(row[1]))-1,7)+1))
        draw_hydroyear.append(int(float(row[2])))
        draw_month.append(int(float(row[3])))
        draw_weatherbin.append(int(float(row[4])))
draw_horizon = np.array(draw_horizon,dtype=int)
draw_dayofweek = np.array(draw_dayofweek)
draw_month = np.array(draw_month)
draw_weatherbin = np.array(draw_weatherbin)

# Import weather bin information
print('Importing weather bin information...')
bin_year = []
bin_month = []
bin_weatherbin = []
with open(os.path.join('bins','weather_bins.csv')) as csvfile:
    csvreader = csv.reader(csvfile)
    csvreader.__next__()
    for row in csvreader:
        year_tmp = int(row[0])
        if year_tmp >= first_weather_year and year_tmp <= last_weather_year:
            bin_year.append(year_tmp)
            bin_month.append(int(row[1]))
            bin_weatherbin.append(int(row[5]))
bin_year = np.array(bin_year)
bin_month = np.array(bin_month)
bin_weatherbin = np.array(bin_weatherbin)

draw_weights = np.zeros(len(draw_weatherbin))
# loop through each month and weatherbin and derive weights to adjust to the desired weather year range
for i in range(12):
    
    month_tmp = i+1
    
    # count the number of days that fall within the month in the study year
    if month_tmp < 12:
        E_daycount_tmp = (np.datetime64(str(study_year)+'-'+str(month_tmp+1).zfill(2)+'-01')-np.datetime64(str(study_year)+'-'+str(month_tmp).zfill(2)+'-01'))/np.timedelta64(1,'D')
    else:
        E_daycount_tmp = 31
    
    # count the average number of days per year in the Monte Carlo simulation within the month
    MC_daycount_tmp = np.sum(draw_month == month_tmp)/iterations
    
    draw_weights[draw_month == month_tmp] = E_daycount_tmp/MC_daycount_tmp
        



# Import import case
if import_flag == 1:
    print('Importing import case results...')
    import_timepoint = []
    import_unserved_energy = []
    with open(os.path.join('Results',import_case,'region_USE_hourly.csv')) as csvfile:
        csvreader = csv.reader(csvfile)
        csvreader.__next__()
        for row in csvreader:
            ue = float(row[3])
            # only import hours with non-zero unserved energy
            if ue > 0:
                import_timepoint.append(row[2])
                import_unserved_energy.append(ue)  


# import regional case
print('Importing regional case results...')
raw_subproblem = []
raw_timepoint = []
raw_unserved_energy = []
raw_imports = []
with open(os.path.join('Results',scenario_name,'region_USE_hourly.csv')) as csvfile:
    csvreader = csv.reader(csvfile)
    csvreader.__next__()
    for row in csvreader:
        ue = float(row[3])
        # only import hours with non-zero unserved energy
        if ue > 0: 
            raw_subproblem.append(row[1])
            raw_timepoint.append(row[2])
            raw_unserved_energy.append(ue)
            if import_flag == 1:
                try:
                    import_ind = import_timepoint.index(row[2])
                    raw_imports.append(ue - min(import_unserved_energy[import_ind],ue))
                except:
                    raw_imports.append(ue)
            else:
                raw_imports.append(0)
N = len(raw_timepoint) 
raw_subproblem = np.array(raw_subproblem,dtype=int)
raw_timepoint = np.array(raw_timepoint,dtype=int)
raw_unserved_energy = np.array(raw_unserved_energy,dtype=float)
raw_imports = np.array(raw_imports,dtype=float)


# sort the data by timepoint and subtract imports
tp_sort = np.argsort(raw_timepoint)
subproblem = np.take_along_axis(raw_subproblem, tp_sort,axis=0)
timepoint = np.take_along_axis(raw_timepoint, tp_sort,axis=0)
imports = np.take_along_axis(raw_imports, tp_sort,axis=0)
unserved_energy = np.take_along_axis(raw_unserved_energy, tp_sort,axis=0) - imports

# print hourly events with imports
if import_flag == 1:
    with open(os.path.join('Results',scenario_name,'Region_USE_Hourly_WithImports.csv'),'w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['scenario_id','subproblem','timepoint','unserved_energy_mw','imports'])
        for i in range(len(unserved_energy)):
            if unserved_energy[i] > 0:
                csvwriter.writerow(['',subproblem[i],timepoint[i],unserved_energy[i],imports[i]])

# determine temporal information for all timepoints
hour_of_week = np.mod(timepoint-1,1000)+1
day_of_week = np.ceil(hour_of_week/24)
iteration = np.ceil(subproblem/52)
HE = np.mod(hour_of_week-1,24) + 1
tp_tmp = (timepoint - hour_of_week)/1000
week_of_year = np.mod(tp_tmp-1,100)+1
day_of_year = np.array(np.ceil(hour_of_week/24) + (week_of_year-1)*7,dtype=int)
event_date = np.datetime64(str(study_year-1)+'-12-31') + day_of_year*np.timedelta64(1,'D')
month = np.array(pd.to_datetime(event_date).month)

# pull day scaling information for each timepoint
day_scale = np.zeros(len(timepoint))
for i in range(len(timepoint)):
    day_scale[i] = draw_weights[(draw_horizon == subproblem[i])*(draw_dayofweek == day_of_week[i])][0]


print('Calculating loss of load metrics...')
# calculate month-hour averages
LOLH_mo_hr = np.zeros([12,24])
EUE_mo_hr = np.zeros([12,24])
imports_mo_hr = np.zeros([12,24])
max_imports_mo_hr = np.zeros([12,24])
for mo in range(12):
    for hr in range(24):
        mo_hr_inds = (month == mo+1)*(HE == hr+1)
        if np.sum(mo_hr_inds > 0):
            LOLH_mo_hr[mo,hr] = np.sum((unserved_energy[mo_hr_inds] > 0)*day_scale[mo_hr_inds])/iterations
            EUE_mo_hr[mo,hr] = np.sum(unserved_energy[mo_hr_inds]*day_scale[mo_hr_inds])/iterations
            imports_mo_hr[mo,hr] = np.mean(imports[mo_hr_inds])
            max_imports_mo_hr[mo,hr] = np.max(imports[mo_hr_inds])
        else:
            imports_mo_hr[mo,hr] = np.nan
            max_imports_mo_hr[mo,hr] = np.nan

# remove timepoints where imports have eliminated unserved energy
if import_flag == 1:
    keep_inds = unserved_energy > 0
    timepoint = timepoint[keep_inds]
    unserved_energy = unserved_energy[keep_inds]
    iteration = iteration[keep_inds]
    day_scale = day_scale[keep_inds]
    day_of_year = day_of_year[keep_inds]
    month = month[keep_inds]
    HE = HE[keep_inds]
    subproblem = subproblem[keep_inds]
    day_of_week = day_of_week[keep_inds]

# process loss of load events
unique_events = np.unique([iteration,day_of_year],axis=1)
N_events = np.shape(unique_events)[1]
event_iteration = np.zeros(N_events,dtype=int)
event_month = np.zeros(N_events,dtype=int)
event_MW = np.zeros(N_events,dtype=float)
event_MWh = np.zeros(N_events,dtype=float)
event_duration = np.zeros(N_events,dtype=int)
event_weight = np.ones(N_events,dtype=float)
event_subproblem = np.zeros(N_events,dtype=int)
event_dayofweek = np.zeros(N_events,dtype=int)
for i in range(N_events):
    events_inds = (iteration == unique_events[0,i])*(day_of_year == unique_events[1,i])
    event_iteration[i] = unique_events[0,i]
    event_month[i] = month[events_inds][0]
    event_MW[i] = np.max(unserved_energy[events_inds])
    event_MWh[i] = np.sum(unserved_energy[events_inds])
    event_duration[i] = np.sum(events_inds)
    event_weight[i] = day_scale[events_inds][0]
    event_subproblem[i] = subproblem[events_inds][0]
    event_dayofweek[i] = day_of_week[events_inds][0]
    

# count total events
event_count = np.sum(event_weight)

    
# count total calendar days in study year
studyyear_calendar_days = (np.datetime64(str(study_year+1)+'-01-01') - np.datetime64(str(study_year)+'-01-01'))/np.timedelta64(1,'D')


# Calculate LOLP metrics
LOLP_year = len(np.unique(iteration))/iterations
LOLP_year_unc = np.sqrt(iterations*LOLP_year*(1-LOLP_year))*z_unc/iterations
LOLP_day = event_count/(iterations*studyyear_calendar_days)
LOLP_day_unc = np.sqrt(iterations*studyyear_calendar_days*LOLP_day*(1-LOLP_day))*z_unc/(iterations*studyyear_calendar_days)
LOLP_hour = np.sum(LOLH_mo_hr)/(studyyear_calendar_days*24)
LOLP_hour_unc = np.sqrt(iterations*studyyear_calendar_days*24*LOLP_hour*(1-LOLP_hour))*z_unc/(iterations*studyyear_calendar_days*24)

LOLE = LOLP_day*studyyear_calendar_days*10
LOLE_unc = LOLP_day_unc*studyyear_calendar_days*10
LOLH = LOLP_hour*studyyear_calendar_days*24
LOLH_unc = LOLP_hour_unc*studyyear_calendar_days*24
EUE = np.sum(EUE_mo_hr)

if EUE == 0:
    EUE_day_event = 0
    EUE_hour_event = 0
else:
    EUE_day_event = EUE/(LOLP_day*studyyear_calendar_days)
    EUE_hour_event = EUE/LOLH

if event_count == 0:
    Avg_duration = 0
else:
    Avg_duration = np.sum(event_duration*event_weight)/np.sum(event_weight)


print('Determining resource needs...')

# Determine resource needs to achieve LOLE = 1 day every 10 years
allowed_events = 1/10*iterations
cap_needed = np.zeros(24)
cap_unc = np.zeros(24)
for i in range(24):
    
    # determine how much capacity of duration i+1 hours would be required to eliminate each energy and capacity shortage
    cap_tmp = np.maximum(event_MWh/(i+1),event_MW)

    # sort in descending order
    ind_sort = np.argsort(-cap_tmp)
    cap_sorted = np.take_along_axis(cap_tmp, ind_sort,axis=0)
    weight_sorted = np.take_along_axis(event_weight, ind_sort,axis=0)
    
    # find the capacity that must be added so the number of remainined events is less than or equal to the number of allowed events
    allowed_ind = max(np.sum(np.cumsum(weight_sorted) <= allowed_events)-1,0)
    if allowed_events <= np.sum(weight_sorted):
        cap_needed[i] = cap_sorted[allowed_ind]

    # estimate 95% confidence interval
    a = (z_unc**2)/(iterations*365) + 1
    b = -(2*allowed_events + z_unc**2)
    c = allowed_events**2
    allowed_events_high = (-b+np.sqrt(b**2-4*a*c))/(2*a)
    allowed_events_low = (-b-np.sqrt(b**2-4*a*c))/(2*a)

    allowed_ind_high = max(np.sum(np.cumsum(weight_sorted) <= allowed_events_high)-1,0)
    allowed_ind_low = max(np.sum(np.cumsum(weight_sorted) <= allowed_events_low)-1,0)

    if len(cap_sorted) > 0:
        cap_unc[i] = max(cap_sorted[allowed_ind_low] - cap_sorted[allowed_ind],cap_sorted[allowed_ind] - cap_sorted[allowed_ind_high])
    else:
        cap_unc[i] = 0


        
# Calculate convergence metrics
print('Calculating convergence metrics...')
LOLE_conv = np.zeros(iterations)
LOLE_unc = np.zeros(iterations)
cap_conv = np.zeros(iterations)
cap_unc_up = np.zeros(iterations)
cap_unc_down = np.zeros(iterations)
for i in range(iterations):
    
    event_inds = event_iteration<=i+1
    
    event_count_tmp = np.sum(event_weight[event_inds])
    LOLE_conv[i] = event_count_tmp/(i+1)*10
    LOLE_unc[i] = z_unc*np.sqrt(event_count_tmp)/(i+1)*10
    
    allowed_events_tmp = 1/10*(i+1)
        
    # determine how much perfect capacity would be required to eliminate each energy and capacity shortage
    cap_tmp = event_MW[event_inds]

    # sort in descending order
    ind_sort = np.argsort(-cap_tmp)
    cap_sorted = np.take_along_axis(cap_tmp, ind_sort,axis=0)
    weight_sorted = np.take_along_axis(event_weight[event_inds], ind_sort,axis=0)
    
    # find the capacity that must be added so the number of remainined events is less than or equal to the number of allowed events
    allowed_ind = max(np.sum(np.cumsum(weight_sorted) <= allowed_events_tmp)-1,0)
    if allowed_ind < len(cap_tmp):
        cap_conv[i] = cap_sorted[allowed_ind]
        a = (z_unc**2)/((i+1)*365) + 1
        b = -(2*allowed_events_tmp + z_unc**2)
        c = allowed_events_tmp**2
        allowed_events_high = (-b+np.sqrt(b**2-4*a*c))/(2*a)
        allowed_events_low = (-b-np.sqrt(b**2-4*a*c))/(2*a)

        allowed_ind_high = max(np.sum(np.cumsum(weight_sorted) <= allowed_events_high)-1,0)
        allowed_ind_low = max(np.sum(np.cumsum(weight_sorted) <= allowed_events_low)-1,0)
        
        cap_unc_up[i] = cap_sorted[allowed_ind_low] - cap_conv[i]
        cap_unc_down[i] = cap_conv[i] - cap_sorted[allowed_ind_high]
        

print('Printing results...')
if import_flag == 1:
    filename = scenario_name+'_imports'
else:
    filename = scenario_name

with open(os.path.join('Results',scenario_name,filename+'_events.csv'),'w',newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Event Number','Iteration','Month','Subproblem','Day of Week','Capacity Shortage (MW)','Energy Shortage (MWh)','Weight (days/yr)'])
    for i in range(N_events):
        csvwriter.writerow([i+1,event_iteration[i],event_month[i],event_subproblem[i],event_dayofweek[i],event_MW[i],event_MWh[i],event_weight[i]])

with open(os.path.join('Results',scenario_name,filename+'_convergence.csv'),'w',newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Iteration','LOLE','LOLE uncertainty','Perfect Capacity Need','Capacity Need Uncertainty Down','Capacity Need Uncertainty Up'])
    for i in range(iterations):
        csvwriter.writerow([i+1,LOLE_conv[i],LOLE_unc[i],cap_conv[i],cap_unc_down[i],cap_unc_up[i]])

with open(os.path.join('Results',scenario_name,filename+'_summary.csv'),'w',newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    csvwriter.writerow(['Loss of Load Statistics'])
    csvwriter.writerow(['Metric','Value','Uncertainty'])
    csvwriter.writerow(['LOLP_year',LOLP_year,LOLP_year_unc])
    csvwriter.writerow(['LOLP_day',LOLP_day,LOLP_day_unc])
    csvwriter.writerow(['LOLP_hour',LOLP_hour,LOLP_hour_unc])
    csvwriter.writerow(['LOLE (days/10yrs)',LOLE,LOLE_unc[-1]])
    csvwriter.writerow(['LOLH (hrs/yrs)',LOLH,LOLH_unc])
    csvwriter.writerow(['EUE (MWh/yr)',EUE])
    csvwriter.writerow(['EUE_LOLday (MWh/loss-of-load-day)',EUE_day_event])
    csvwriter.writerow(['EUE_LOLhour (MW/loss-of-load-hour)',EUE_hour_event])
    csvwriter.writerow(['Average duration (hrs)',Avg_duration])

    csvwriter.writerow([''])

    csvwriter.writerow(['LOLH (hrs/yr) by month-hour'])
    row_tmp = ['']
    for i in range(12):
        row_tmp.append(i+1)
    csvwriter.writerow(row_tmp)
    for i in range(24):
        row_tmp = [i+1]
        for j in range(12):
            row_tmp.append(LOLH_mo_hr[j,i])
        csvwriter.writerow(row_tmp)
    
    csvwriter.writerow([''])
    
    csvwriter.writerow(['EUE (MWh/yr) by month-hour'])
    row_tmp = ['']
    for i in range(12):
        row_tmp.append(i+1)
    csvwriter.writerow(row_tmp)
    for i in range(24):
        row_tmp = [i+1]
        for j in range(12):
            row_tmp.append(EUE_mo_hr[j,i])
        csvwriter.writerow(row_tmp)
    
    csvwriter.writerow([''])
    
    row_tmp = ['Average imports during constrained hours (MW)']
    for i in range(12):
        row_tmp.append('')
    row_tmp.append('')
    row_tmp.append('Maximum imports during constrained hours (MW)') 
    csvwriter.writerow(row_tmp)
    
    row_tmp = ['']
    for i in range(12):
        row_tmp.append(i+1)
    row_tmp.append('')
    row_tmp.append('')
    for i in range(12):
        row_tmp.append(i+1)  
    csvwriter.writerow(row_tmp)
    
    for i in range(24):
        row_tmp = [i+1]
        for j in range(12):
            if np.isnan(imports_mo_hr[j,i]):
                row_tmp.append('')
            else:
                row_tmp.append(imports_mo_hr[j,i])
        row_tmp.append('')
        row_tmp.append(i+1)
        for j in range(12):
            if np.isnan(max_imports_mo_hr[j,i]):
                row_tmp.append('')
            else:
                row_tmp.append(max_imports_mo_hr[j,i])
        csvwriter.writerow(row_tmp)
    
    csvwriter.writerow([''])

    csvwriter.writerow(['Event Duration Frequency'])
    csvwriter.writerow(['Duration (hrs)','Event Count','Events Per Year','Percent of Events'])
    for i in range(24):
        duration_count = np.sum(event_weight[event_duration == i+1])
        if event_count == 0:
            event_percent = 0
        else:
            event_percent = duration_count/event_count
        csvwriter.writerow([i+1,duration_count,duration_count/iterations,event_percent])
    
    csvwriter.writerow([''])

    csvwriter.writerow(['Capacity needed to meet one-day-in-10-year standard'])
    csvwriter.writerow(['Duration (hrs)','Capacity (MW)','Uncertainty (MW)'])
    for i in range(24):
        csvwriter.writerow([i+1,cap_needed[i],cap_unc[i]])
        

print('Complete.')



        
        
        
        



