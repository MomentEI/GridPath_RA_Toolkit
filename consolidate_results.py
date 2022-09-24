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
Consolidates unserved energy results from GridPath simulation

To call in command line:
python consoliate_results.py [subscenario_name]

Notes:
Call after GridPath simulation has completed

"""

import sys
import os
import csv
import numpy as np

scenario = sys.argv[1]

if os.path.exists('Results') == False:
    os.mkdir('Results')
if os.path.exists(os.path.join('Results',scenario)) == False:
    os.mkdir(os.path.join('Results',scenario))

# determine if the scenario is a subscenario
with open(os.path.join('settings','subscenarios.csv')) as csvfile:
    csvreader = csv.reader(csvfile)
    subscenario_names = csvreader.__next__()
    base_scenarios = csvreader.__next__()
if scenario in subscenario_names:
    subscenario_ind = subscenario_names.index(scenario)
    base_scenario = base_scenarios[subscenario_ind]
else:
    base_scenario = scenario

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


N_weeks = 52*iterations
max_timepoint = iterations*100*1000 + 52*1000 + 168
USE = np.zeros(max_timepoint)
subproblem = np.zeros(max_timepoint,dtype=int)

complete_count = 0
for i in range(N_weeks):
    
    sub = i + 1
    print(sub)
    subproblem_path = os.path.join('Simulations',scenario,str(sub),'results')
    if os.path.exists(subproblem_path):
        complete_count += 1
        
        if os.path.exists(os.path.join(subproblem_path,'load_balance.csv')):
            with open(os.path.join(subproblem_path,'load_balance.csv')) as csvfile:
                csvreader = csv.reader(csvfile)
                csvreader.__next__()
                for row in csvreader:
                    timepoint = int(row[2])
                    USE[timepoint-1] += float(row[9])
                    subproblem[timepoint-1] = sub

print('')
print(str(complete_count) + ' subproblems complete.')
print('')

with open(os.path.join('Results',scenario,'region_USE_hourly.csv'),'w',newline = '') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['scenario','subproblem_id','timepoint','unserved_energy_mw'])
    for i in range(iterations):
        print(i+1)
        for j in range(52):
            for k in range(168):
                timepoint = (i+1)*100*1000 + (j+1)*1000 + (k+1)
                if USE[timepoint-1] > 0:
                    csvwriter.writerow([scenario,subproblem[timepoint-1],timepoint,USE[timepoint-1]])