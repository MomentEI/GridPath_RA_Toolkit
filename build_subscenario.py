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
Generates the  GridPath input data for a scenario that represents a subset of an already-generated scenario (the "base_scenario")
This can be used to simulate only a subset of the loads/resources/zones from a prior run under the same conditions 

To call in command line:
python build_subscenario.py [subscenario_name]

Notes:
settings must be populated for [subscenario_name] in subscenarios.csv
files and directories listed in subscenarios.csv for [subscenario_name] must be populated
settings must be populated for the specified [base_scenario] in scenario_settings.csv
[base_scenario] must have already been built

"""

import sys
import os
import csv
import glob
import shutil


subscen = sys.argv[1]

print('removing old directories...')

if os.path.isdir(os.path.join('Simulations',subscen)) == True:
    shutil.rmtree(os.path.join('Simulations',subscen))
     
if os.path.isdir(os.path.join('Simulations',subscen+'_log')):
    shutil.rmtree(os.path.join('Simulations',subscen+'_log'))

print('importing subscenario settings...')
# import subscenario settings
with open(os.path.join('settings','subscenarios.csv')) as csvfile:
    csvreader = csv.reader(csvfile)
    subscenario_names = csvreader.__next__()
    subscenario_ind = subscenario_names.index(subscen)
    base_scenario = csvreader.__next__()[subscenario_ind]
    aggregation_list = csvreader.__next__()[subscenario_ind]
    common_files = csvreader.__next__()[subscenario_ind]

# import aggregation list
aggregations = []
with open(os.path.join('settings',aggregation_list)) as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        aggregations.append(row[0])

# create a directory for the subscenario
if os.path.exists(os.path.join('Simulations',subscen)) == False:
    os.path.join('Simulations',subscen)
if os.path.exists(os.path.join('Simulations',subscen+'_log')) == False:
    os.mkdir(os.path.join('Simulations',subscen+'_log'))

# copy draw data to subscenario
shutil.copy2(os.path.join('Simulations',base_scenario+'_log','draw_data.csv'),os.path.join('Simulations',subscen+'_log'))

# loop through contents of base scenario
print('printing subscenario inputs...')
for f in os.listdir(os.path.join('Simulations',base_scenario)):
    
    if os.path.isdir(os.path.join('Simulations',base_scenario,f)):

        os.makedirs(os.path.join('Simulations',subscen,f,'inputs'))
        
        # copy common files over to draw
        files = glob.glob(os.path.join('common_files',common_files,'subproblems','*'))
        for file in files:
            shutil.copy(file,os.path.join('Simulations',subscen,f,'inputs'))
        
        # copy temporal files over to draw
        shutil.copy2(os.path.join('Simulations',base_scenario,f,'inputs','horizon_timepoints.tab'),os.path.join('Simulations',subscen,f,'inputs'))
        shutil.copy2(os.path.join('Simulations',base_scenario,f,'inputs','horizons.tab'),os.path.join('Simulations',subscen,f,'inputs'))
        shutil.copy2(os.path.join('Simulations',base_scenario,f,'inputs','periods.tab'),os.path.join('Simulations',subscen,f,'inputs'))
        shutil.copy2(os.path.join('Simulations',base_scenario,f,'inputs','timepoints.tab'),os.path.join('Simulations',subscen,f,'inputs'))
        
        # write draw files for subscenario
        with open(os.path.join('Simulations',subscen,f,'inputs','project_availability_exogenous.tab'),'w',newline='') as subfile:
            subwriter = csv.writer(subfile,delimiter = '\t')
            with open(os.path.join('Simulations',base_scenario,f,'inputs','project_availability_exogenous.tab')) as basefile:
                basereader = csv.reader(basefile,delimiter = '\t')
                subwriter.writerow(basereader.__next__())
                for row in basereader:
                    if row[0] in aggregations:
                        subwriter.writerow(row)
        
        with open(os.path.join('Simulations',subscen,f,'inputs','variable_generator_profiles.tab'),'w',newline='') as subfile:
            subwriter = csv.writer(subfile,delimiter = '\t')
            with open(os.path.join('Simulations',base_scenario,f,'inputs','variable_generator_profiles.tab')) as basefile:
                basereader = csv.reader(basefile,delimiter = '\t')
                subwriter.writerow(basereader.__next__())
                for row in basereader:
                    if row[0] in aggregations:
                        subwriter.writerow(row)
        
        with open(os.path.join('Simulations',subscen,f,'inputs','hydro_conventional_horizon_params.tab'),'w',newline='') as subfile:
            subwriter = csv.writer(subfile,delimiter = '\t')
            with open(os.path.join('Simulations',base_scenario,f,'inputs','hydro_conventional_horizon_params.tab')) as basefile:
                basereader = csv.reader(basefile,delimiter = '\t')
                subwriter.writerow(basereader.__next__())
                for row in basereader:
                    if row[0] in aggregations:
                        subwriter.writerow(row)
        
        with open(os.path.join('Simulations',subscen,f,'inputs','load_mw.tab'),'w',newline='') as subfile:
            subwriter = csv.writer(subfile,delimiter = '\t')
            with open(os.path.join('Simulations',base_scenario,f,'inputs','load_mw.tab')) as basefile:
                basereader = csv.reader(basefile,delimiter = '\t')
                subwriter.writerow(basereader.__next__())
                for row in basereader:
                    if row[0] in aggregations:
                        subwriter.writerow(row)
        
        
    # copy any other files in the scenario directory    
    else:
        shutil.copy2(os.path.join('Simulations',base_scenario,f),os.path.join('Simulations',subscen))
        
print('done.')
