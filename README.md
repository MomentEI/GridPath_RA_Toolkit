# GridPath RA Toolkit
Companion code for running GridPath for resource adequacy (RA) applications.

## Quick How-To Guide

### 1. How to install GridPath:
The GridPath RA Toolkit is compatible with v0.14.1 of GridPath. You can download GridPath v0.14.1 here: 
[https://github.com/blue-marble/gridpath/releases/tag/v0.14.1](https://github.com/blue-marble/gridpath/releases/tag/v0.14.1)
You can find installation instructions for GridPath including how to download and install the Cbc solver here:
[https://gridpath.readthedocs.io/en/latest/installation.html](https://gridpath.readthedocs.io/en/latest/installation.html)

### 2. How to get the GridPath RA Toolkit accompanying code:
You can download the GridPath RA Toolkit accompanying code here: [https://github.com/MomentEI/GridPath_RA_Toolkit](https://github.com/MomentEI/GridPath_RA_Toolkit)
This includes the code that you’ll need to run and post-process RA cases in GridPath. It also includes input data for a simple toy scenario (ToyTest).

### 3. How to get the GridPath RA Toolkit Western US dataset:
You can download the full dataset used in the Western US RA case study here: (link to MonteCarlo_Inputs.zip download on website).
Note that running the full Western US scenarios requires substantial computing resources. A single run can take several days to complete on a machine with 32 CPUs and 128 MW of memory as well as require several hundred GB of disk space for the inputs and results data.

### 4. How to run the toy scenario
i. Open an Anaconda Prompt
ii. Activate the GridPath virtual environment you created in Step 1
iii. Navigate to the directory where the GridPath RA Toolkit code and data live
iv. Build the input data for the GridPath simulation for the ToyTest1 scenario: 'python build_scenario.py ToyTest1 [# of threads]'
v. Run the ToyTest1 scenario in GridPath: 'gridpath_run --log --results_export_rule USE --n_parallel_solve [# of threads] --scenario_location Simulations --scenario ToyTest1'
vi. Consolidate the unserved energy results from the GridPath simulation outputs: 'python consolidate_results.py ToyTest1'
vii. Process the unserved energy results and calculate RA metrics: 'python process_results.py ToyTest1'

### 5. How to run a subscenario
A subscenario considers only a subset of the loads, resources, and/or zones from a base scenario, but tests all the same conditions. Running a subscenario requires that the GridPath inputs have already been built for the base scenario. As an example, a simple toy subscenario (ToyTest1_NoGas) has been set up based on the ToyTest1 scenario to show how removing the gas resources would affect the toy system.  

i. Build the input data for the GridPath simulation for the ToyTest1_NoGas subscenario: 'python build_subscenario.py ToyTest1_NoGas'
ii. Run the ToyTest1_NoGas scenario in GridPath: 'gridpath_run --log --results_export_rule USE --n_parallel_solve [# of threads] --scenario_location Simulations --scenario ToyTest1_NoGas'
iii. Consolidate the unserved energy results from the GridPath simulation outputs: 'python consolidate_results.py ToyTest1_NoGas'
iv. Process the unserved energy results and calculate RA metrics: 'python process_results.py ToyTest1_NoGas'
