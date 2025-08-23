# Elasto-dynamic conditions  
## Code execution order:  
1. FE_full_elasto_dynamics.py is the script to generate the ground truth of an linear elastic square under static loading by FEM.  
   The results are saved in folder: FE_full_elasto_dynamic_ground_truth.  

2. Prepare_DeepONet_hyper_elastic_quasi_static.py is the script to train the Deep Operator Neural Network (DeepONet) for different time intervals.  
   The results are saved in folder: Pretrained_DeepONet_hyper_elastic_quasi_static.  

3. FE_DeepONet_hyper_elasticity_quasi_static_coupling.py is the framework which couples the FE and DeepONet to solve the hyper-elasticity under quasi-static loading. The results are saved in folder: FE_DeepONet_hyper_elasticity_quasi_static_coupling_results.

   Note: The prerequisite for running **FE_DeepONet_hyper_elasticity_quasi_static_coupling.py** is that **DeepONet_hyper_elasticity_quasi_static.pkl** already exists in directory Pretrained_DeepONet_hyper_elasticity_quasi_static.
