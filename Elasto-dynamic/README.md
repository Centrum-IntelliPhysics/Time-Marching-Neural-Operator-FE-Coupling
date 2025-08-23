# Elasto-dynamic conditions  
## Code execution order:  
1. FE_full_elasto_dynamics.py is the script to generate the ground truth of an linear elastic square under static loading by FEM.  
   The results are saved in folder: FE_full_elasto_dynamic_ground_truth.  

2. prepare_DeepONet_elasto_dynamic_ts_ts0_ts1 is the script to train DeepONet for different time intervals [$ts_0$, $ts_1$].  
   The results are saved in folder: prepare_DeepONet_Elasto_dynamic_square_square_ts0_ts1.  

3. FE_DeepONet_coupling_Elasto_dynamic_suqare_square_89_139.py is the framework which couples the FE and DeepONet to solve the elasto_dynamic from 89 to 139 time steps. The results are saved in folder: FE_DeepONet_coupling_elasto_dynamic_suqare_square_89_139_vmax_vmin.

   Note: The prerequisite for running **FE_DeepONet_coupling_Elasto_dynamic_suqare_square_89_139.py** is that **DeepONet_ED_ts0_ts1.pkl** already exists in directory Pretrained_DeepONet_hyper_elasticity_quasi_static.
