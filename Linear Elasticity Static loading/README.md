# Linear Elasticity under static loading conditions  
## Code execution order:  
1. FE_full_static.py is the script to generate the ground truth of an intact linear elastic square under static loading.  
   The results are saved in folder: static_data_ground_truth.  

2. Prepare_DeepONet_static.py is the script to train the Deep Operator Neural Network (DeepONet).  
   The results are saved in folder: Pretrained_DeepONet_static.  

3. FE_DeepONet_static_coupling.py is the framework which couples the FE and DeepONet to solve the linear elasticity under static loading.  
   The results are saved in folder: FE_DeepONet_static_coupling_results.
     
   Note: The prerequisite for running **FE_DeepONet_static_coupling.py** is that **DeepONet_static.pkl** already exists in directory Pretrained_DeepONet_static.
   
## results: 
![Linear-elastic_static_loadings_u_x](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Linear%20Elasticity%20Static%20loading/readme_figures_LE/Fig.4_linear_static_coupling_u.jpg)

![Linear-elastic_static_loading_v](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Linear%20Elasticity%20Static%20loading/readme_figures_LE/Fig.4_linear_static_coupling_v.jpg)
