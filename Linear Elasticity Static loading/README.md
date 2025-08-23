# Linear Elasticity under static loading conditions  
## Code execution order:  
1. `FE_full_static.py is` the script to generate the ground truth for a linear elastic square under static loading by FEM.  
   The results are saved in folder: `static_data_ground_truth`.  

2. `Prepare_DeepONet_static.py` is the script to train the Physics-informed DeepONet.  
   The results are saved in folder: `Pretrained_DeepONet_static`.  

3. `FE_DeepONet_static_coupling.py` is the framework which couples the FE and DeepONet to solve the linear elasticity under static loading.  
   The results are saved in folder: `FE_DeepONet_static_coupling_results`.
     
   Note: The prerequisite for running **FE_DeepONet_static_coupling.py** is that **DeepONet_static.pkl** already exists in directory Pretrained_DeepONet_static.
   
## results: 
Response in x-direction ($u_x$) of the linear elastic coupling model under static loading conditions: (a) Schematic of decomposed domains for the spatial coupling framework, where the bottom edge has fixed boundary conditions and the top edge is subjected to an applied displacement in the y-direction (uy = 0.01); (b) Ground truth displacement ux obtained by solving the intact domain using FEniCSx. The blue-dashed box contains: Columns 1-3 showing the evolution of ux in ΩI for FE-FE coupling (top row) at iterations j = 0,8,16 and FE-NO coupling (bottom row) at iterations j = 0,5,11, with column 4 displaying the absolute error between the converged solution (j = 16 for FE-FE and j = 11 for FE-NO) and the ground truth. The red-dashed box contains: Columns 1-3 showing the evolution of ux in ΩII for both coupling frameworks at the same iterations, with column 4 similarly displaying the absolute error relative to the ground truth.
![Linear-elastic_static_loadings_u_x](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Linear%20Elasticity%20Static%20loading/readme_figures_LE/Fig.4_linear_static_coupling_u.jpg)

Response in y-direction ($u_y$) of the linear elastic coupling model under static loading conditions: (a) Schematic of decomposed domains for the spatial coupling framework, where the bottom edge has fixed boundary conditions and the top edge is subjected to an applied displacement in the y-direction (uy = 0.01); (b) Ground truth displacement uy obtained by solving the intact domain using FEniCSx. The blue-dashed box contains: Columns 1-3 showing the evolution of uy in ΩI for FE-FE coupling (top row) at iterations j = 0,8,16 and FE-NO coupling (bottom row) at iterations j = 0,5,11, with column 4 displaying the absolute error between the converged solution (j = 16 for FE-FE and j = 11 for FE-NO) and the ground truth. The red-dashed box contains: Columns 1-3 showing the evolution of uy in ΩII for both coupling frameworks at the same iterations, with column 4 similarly displaying the absolute error relative to the ground truth.
![Linear-elastic_static_loadings_u_y](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Linear%20Elasticity%20Static%20loading/readme_figures_LE/Fig.5_linear_static_coupling_v.jpg)
