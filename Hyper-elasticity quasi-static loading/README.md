# Hyper-elasticity under quasi-static loading conditions  
## Code execution order:  
1. FE_full_square_hyper.py is the script to generate the ground truth of an intact linear elastic square under static loading.  
   The results are saved in folder: static_data_ground_truth.  

2. Prepare_DeepONet_hyper_elastic_quasi_static.py is the script to train the Deep Operator Neural Network (DeepONet).  
   The results are saved in folder: Pretrained_DeepONet_hyper_elastic_quasi_static.  

3. FE_DeepONet_hyper_elasticity_quasi_static_coupling.py is the framework which couples the FE and DeepONet to solve the hyper-elasticity under quasi-static loading. The results are saved in folder: FE_DeepONet_hyper_elasticity_quasi_static_coupling_results.

   Note: The prerequisite for running **FE_DeepONet_hyper_elasticity_quasi_static_coupling.py** is that **DeepONet_hyper_elasticity_quasi_static.pkl** already exists in directory Pretrained_DeepONet_hyper_elasticity_quasi_static.

## results:
Response in x-direction ($u_x$) of the hyperelastic coupling model under quasi-static loading conditions: (a) Schematics of decomposed domains for spatial coupling framework, where the left and bottom edges have fixed boundary conditions and the top and right edges are subjected to an applied displacement at time step n in y-direction ($u_y$ = 0.05(n + 1)) and x-direction ($u_x$ = 0.05(n + 1)); (b) Ground truth displacement $u_x$ at time step n = 4 obtained by solving the intact domain using FEniCSx; The blue-dashed box contains: Columns 1-3 showing the evolution of ux in $Ω_I$ for FE-FE coupling (top row) and FE-NO coupling (bottom row) at time step n = 0,2,4, with column 4 displaying the absolute error between the converged solution at time step n = 4 and ground truth. The red-dashed box contains: Columns 1-3 showing the evolution of $u_x$ in $Ω_{II}$ for both coupling frameworks at the same time steps, with column 4 displaying the absolute error relative to the ground truth
![hyper_displacement_u](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Hyper-elasticity%20quasi-static%20loading/readme_figures_HP/Fig.14_hyper_u.jpg)

![hyper_displacement_v](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Hyper-elasticity%20quasi-static%20loading/readme_figures_HP/Fig.14_hyper_v.jpg)
