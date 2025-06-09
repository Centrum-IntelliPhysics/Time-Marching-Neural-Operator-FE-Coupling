
# -*- coding: utf-8 -*-
"""PI DeepONet for 2D elasticity problem
   New structure for DeepONet
"""

# Commented out IPython magic to ensure Python compatibility.
import os
import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian, lax
from jax.example_libraries import optimizers
from jax.nn import relu
from jax import config
#from jax.ops import index_update, index
from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
#from torch.utils import data
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import math  
from scipy.interpolate import griddata
import pickle
import scipy
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import Rbf, interp1d, griddata
# %matplotlib inline

# region neural net
# Define the neural net
def MLP(layers, activation=relu):
  ''' Vanilla MLP'''
  def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(k1, (d_in, d_out))
          b = np.zeros(d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:])) # d_in d_out initiate
      return params
  def apply(params, inputs):
      for W, b in params[:-1]:
          outputs = np.dot(inputs, W) + b
          inputs = activation(outputs)
      W, b = params[-1]
      outputs = np.dot(inputs, W) + b
      return outputs
  return init, apply

# region model 
# Define the model
class PI_DeepONet:
    def __init__(self, branch_layers, trunk_layers, **ela_model):
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=np.tanh)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=np.tanh)
        
        #elastic_model 
        self.E = ela_model['E']
        self.nu = ela_model['nu']
        
        # Initialize
        branch_params = self.branch_init(rng_key = random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
        branch_params_v = self.branch_init(rng_key = random.PRNGKey(12341))
        branch_params_ax = self.branch_init(rng_key = random.PRNGKey(12342))
        branch_params_ay = self.branch_init(rng_key = random.PRNGKey(12343))
        #trunk_params_v = self.trunk_init(rng_key = random.PRNGKey(43211))
        
        params_u = (branch_params, trunk_params)
        params_v = (branch_params_v, trunk_params)

        params = (params_u, params_v)
        
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(1e-4,
                                                                      decay_steps=50000,
                                                                      decay_rate=0.9))
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()

        # Loggers
        self.loss_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    # region architecture 
    # Define DeepONet architecture
    @partial(jax.jit, static_argnums=(0,))
    def operator_net_u(self, params, u, x, y):
        branch_params, trunk_params = params
        h = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
        B = self.branch_apply(branch_params, u)  # B(u(xm)) 
        T = self.trunk_apply(trunk_params, h)
        print('B.shape=', B.shape, 'T.shape=', T.shape)
        # Compute the final output
        # Input shapes:
        # branch: [batch_size, 4m]
        # trunk: [p, 2]
        # output: [batch_size, p]
        outputs = np.einsum('ij,kj->ik', B, T)
        print(outputs.shape)
        return  outputs
    
    @partial(jax.jit, static_argnums=(0,))
    def operator_net_v(self, params, v, x, y):
        branch_params, trunk_params = params
        h = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
        B = self.branch_apply(branch_params, v)  # B(u(xm)) 
        T = self.trunk_apply(trunk_params, h)
        # Compute the final output
        # Input shapes:
        # branch: [batch_size, 4m]
        # trunk: [p, 2]
        # output: [batch_size, p]
        outputs = np.einsum('ij,kj->ik', B, T)
        return  outputs

    # region Def_grad 
    # Deformation gradient F = I + grad(u) 
    # F11 = 1 + u_x
    @partial(jax.jit, static_argnums=(0,))
    def F11(self, params_u, u, x, y):
        s_u_x= jax.jvp(lambda x: self.operator_net_u(params_u, u, x, y), (x,), (np.ones_like(x),))[1]
        
        return 1 + s_u_x

    # F12 = u_y
    @partial(jax.jit, static_argnums=(0,))
    def F12(self, params_u, u, x, y):
        s_u_y= jax.jvp(lambda y: self.operator_net_u(params_u, u, x, y), (y,), (np.ones_like(y),))[1]
        
        return s_u_y
    
    # F21 = v_x
    @partial(jax.jit, static_argnums=(0,))
    def F21(self, params_v, v, x, y):
        s_v_x= jax.jvp(lambda x: self.operator_net_v(params_v, v, x, y), (x,), (np.ones_like(x),))[1]
        
        return s_v_x
    
    # F22 = 1 + v_y
    @partial(jax.jit, static_argnums=(0,))
    def F22(self, params_v, v, x, y):
        s_v_y= jax.jvp(lambda y: self.operator_net_v(params_v, v, x, y), (y,), (np.ones_like(y),))[1]
        
        return 1 + s_v_y
    
    # the strain energy density function of hyper-elastic material
    @partial(jax.jit, static_argnums=(0,))
    def Psi(self, F11, F12, F21, F22):
        
        J_def = F11 * F22 - F12 * F21
        # assure the positive definiteness of J
        J = J_def
        #J = np.maximum(J_def, 1e-8) 
        kapa = self.E/3*(1-2*self.nu) # bulk modulus
        mu = self.E/(2*(1 + self.nu)) # shear modulus
        lmbda = self.nu*self.E/((1 + self.nu)*(1 - 2*self.nu)) # Lame's first parameter
        I1 = F11**2 + F12**2+ F21**2 + F22**2
        #epsilon = 1e-8 # smooth the log function
        Psi = 1/2*mu*(I1 - 3) + lmbda/2*np.log(J)**2 - mu*np.log(J)
        #Psi = mu / 2 * (I1 - 3 - 2 * np.log(J)) + lmbda / 2 * (J - 1) ** 2
        #Psi = mu/2 * (I1 - 3)  + lmbda/2 * (J - 1)**2
        return Psi

    # First derivative of Psi
    @partial(jax.jit, static_argnums=(0,))
    def Psi_deri(self, params_u, params_v, u, v, x, y):
        F11 = self.F11(params_u, u, x, y)
        F12 = self.F12(params_u, u, x, y)
        F21 = self.F21(params_v, v, x, y)
        F22 = self.F22(params_v, v, x, y)
        Psi = self.Psi(F11, F12, F21, F22)
        Psi_F11 = jax.jvp(lambda F11 : self.Psi(F11, F12, F21, F22), (F11,), (np.ones_like(F11),))[1]
        Psi_F12 = jax.jvp(lambda F12 : self.Psi(F11, F12, F21, F22), (F12,), (np.ones_like(F12),))[1]
        Psi_F21 = jax.jvp(lambda F21 : self.Psi(F11, F12, F21, F22), (F21,), (np.ones_like(F21),))[1]
        Psi_F22 = jax.jvp(lambda F22 : self.Psi(F11, F12, F21, F22), (F22,), (np.ones_like(F22),))[1]
        return Psi_F11, Psi_F12, Psi_F21, Psi_F22


    # region residual net 
    # Define ODE/PDE residual
    def residual_net(self, params, u, v, x, y):
        #s = self.operator_net(params, u, x, y)
        #s_t = grad(self.operator_net, argnums=3)(params, u, x, t)
        params_u, params_v = params
        
        F11 = self.F11(params_u, u, x, y)
        F12 = self.F12(params_u, u, x, y)
        F21 = self.F21(params_v, v, x, y)
        F22 = self.F22(params_v, v, x, y)

        Psi_dF11dx = jax.jvp(lambda x: self.Psi_deri(params_u, params_v, u, v, x, y)[0], (x,), (np.ones_like(x),))[1]
        Psi_dF12dx = jax.jvp(lambda x: self.Psi_deri(params_u, params_v, u, v, x, y)[1], (x,), (np.ones_like(x),))[1]
        Psi_dF21dy = jax.jvp(lambda y: self.Psi_deri(params_u, params_v, u, v, x, y)[2], (y,), (np.ones_like(y),))[1]
        Psi_dF22dy = jax.jvp(lambda y: self.Psi_deri(params_u, params_v, u, v, x, y)[3], (y,), (np.ones_like(y),))[1]

        ###Newton's second law in plane strain 
        # res0 = div(sigma) - f = 0 in configuration coordinates
        res0 = Psi_dF11dx + Psi_dF12dx
        res1 = Psi_dF21dy + Psi_dF22dy

        return res0, res1

    # region stress
    def stress(self, params, u, v, Y_star):
        params_u, params_v = params
        x, y = Y_star[:,0], Y_star[:,1]
        #u, v = u.reshape(-1,1), v.reshape(-1,1)
        
        #print(s_u.shape)
        s_u_y = jax.jvp(lambda y: self.operator_net_u(params_u, u, x, y), (y,), (np.ones_like(y),))[1]
        
        s_u_x= jax.jvp(lambda x: self.operator_net_u(params_u, u, x, y), (x,), (np.ones_like(x),))[1]

        s_v_y= jax.jvp(lambda y: self.operator_net_v(params_v, v, x, y), (y,), (np.ones_like(y),))[1]
                       
        s_v_x=  jax.jvp(lambda x: self.operator_net_v(params_v, v, x, y), (x,), (np.ones_like(x),))[1]

        para1 = self.E/((1 + self.nu)*(1-2*self.nu))
        sigma_x = para1*((1-self.nu) * s_u_x + self.nu * s_v_y) 
        sigma_y = para1*((1-self.nu) * s_v_y + self.nu * s_u_x) 
        sigma_xy = para1*(1-2*self.nu) * (s_u_y + s_v_x)/2  

        return sigma_x, sigma_y, sigma_xy

    # region loss 
    # Define boundary loss
    def loss_bcs(self, params, batch):
        inputs, outputs = batch
        u, v, h = inputs
        params_u, params_v = params
        # Compute forward pass
        s_u_pred = self.operator_net_u(params_u, u, h[:, 0], h[:, 1])
        s_v_pred = self.operator_net_v(params_v, v, h[:, 0], h[:, 1])
        # Compute loss
        #print('u,v=',np.mean((outputs[0] - s_u_pred)**2) + np.mean((outputs[1] - s_v_pred)**2))
        #print('a=',np.mean((outputs[2] - s_ax_pred)**2) + np.mean((outputs[3] - s_ay_pred)**2))
        loss =  np.mean((outputs[0] - s_u_pred)**2) + np.mean((outputs[1] - s_v_pred)**2)
        return loss

    # Define residual loss
    def loss_res(self, params, batch):
        # Fetch data
        # inputs: (u1, y), shape = (Nxm, m), (Nxm,1)
        # outputs: u2, shape = (Nxm, 1)
        ## u2 is s_res, in this case s_res = u(x)
        
        inputs, outputs = batch
        u, v, h = inputs
        # Compute forward pass
        res0, res1 = self.residual_net(params, u, v, h[:,0], h[:,1])
        #print(pred.shape, outputs.shape)
        # Compute loss
        loss = np.mean((outputs[0] - res0)**2) + np.mean((outputs[1] - res1)**2)
       
        return  loss

    # Define total loss
    def loss(self, params, bcs_batch, res_batch):
        loss_bcs = self.loss_bcs(params, bcs_batch)
        loss_res = self.loss_res(params, res_batch)
        loss =  loss_bcs + loss_res
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, bcs_batch, res_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, bcs_batch, res_batch)
        return self.opt_update(i, g, opt_state)

    # region training 
    # Optimize parameters in a loop
    def train(self, bcs_dataset, res_dataset, nIter = 10000):
        # Define data iterators
        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            bcs_batch= next(bcs_data)
            res_batch = next(res_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, bcs_batch, res_batch)

            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, bcs_batch, res_batch)
                loss_bcs_value = self.loss_bcs(params, bcs_batch)
                loss_res_value = self.loss_res(params, res_batch)

                # Store losses
                self.loss_log.append(loss_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)

                # Print losses
                pbar.set_postfix({'Loss': loss_value,
                                  'loss_bcs' : loss_bcs_value,
                                  'loss_physics': loss_res_value})

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, V_star, Y_star):
        params_u, params_v = params
        s_u_pred = self.operator_net_u(params_u, U_star, Y_star[:,0], Y_star[:,1])
        s_v_pred = self.operator_net_v(params_v, V_star, Y_star[:,0], Y_star[:,1])
        return s_u_pred, s_v_pred

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, U_star, V_star, Y_star):
        r_pred = vmap(self.residual_net, (None, 0, 0, 0))(params, U_star, V_star, Y_star[:,0], Y_star[:,1])
        return r_pred

# region utils 
#utils 
def plot_disp(X, Y, u, foldername, title):
    '''
    Plot the displacement field
    '''
    plt.figure(figsize=(10, 8.5))
    
    # Set a compact layout between the main plot and the colorbar
    ax = plt.gca()  # Get the current axis
    scatter = ax.scatter(X, Y, c=u, cmap='seismic')

    # Add a colorbar
    cbar = plt.colorbar(scatter, orientation='vertical', pad=0.02, location='right')  # Reduce the padding
    cbar.formatter = ScalarFormatter()  # Set the default formatter
    cbar.formatter.set_scientific(True)  # Enable scientific notation
    cbar.formatter.set_powerlimits((-2, 2))  # Show scientific notation for numbers smaller than 0.01
    cbar.ax.get_yaxis().offsetText.set_fontsize(32)  # Set scientific notation size  # fontsize of formatter 12
    #cbar.ax.get_yaxis().offsetText.set_x(1.1)  # Move scientific notation to the right 
    cbar.ax.tick_params(labelsize=32)  # Adjust the font size of the colorbar labels
    # Set font properties
    ax.set_xlabel('x', fontsize=32, fontweight='bold')
    ax.set_ylabel('Y', fontsize=32, fontweight='bold')
    ax.set_title(title, fontsize=40, fontweight='bold', y = 1.05)  # Increase the distance between the title and the plot

    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=32, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=32, length=5, width=1)  # Minor ticks

    # Adjust the subplot layout
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # The rect parameter controls the overall layout (reduce right margin)

    # Save and display
    plt.savefig(foldername + ".jpg", dpi=700, bbox_inches='tight')  # bbox_inches='tight' reduces the extra margins
    plt.show()
    plt.close()

def plot_relative_error(X, Y, u, foldername, title):
    '''
    Plot the displacement field
    '''
    plt.figure(figsize=(10, 8.5))
    
    # Set a compact layout between the main plot and the colorbar
    ax = plt.gca()  # Get the current axis
    scatter = ax.scatter(X, Y, c=u, cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(scatter, orientation='vertical', pad=0.02, location='right')  # Reduce the padding
    cbar.formatter = ScalarFormatter()  # Set the default formatter
    cbar.formatter.set_scientific(True)  # Enable scientific notation
    cbar.formatter.set_powerlimits((-2, 2))  # Show scientific notation for numbers smaller than 0.01
    cbar.ax.get_yaxis().offsetText.set_fontsize(32)  # Set scientific notation size  # fontsize of formatter 12
    #cbar.ax.get_yaxis().offsetText.set_x(1.1)  # Move scientific notation to the right 
    cbar.ax.tick_params(labelsize=32)  # Adjust the font size of the colorbar labels
    # Set font properties
    ax.set_xlabel('X', fontsize=32, fontweight='bold')
    ax.set_ylabel('Y', fontsize=32, fontweight='bold')
    ax.set_title(title, fontsize=40, fontweight='bold')
    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=32, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=32, length=5, width=1)  # Minor ticks

    # Adjust the subplot layout
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # The rect parameter controls the overall layout (reduce right margin)

    # Save and display
    plt.savefig(foldername + ".jpg", dpi=700, bbox_inches='tight')  # bbox_inches='tight' reduces the extra margins
    plt.show()
    plt.close()


def plot_loss(loss_bcs_log, loss_res_log):

    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_bcs_log, lw=2, label='bcs')
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_res_log, lw=2, label='res')

    plt.xlabel('Iteration', fontsize=32)
    plt.ylabel('Loss', fontsize=32)
    plt.yscale('log')
    plt.legend(
        loc='upper right', bbox_to_anchor=(0.95, 1), frameon=False, fontsize=36
    )
    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=32, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=32, length=5, width=1)  # Minor ticks

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0)) 
    ax.xaxis.set_major_formatter(formatter)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(32)

    plt.tight_layout()
    plt.savefig('loss' + ".jpg", dpi=700)
    plt.show()
    plt.close()
    
def plot_bc(bc, dis, filename):
    plt.figure(figsize = (6,5))
    plt.plot(bc,dis, lw=2, label=filename)
    plt.xlabel('x')
    plt.ylabel('displament')
    #plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename + ".jpg", dpi=700)
    plt.show()
    plt.close()    
    


def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)

# region Data generator
class DataGenerator():
    def __init__(self, u, v, h, s_u, s_v, batch_size, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u # input sample
        self.v = v
        self.h = h # location
        self.s_u = s_u # labeled data evulated at y (solution measurements, BC/IC conditions, etc.)
        self.s_v = s_v
        self.N = u.shape[0]
        self.batch_size = batch_size  
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        #print('key', subkey)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        rand_num = random.randint(key, (1,), 0, self.N + 1)[0]
        s_u = self.s_u[idx,:]
        s_v = self.s_v[idx,:]
        h = self.h[rand_num,:,:]
        u = self.u[idx,:]
        v = self.v[idx,:]
        # Construct batch
        inputs = (u, v, h)
        outputs = (s_u, s_v)
        return inputs, outputs

#region RBF 
### symetric RBF
def RBF(x1, x2, params): #radial basis function 
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales , 1) - \
            np.expand_dims(x2 / lengthscales , 0)
    r2 = np.abs(np.sum(diffs, axis=2))
    ub = np.ones(r2.shape)*np.max(x1 / lengthscales)
    d = ub -r2
    C  = np.minimum(r2, d)

    return output_scale**2 * np.exp(-0.5 * C**2)* (0.5 * C)

# To generate (x,t) (u, y)
def solve_ADR(key, Nx, Nt, P, length_scale):
    """No need explicit resolution 
    """
    # Generate subkeys
    xmin, xmax = 0, 10
    key, subkey0, subkey1= random.split(key, 3)
    subkeys0 = random.split(subkey0, 6)
    subkeys1 = random.split(subkey1, 6)
    subkeys2 = random.split(key, 6)
    subkeys3 = random.split(key, 6)
    # Generate a GP sample
    N = 512
    length_scale_u = length_scale[0]
    length_scale_v = length_scale[1]
    gp_params_u1 = (0.1, length_scale_u)
    gp_params_u2 = (0.1, length_scale_u)
    gp_params_v1 = (0.1, length_scale_v)
    gp_params_v2 = (0.1, length_scale_v)


    jitter = 1e-10
    X = np.linspace(xmin, xmax, N)[:,None]

    K_u1 = RBF(X, X, gp_params_u1)
    ### symetric (why np.linalg.cholesky() cannot work)
    import numpy as npr
    D, V = npr.linalg.eigh(K_u1 + jitter*np.eye(N))
    D = np.maximum(D, 0)
    L_u1 = V @ np.diag(np.sqrt(D))

    K_u2 = RBF(X, X, gp_params_u2)
    D_u2, V_u2 = npr.linalg.eigh(K_u2 + jitter*np.eye(N))
    D_u2 = np.maximum(D_u2, 0)
    L_u2 = V_u2 @ np.diag(np.sqrt(D_u2))


    K_v1 = RBF(X, X, gp_params_v1)
    D_v1, V_v1 = npr.linalg.eigh(K_v1 + jitter*np.eye(N))
    D_v1 = np.maximum(D_v1, 0)
    L_v1 = V_v1 @ np.diag(np.sqrt(D_v1))

    K_v2 = RBF(X, X, gp_params_v2)
    D_v2, V_v2 = npr.linalg.eigh(K_v2 + jitter*np.eye(N))
    D_v2 = np.maximum(D_v2, 0)
    L_v2 = V_v2 @ np.diag(np.sqrt(D_v2))
    
    def gp_sample(key, L):
        gp_sample = np.dot(L, random.normal(key, (N,)))
        return gp_sample
    
    gp_sample_u1 = vmap(gp_sample, (0,None))(subkeys0, L_u1)
    gp_sample_u2 = vmap(gp_sample, (0,None))(subkeys1, L_u2)

    gp_sample_v1 = vmap(gp_sample, (0,None))(subkeys1, L_v1)
    gp_sample_v2 = vmap(gp_sample, (0,None))(subkeys1, L_v2)
    # Create a callable interpolation function
    f_fn_u1 = lambda x: vmap(np.interp,(None, None, 0))(x, X.flatten(), gp_sample_u1)
    f_fn_u2 = lambda x: vmap(np.interp,(None, None, 0))(x, X.flatten(), gp_sample_u2)

    f_fn_v1 = lambda x: vmap(np.interp,(None, None, 0))(x, X.flatten(), gp_sample_v1)
    f_fn_v2 = lambda x: vmap(np.interp,(None, None, 0))(x, X.flatten(), gp_sample_v2)
    x_c = np.linspace(xmin, xmax, m)
    u_c1, u_c_p, u_c_n, _, _, _ = f_fn_u1(x_c)
    u_c2, _, _, _, _, _= f_fn_u2(x_c)
    v_c1, v_c_p, v_c_n, _, _, _ = f_fn_v1(x_c)
    v_c2, _, _, _, _, _ = f_fn_v2(x_c)

    # positive and negative
    '''u_c_p, v_c_p = np.maximum(u_c_p, 0), np.maximum(v_c_p, 0)
    u_c_n, v_c_n = np.minimum(u_c_n, 0), np.minimum(v_c_n, 0)'''

    # Input sensor locations and measurements

    return u_c1, v_c1 #, u_c_p, v_c_p, u_c_n, v_c_n 

# region training data 
def generate_one_training_data(key, P, Q, N):
    keys = random.split(key, N)
    # Numerical solution
    # adding 0 interval dosn't work in Static solution 
    '''u_c_, v_c_, u_c_p, v_c_p, u_c_n, v_c_n =  vmap(solve_ADR, (0, None, None, None, None))(keys, Nx , Ny, P, length_scale)
    u_c_tot = np.vstack([u_c_, u_c_p, u_c_n])
    v_c_tot = np.vstack([v_c_, v_c_p, v_c_n])
    idx0 = random.choice(keys[0], u_c_tot.shape[0], shape = (N,), replace=False) 
    idx1 = random.choice(keys[1], v_c_tot.shape[0], shape = (N,), replace=False)
    u_c = u_c_tot[idx0,:]
    v_c = v_c_tot[idx1,:]'''

    u_c, v_c = vmap(solve_ADR, (0, None, None, None, None))(keys, Nx , Ny, P, length_scale)
    '''idx_u = random.choice(keys[0], u_c1.shape[0] + u_c2.shape[0], shape = (N,), replace=False)
    idx_v = random.choice(keys[1], v_c1.shape[0] + v_c2.shape[0], shape = (N,), replace=False)
    u_c = np.vstack([u_c1, u_c2])[idx_u,:]
    v_c = np.vstack([v_c1, v_c2])[idx_v,:]'''

    num_points = m           # m is the sensor number  
    x_list = []
    y_list = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        x_list.append(x)
        y_list.append(y)  

    x = np.array(x_list).reshape(-1,1)
    y = np.array(y_list).reshape(-1,1)
    # Geneate subkeys
    subkeys = random.split(key, 4)

    # Sample points from the boundary and the inital conditions
    # Here we regard the initial condition as a special type of boundary conditions
    
    h_train_l = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
    h_train =np.tile(h_train_l, (N,1,1)).reshape(N, m, 2)
    
    # Training data for BC and IC
    
    u_train = np.hstack([u_c.reshape(-1,m), v_c.reshape(-1,m)])

    plot_bc(np.linspace(0,1,m),u_c.reshape(-1,m)[0,:],'u_l_0')
    plot_bc(np.linspace(0,1,m),u_c.reshape(-1,m)[5,:],'u_l_5')
    plot_bc(np.linspace(0,1,m),v_c.reshape(-1,m)[0,:],'v_l_0')
    plot_bc(np.linspace(0,1,m),v_c.reshape(-1,m)[5,:],'v_l_5')    

    print('u_l.shape=', u_c.shape, u_train.shape)
    v_train = np.hstack([u_c.reshape(-1,m), v_c.reshape(-1,m)])
    s_u_train = u_c.reshape(-1,1)
    s_v_train = v_c.reshape(-1,1)
    # Sample collocation points
    '''x_r_idx= random.choice(subkeys[2], np.arange(Nx), shape = (Q,1))
    x_r = x[x_r_idx]
    y_r = random.uniform(subkeys[3], minval = 0, maxval = 1, shape = (Q,1))'''
    def generate_collocation_points(key, Q):
        subkeys = random.split(key, 2)
        r = random.uniform(subkeys[0], (Q,1), minval=0, maxval=radius)
        theta = random.uniform(subkeys[1], (Q,1), minval=0, maxval=2*np.pi)   
        xc = center[0] + r * np.cos(theta)
        yc = center[1] + r * np.sin(theta)
        
        return np.hstack([xc, yc])
    subkeys_ = random.split(key, N)
    #print('u.shape=', u.shape)
    
    # Training data for the PDE residual
    u_r_train = np.hstack([u_c, v_c])
    v_r_train = np.hstack([u_c, v_c])
    h_r_train = vmap(generate_collocation_points, (0,None))(subkeys_, Q)
    print('h_r_train.shape=', h_r_train[0,:5,:5])
    s_r_train = np.zeros((N,Q,1))  # here we can see the N(s) = u(x), we have m = Nx 
   
    #print('u_r_train.shape=', u_r_train.shape)
    return u_train, v_train,h_train, s_u_train, s_v_train, u_r_train, v_r_train, h_r_train, s_r_train


# region test data
# Geneate test data corresponding to one input sample
def generate_one_test_data(key, P):
    Nx = P
    Ny = P
    u_c, v_c,  = solve_ADR(key, Nx , Ny, P, length_scale)
    # choose the 0 interval solution to predict 
    u_l = u_c
    v_l = v_c

    theta = np.linspace(0, 2 * np.pi, P)
    r = np.linspace(0, radius, P)
    T, R = np.meshgrid(theta, r)
    XX = center[0] + R * np.cos(T)
    YY = center[1] + R * np.sin(T)
    u_test = np.hstack([u_l, v_l])
    v_test = np.hstack([u_l, v_l])

    h_test = np.hstack([XX.flatten()[:,None], YY.flatten()[:,None]]) ### because XX TT shape = P**2 transform them to (P**2,1) for h_stack (P**2,2)

    return u_test, v_test, h_test

# Geneate training data corresponding to N input sample
def generate_training_data(key, N, P, Q):
    config.update("jax_enable_x64", True)
    u_train, v_train, h_train, s_u_train, s_v_train, u_r_train, v_r_train, h_r_train, s_r_train = generate_one_training_data(key, P, Q ,N)
    #print(u_r_train.shape)  ## vmap return N * u_r_train different  (N, P, m)
    print('u_train',u_train.shape)
    u_train = np.float32(u_train.reshape(N ,-1))  #turn to be (N,m)
    v_train = np.float32(v_train.reshape(N ,-1)) 
    h_train = np.float32(h_train.reshape(N , P,-1))
    s_u_train = np.float32(s_u_train.reshape(N,-1))
    s_v_train = np.float32(s_v_train.reshape(N,-1))
    u_r_train = np.float32(u_r_train.reshape(N ,-1))
    v_r_train = np.float32(v_r_train.reshape(N ,-1))
    h_r_train = np.float32(h_r_train.reshape(N , Q ,-1))
    print('h_r_train.shape1=', h_r_train[0,:5,:5])
    s_r_train = np.float32(s_r_train.reshape(N ,-1))

    config.update("jax_enable_x64", False)
    return     u_train, v_train,h_train, s_u_train,  s_v_train, u_r_train, v_r_train, h_r_train, s_r_train
    
# Geneate test data corresponding to N input sample
def generate_test_data(key, N, P):

    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    keys = keys[0]
    u_test, v_test, y_test = generate_one_test_data(keys, P)
    u_test = np.float32(u_test.reshape(N ,-1))
    v_test = np.float32(v_test.reshape(N ,-1))
    y_test = np.float32(y_test.reshape(N * P**2 ,-1))
    config.update("jax_enable_x64", False)
    return u_test, v_test, y_test
    #return np.zeros(u_test.shape), y_test



# GRF length scale
length_scale = [1, 1] #0.2 for symetric RBF big length_scale

# Resolution of the solution
Nx = 200
Ny = 200
d = 2
N = 1000 # number of input samples
m = Nx   # number of input sensors
P_train = m # number of output sensors, 100 for each side
Q_train = 1600 #400  # number of collocation points for each input sample
center = (0.0, 0.5, 0.0)  # Center of the circle
radius = 0.3             # Radius of the circle
# region main 
if __name__ == "__main__":
    originalDir ='/nfsv4/21040463r/FEM_DeepONet_final_results/FEM_DeepONet_hyper/'
    os.chdir(os.path.join(originalDir))

    foldername = 'prepare_DeepONet_static_hyper_elastic_200w_adam_lr_RBF_E_01_2tractions_figure_manu_lv_lu_1'  
    createFolder(foldername)
    os.chdir(os.path.join(originalDir, './'+ foldername + '/'))
    
    ### define the elastic model 
    ela_model = dict()
    ela_model['E'] = 0.1 #1000 
    ela_model['nu'] = 0.3 
    os.chdir(os.path.join(originalDir, './'+ foldername + '/'))
    origin_real  = os.path.join(originalDir, './'+ foldername + '/')
    
    key = random.PRNGKey(0)
    u_bcs_train, v_bcs_train,h_bcs_train, s_u_train, s_v_train, u_res_train, v_res_train, h_res_train, s_res_train\
            = generate_training_data(key, N, P_train, Q_train)
   
    # Initialize model
    branch_layers = [2*m, 100, 100, 100, 100, 800]
    trunk_layers =  [d, 100, 100, 100, 100, 800]
    model = PI_DeepONet(branch_layers, trunk_layers, **ela_model)
    
    # Create data set
    batch_size =  100 #10000
    bcs_dataset = DataGenerator(u_bcs_train, v_bcs_train, h_bcs_train, s_u_train, s_v_train, batch_size)
    
    res_dataset = DataGenerator(u_res_train, v_res_train,h_res_train, s_res_train, s_res_train, batch_size)
    
    # Train
    model.train(bcs_dataset, res_dataset, nIter=2000000)

    
    # region prediction 
    # Predict

    '''with open('DeepONet_DR.pkl', 'rb') as f:
        params = pickle.load(f)'''

    params = model.get_params(model.opt_state)
    with open('DeepONet_DR.pkl', 'wb') as f:
        pickle.dump(params, f)


    #Plot for loss function
    plot_loss(model.loss_bcs_log, model.loss_res_log)
    import numpy as npr # jnp donesn't have loadtxt
    npr.savetxt('loss_bcs_hyper.txt', model.loss_bcs_log)
    npr.savetxt('loss_res_hyper.txt', model.loss_res_log)
    


    os.chdir(originalDir+ '/'+ 'FE_FE_2disks_hyper_epsilon_1e-3_2tractions' + '/')

    import numpy as npr # jnp donesn't have loadtxt
    x_c = np.array(npr.loadtxt('xc1_out.txt')).reshape(1,-1)
    y_c = np.array(npr.loadtxt('yc1_out.txt')).reshape(1,-1)
    #u2c = np.array(npr.loadtxt('u2c_out ts=4 iter=99')).reshape(1,-1)
    #v2c= np.array(npr.loadtxt('v2c_out ts=4 iter=99')).reshape(1,-1)
    #u2 = np.array(npr.loadtxt('u2 ts=4 .txt')).reshape(1,-1)
    #v2 = np.array(npr.loadtxt('v2 ts=4 .txt')).reshape(1,-1)
    X1 = np.array(npr.loadtxt('X1.txt')).reshape(1,-1)
    Y1 = np.array(npr.loadtxt('Y1.txt')).reshape(1,-1)

    os.chdir(originalDir+ '/'+ 'FE_full_square_hyper_2tractions' + '/')
    u = np.array(npr.loadtxt('u ts=4 .txt')).reshape(1,-1)
    v = np.array(npr.loadtxt('v ts=4 .txt')).reshape(1,-1)
    X = np.array(npr.loadtxt('X.txt')).reshape(1,-1)
    Y = np.array(npr.loadtxt('Y.txt')).reshape(1,-1)

    os.chdir(origin_real)

    u2_fun = Rbf(X, Y, u)
    v2_fun = Rbf(X, Y, v)

    u2 = u2_fun(X1, Y1).reshape(1,-1)
    v2 = v2_fun(X1, Y1).reshape(1,-1)
    u2c = u2_fun(x_c, y_c).reshape(1,-1)
    v2c = v2_fun(x_c, y_c).reshape(1,-1)

    npr.savetxt('u2.txt', u2)
    npr.savetxt('v2.txt', v2)
    npr.savetxt('u2c.txt', u2c)
    npr.savetxt('v2c.txt', v2c)

    u2 = npr.loadtxt('u2.txt').reshape(1,-1)
    v2 = npr.loadtxt('v2.txt').reshape(1,-1)
    u2c = npr.loadtxt('u2c.txt').reshape(1,-1)
    v2c = npr.loadtxt('v2c.txt').reshape(1,-1)


    u_test = np.hstack([u2c, v2c]).reshape(1,-1) 
    v_test = np.hstack([u2c, v2c]).reshape(1,-1)

    num_points = m           # m is the sensor number  
    x_list = []
    y_list = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        x_list.append(x)
        y_list.append(y)  

    x = np.array(x_list).reshape(1,-1)
    y = np.array(y_list).reshape(1,-1)
    index = []
    # for resort data to use in the sensor points 
    for i in range(x.shape[1]):
        index_1 = np.where(np.isclose(x_c,x[0,i])&np.isclose(y_c,y[0,i]))[1]
        index.append(index_1)
    
    #print('index',x_c[0,index].reshape(1,-1)[0, 40:50], x[0,40:50])
    #plot_s(x, y, v_l_bc,'test V1c_deep')
    print('u2c.shape=', u2c.shape, v2c.shape)
    u2c = u2c[0,index].reshape(1,-1)
    v2c = v2c[0,index].reshape(1,-1)
    print('utest=',u_test.shape, v_test.shape)
    
    #plot_s(x, y, v_l_bc,'test V1c_deep changed')
    u_test = np.hstack([u2c, v2c]).reshape(1,-1) 
    v_test = np.hstack([u2c, v2c]).reshape(1,-1)
    #print(u_l)
    y_test = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
    s_u_pred, s_v_pred= model.predict_s(params, u_test, v_test, y_test)
    #plot_s(x, y, s_v_pred, 'test V1c pre')
    y_test_c = np.hstack([x_c.reshape(-1,1), y_c.reshape(-1,1)])
    sigmax_pred0, _, _ = model.stress(params, u_test, v_test, y_test_c)
    #plot_s(x_c, y_c, sigmax_pred0, 'test sigmax pre')
    #plot_s(x_c, y_c, sigma0_x_real, 'test sigmax real')
    #plot_s(x_c, y_c, sigmax_pred0 - sigma0_x_real, 'test sigmax error')



    plot_bc(np.linspace(0,1,int(s_u_pred[0,:].shape[0])),s_u_pred[0,:int(u_test[0,:].shape[0])],'test u_l_bc')
    plot_bc(np.linspace(0,1,int(u2c.shape[1])),u2c.flatten(),'test u_l_bc1')
    plot_bc(np.linspace(0,1,int(u2c.shape[1])),s_u_pred[0,:int(u_test[0,:].shape[0])] - u2c.flatten(),'test u_l_error')
    
    plot_bc(np.linspace(0,1,int(s_v_pred[0,:].shape[0])),s_v_pred[0,:int(v_test[0,:].shape[0])],'test v_l_bc')
    plot_bc(np.linspace(0,1,int(v2c.shape[1])), v2c.flatten(),'test v_l_bc1')
    plot_bc(np.linspace(0,1,int(v2c.shape[1])),s_v_pred[0,:int(v_test[0,:].shape[0])] - v2c.flatten(),'test v_l_error')


    y_test = np.hstack([X1.reshape(-1,1), Y1.reshape(-1,1)])
    s_u_pred, s_v_pred= model.predict_s(params, u_test, v_test, y_test)
    plot_disp(X1,Y1, s_u_pred, 's_u_test1_0',rf'$u_{{\mathrm{{NO}},\Omega_{{II}}}}^{{4}}$')
    plot_disp(X1,Y1, u2, 's_u_real_0', rf'$u_{{\mathrm{{FE}},\Omega_{{II}}}}^{{4}}$')
    plot_disp(X1,Y1, s_v_pred,'s_v_test1_0',rf'$v_{{\mathrm{{NO}},\Omega_{{II}}}}^{{4}}$')
    plot_disp(X1,Y1, v2, 's_v_real_0',rf'$v_{{\mathrm{{FE}},\Omega_{{II}}}}^{{4}}$')
    plot_relative_error(X1,Y1, np.abs(s_u_pred - u2),'s_u_error_01', rf'$|u_{{\mathrm{{FE}}}}^{{4}} - u_{{\mathrm{{FE-FE}},\Omega_{{II}}}}^{{4}}|$')
    plot_relative_error(X1,Y1, np.abs(s_v_pred - v2), 's_v_error_01', rf'$|v_{{\mathrm{{FE}}}}^{{4}} - u_{{\mathrm{{FE-FE}},\Omega_{{II}}}}^{{4}}|$')
    













