"""
=======================================================================
Part of the FEM-DeepONet coupling work
-----------------------------------------------------------------------
DeepONet Model for elasto-dynamic for time steps 109 to 119
2D plane strain problem
Square + square domain (overlapping boundary)
"""

import os
import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian, lax
from jax.example_libraries import optimizers
from jax.nn import relu
from jax import config
from jax.flatten_util import ravel_pytree
import itertools
from functools import partial
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import math  
from scipy.interpolate import griddata, Rbf
import pickle
import scipy
import jax.nn as jnn
from jax.lax import conv_general_dilated as conv_lax
from interpax import Interpolator2D
import flax.linen as fnn
from dynamic_utils import createFolder, plot_disp, plot_relative_error, plot_loss
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



#region CNN 
# Initialize the Glorot (Xavier) normal distribution for weight initialization
initializer = jax.nn.initializers.glorot_normal()
rng_key = random.PRNGKey(0)
key1, key2, key3, key4 = random.split(rng_key, 4)

def conv(x, w, b, pool_window=(2, 2), pool_strides=(2, 2)):
    """Convolution operation with VALID padding"""
    # Reshape inputs to match JAX's conv_general_dilated expectations
    # x shape: (H, W, C_in)
    # w shape: (H, W, C_in, C_out)
    conv_out = conv_lax(
        lhs=x[None, ...],  # Add batch dimension: (1, H, W, C_in)
        rhs=w,             # Kernel: (H, W, C_in, C_out)
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    conv_out = conv_out[0] + b
    pooled_out = fnn.avg_pool(conv_out, window_shape=pool_window, strides=pool_strides, padding='VALID')
    return pooled_out  # Remove batch dimension and add bias

def init_cnn_params(p, key=random.PRNGKey(0)):
    """
    Initialize CNN parameters for the branch network as a list of tuples.
    Each tuple contains (weights, biases) for a layer.

    Returns:
    list: List of tuples, each containing weights and biases for a layer
    """
    key1, key2, key3, key4, key5 = random.split(key, 5)
    out_put_channels = 4
    # Conv1: (3,3,4,6) - input has 4 channels, 6 output filters ### error maybe 
    # 4 channels for u v vx vy in the previous step 
    conv1_w = random.normal(key1, (3, 3, 4, out_put_channels)) * 0.1
    conv1_b = random.normal(key2, (out_put_channels,)) * 0.1                               

    conv2_w = random.normal(key2, (3, 3, out_put_channels, out_put_channels)) * 0.1
    conv2_b = random.normal(key3, (out_put_channels,)) * 0.1

    conv3_w = random.normal(key3, (3, 3, out_put_channels, out_put_channels)) * 0.1
    conv3_b = random.normal(key4, (out_put_channels,)) * 0.1

    # Calculate the output size after the convolutions and pooling
    conv_output_size = nx1
    for _ in range(3):  # 3 convolutional layers
        conv_output_size = (conv_output_size - 3 + 1) // 2  # kernel size 3, stride 2

    flat_size = conv_output_size * conv_output_size * out_put_channels  # 6*4 filters in the last conv layer
    # here flat size is nx1*nx1*4 = 82*82*4 


    dense1_w = random.normal(key4, (flat_size, 256)) * np.sqrt(2.0 / flat_size)
    dense1_b = np.zeros(256)

    dense2_w = random.normal(key5, (256, p)) * np.sqrt(2.0 / 256)
    dense2_b = np.zeros(p)

    return [
        (conv1_w, conv1_b),
        (conv2_w, conv2_b),
        (conv3_w, conv3_b),
        (dense1_w, dense1_b),
        (dense2_w, dense2_b)
    ]

def BranchNet_dil(params, x):
    """
    CNN-based branch network for the DeepONet.

    Args:
    params (list): List of tuples containing weights and biases
    x (array): Input tensor of shape (batch_size, 40, 41)

    Returns:
    array: Output tensor of shape (batch_size, 2*p)
    """
    def single_forward(params, x):
        # Unpack conv and dense layer parameters
        (conv1_w, conv1_b), (conv2_w, conv2_b), (conv3_w, conv3_b), \
        (dense1_w, dense1_b), (dense2_w, dense2_b) = params

        # Reshape input to (m_i, m_i, 4) - adding channel dimension
        x = np.transpose(x.reshape(4, nx1, nx1), (1,2,0))

        # Convolution layers with SiLU activation
        x = np.tanh(conv(x, conv1_w, conv1_b))
        x = np.tanh(conv(x, conv2_w, conv2_b))
        x = np.tanh(conv(x, conv3_w, conv3_b))

        # Flatten
        x = x.reshape(-1) # add the boundary sensor points
        # Dense layers
        x = jnn.silu(np.dot(x, dense1_w) + dense1_b)
        outputs = np.dot(x, dense2_w) + dense2_b

        return outputs

    return vmap(partial(single_forward, params))(x)



# region model 
# Define the model
class PI_DeepONet:
    def __init__(self, branch_layers_1, trunk_layers, **ela_model):
        # Network initialization and evaluation functions
        self.branch_init_1, self.branch_apply_1 = MLP(branch_layers_1, activation=np.tanh)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=np.tanh)
        self.branch_apply = BranchNet_dil
        #elastic_model 
        self.E = ela_model['E']
        self.nu = ela_model['nu']
        self.rho = ela_model['rho']
        
        # Initialize
        branch_params = init_cnn_params(trunk_layers[-1],  key = random.PRNGKey(1234))
        branch_params_v = init_cnn_params(trunk_layers[-1], key = random.PRNGKey(12341))

        branch_params_1 = self.branch_init_1(rng_key = random.PRNGKey(123411))
        branch_params_v_1 = self.branch_init_1(rng_key = random.PRNGKey(1234111))

        trunk_params_u = self.trunk_init(rng_key = random.PRNGKey(4321))
        trunk_params_v = self.trunk_init(rng_key = random.PRNGKey(43211))
        
        params_u = (branch_params, branch_params_1, trunk_params_u)
        params_v = (branch_params_v, branch_params_v_1, trunk_params_v)


        params = (params_u, params_v)
        
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3,
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
    def operator_net_u(self, params, v, x, y):
        branch_params_v, branch_params_v_1, trunk_params_v = params
        h = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
        B = self.branch_apply(branch_params_v, v[:,:-2*4*m])  
        B_1 = self.branch_apply_1(branch_params_v_1, v[:,-2*4*m:])
        T = self.trunk_apply(trunk_params_v, h)
        # Compute the final output
        # Input shapes:
        # branch: [batch_size, 4*ms] --> [batch_size, 800]
        # branch_1: [batch_size, 2*m]--> [batch_size, 800]
        # trunk: [p, 2]
        # output: [batch_size, p]
        B_tot = B * B_1 # element-wise multiplication keep the same shape
        outputs = np.einsum('ij,kj->ik', B_tot, T)
        return  outputs
    
    @partial(jax.jit, static_argnums=(0,))
    def operator_net_v(self, params, v, x, y):
        branch_params_v, branch_params_v_1, trunk_params_v = params
        h = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
        B = self.branch_apply(branch_params_v, v[:,:-2*4*m])  
        B_1 = self.branch_apply_1(branch_params_v_1, v[:,-2*4*m:])
        T = self.trunk_apply(trunk_params_v, h)
        # Compute the final output
        # Input shapes:
        # branch: [batch_size, 4*ms] --> [batch_size, 800]
        # branch_1: [batch_size, 2*m]--> [batch_size, 800]
        # trunk: [p, 2]
        # output: [batch_size, p]
        B_tot = B * B_1 # element-wise multiplication keep the same shape
        outputs = np.einsum('ij,kj->ik', B_tot, T)
        return  outputs

       
    # region residual net 
    # Define ODE/PDE residual
    def residual_net(self, params, u, v, x, y):
        params_u, params_v = params

        s_u_yy= jax.jvp(lambda y : jax.jvp(lambda y: self.operator_net_u(params_u, u, x, y), (y,), (np.ones_like(y),))[1]
                        , (y,), (np.ones_like(y),))[1]
        s_u_xx= jax.jvp(lambda x : jax.jvp(lambda x: self.operator_net_u(params_u, u, x, y), (x,), (np.ones_like(x),))[1]    
                        , (x,), (np.ones_like(x),))[1]
        s_u_xy= jax.jvp(lambda x : jax.jvp(lambda y: self.operator_net_u(params_u, u, x, y), (y,), (np.ones_like(y),))[1]
                        , (x,), (np.ones_like(x),))[1]
        s_v_yy= jax.jvp(lambda y : jax.jvp(lambda y: self.operator_net_v(params_v, v, x, y), (y,), (np.ones_like(y),))[1]
                        , (y,), (np.ones_like(y),))[1]
        s_v_xx= jax.jvp(lambda x : jax.jvp(lambda x: self.operator_net_v(params_v, v, x, y), (x,), (np.ones_like(x),))[1]
                        , (x,), (np.ones_like(x),))[1] 
        s_v_xy= jax.jvp(lambda x : jax.jvp(lambda y: self.operator_net_v(params_v, v, x, y), (y,), (np.ones_like(y),))[1]
                        , (x,), (np.ones_like(x),))[1]
        
        
        u_old, v_old, vx_old, vy_old = u[:, :m_s], u[:, m_s:2*m_s], u[:, 2*m_s:3*m_s]*100, u[:, 3*m_s:4*m_s]*100

        s_ax = -2/(dt*beta)**2*(u_old + vx_old*dt - self.operator_net_u(params_u, u, x, y)) #+ (1-beta)/(beta) * ax_old
        s_ay = -2/(dt*beta)**2*(v_old + vy_old*dt - self.operator_net_v(params_v, v, x, y)) #+ (1-beta)/(beta) * ay_old

        para1 = self.E/((1 + self.nu)*(1-2*self.nu))
        ###Newton's second law in plane strain 
        res0 = para1*((1-self.nu) * s_u_xx + self.nu * s_v_xy) + para1*(1-2*self.nu)/2*(s_u_yy + s_v_xy) - self.rho * s_ax
        res1 = para1*((1-self.nu) * s_v_yy + self.nu * s_u_xy) + para1*(1-2*self.nu)/2*(s_v_xx + s_u_xy) - self.rho * s_ay

        return res0, res1
    
    
    # region stress
    def stress(self, params, u, v, Y_star):
        params_u, params_v = params
        x, y = Y_star[:,0], Y_star[:,1]
        
        s_u_y = jax.jvp(lambda y: self.operator_net_u(params_u, u, x, y), (y,), (np.ones_like(y),))[1]
        
        s_u_x= jax.jvp(lambda x: self.operator_net_u(params_u, u, x, y), (x,), (np.ones_like(x),))[1]

        s_v_y= jax.jvp(lambda y: self.operator_net_v(params_v, v, x, y), (y,), (np.ones_like(y),))[1]
                       
        s_v_x=  jax.jvp(lambda x: self.operator_net_v(params_v, v, x, y), (x,), (np.ones_like(x),))[1]

        para1 = self.E/((1 + self.nu)*(1-2*self.nu))*1e8
        sigma_x = para1*((1-self.nu) * s_u_x + self.nu * s_v_y) 
        sigma_y = para1*((1-self.nu) * s_v_y + self.nu * s_u_x) 
        sigma_xy = para1*(1-2*self.nu) * (s_u_y + s_v_x)/2  

        return sigma_x, sigma_y, sigma_xy
    
    
    # region loss 
    # Define boundary loss
    def loss_bcs(self, params, batch):
        # Fetch data
        # inputs: (u, v, h), shape = (batch_size, 4*ms + 2*4*m), (batch_size, 4*ms + 2*4*m), (nx1, 2)
        # outputs: (s_u, s_v) shape = (batch_size, 4*nx1), (batch_size, 4*nx1)
        inputs, outputs = batch
        u, v, h = inputs
        params_u, params_v, = params
        # Compute forward pass
        s_u_pred = self.operator_net_u(params_u, u, h[:, 0], h[:, 1])
        s_v_pred = self.operator_net_v(params_v, v, h[:, 0], h[:, 1])
        ## outputs always be 0 
        loss =  np.mean((outputs[0]- s_u_pred)**2) + np.mean((outputs[1] - s_v_pred)**2)
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
        # Compute loss
        loss = np.mean((outputs[0] - res0)**2) + np.mean((outputs[1] - res1)**2)
       
        return loss

    # Define total loss
    def loss(self, params, bcs_batch, res_batch):
        loss_bcs = self.loss_bcs(params, bcs_batch)
        loss_res = self.loss_res(params, res_batch)
        loss = loss_bcs + loss_res
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


# region Data generator
class DataGenerator():
    def __init__(self, u, v, h, s_u, s_v,batch_size, rng_key=random.PRNGKey(1234)):
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

    return output_scale**2 * np.exp(-0.5 * C**2)

    

# To generate (x,t) (u, y)
def PI_data_generation(key, Nx, Nt, P, length_scale):
    '''Generate data for the PI DeepONet model'''
    # Generate subkeys
    xmin, xmax = 0, 10
    key, subkey0, subkey1= random.split(key, 3)
    subkeys0 = random.split(subkey0, 6)
    subkeys1 = random.split(subkey1, 6)
    # Generate a GP sample
    N = 512
    length_scale_u = length_scale[0]
    length_scale_a = length_scale[1]
    gp_params = (0.01, length_scale_u)
    gp_params_a = (0.03, length_scale_a)

    jitter = 1e-10
    X = np.linspace(xmin, xmax, N)[:,None]
    K = RBF(X, X, gp_params)
    ### symetric (why np.linalg.cholesky() cannot work)
    import numpy as npr
    D, V = npr.linalg.eigh(K + jitter*np.eye(N))
    D = np.maximum(D, 0)
    L = V @ np.diag(np.sqrt(D))

    K_a = RBF(X, X, gp_params_a)
    ### symetric (why np.linalg.cholesky() cannot work)
    D_a, V_a = npr.linalg.eigh(K_a + jitter*np.eye(N))
    D_a = np.maximum(D_a, 0)
    L_a = V_a @ np.diag(np.sqrt(D_a))
    
    def gp_sample(key, L):
        gp_sample = np.dot(L, random.normal(key, (N,)))
        return gp_sample
    
    gp_sample_u = vmap(gp_sample, (0,None))(subkeys0, L)
    gp_sample_a = vmap(gp_sample, (0,None))(subkeys1, L_a)
    # Create a callable interpolation function
    f_fn_uv = lambda x: vmap(np.interp,(None, None, 0))(x, X.flatten(), gp_sample_u)
    f_fn_a = lambda x: vmap(np.interp,(None, None, 0))(x, X.flatten(), gp_sample_a)
    x_c = np.linspace(xmin, xmax, m)
    u_c, v_c, u_c_p, v_c_p, u_c_n, v_c_n = f_fn_uv(x_c)
    # positive and negative
    # exchange the positive and negative profile (since ax_c_n ay_c_n have larger zero internal)
    ax_c, ay_c, ax_c_p, ay_c_p, ax_c_n, ay_c_n = f_fn_a(x_c)
    
    u_c_p, v_c_p = np.maximum(ax_c_p, 0)/3, np.maximum(ay_c_p, 0)/3
    u_c_n, v_c_n = np.minimum(ax_c_n, 0)/3, np.minimum(ay_c_n, 0)/3
    ax_c_p, ay_c_p = np.maximum(u_c_p, 0)*3, np.maximum(v_c_p, 0)*3
    ax_c_n, ay_c_n = np.minimum(u_c_n, 0)*3, np.minimum(v_c_n, 0)*3


    # Create grid
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
    # Input sensor locations and measurements

    return u_c, v_c, u_c_p, v_c_p, u_c_n, v_c_n


# region training data 
def generate_one_training_data(key, P, Q, N):
    keys = random.split(key, int(N/2))
    # load correct dataset 
    os.chdir(os.path.join(originalDir, './' + 'dataload_from_full_square_dataset_109_119' + '/'))
    U1_d = npr.loadtxt('U1_d.txt')
    V1_d = npr.loadtxt('V1_d.txt')
    vx_d = npr.loadtxt('vx_d.txt')
    vy_d = npr.loadtxt('vy_d.txt') 
    u_l =  npr.loadtxt('U_l.txt')
    v_l =  npr.loadtxt('V_l.txt')
    u_r = npr.loadtxt('U_r.txt')
    v_r = npr.loadtxt('V_r.txt')
    u_up = npr.loadtxt('U_up.txt')
    v_up = npr.loadtxt('V_up.txt')
    u_down = npr.loadtxt('U_down.txt')
    v_down = npr.loadtxt('V_down.txt')
    row_delete = list(range(10, U1_d.shape[0], 11))

    U1_d = npr.delete(U1_d, row_delete, axis=0)
    V1_d = npr.delete(V1_d, row_delete, axis=0)
    vx_d = npr.delete(vx_d, row_delete, axis=0)
    vy_d = npr.delete(vy_d, row_delete, axis=0)

    row_delete_bc = list(range(0,u_l.shape[0], 11))
    u_l = npr.delete(u_l, row_delete_bc, axis=0)
    v_l = npr.delete(v_l, row_delete_bc, axis=0)
    u_r = npr.delete(u_r, row_delete_bc, axis=0)
    v_r = npr.delete(v_r, row_delete_bc, axis=0)
    u_up = npr.delete(u_up, row_delete_bc, axis=0)
    v_up = npr.delete(v_up, row_delete_bc, axis=0)
    u_down = npr.delete(u_down, row_delete_bc, axis=0)
    v_down = npr.delete(v_down, row_delete_bc, axis=0)

    os.chdir(originalDir_real)
    u_tot_l, v_tot_l, vx_tot_l, vy_tot_l = U1_d, V1_d, vx_d, vy_d

    # Sample points from the boundary and the inital conditions
    h_train_up = np.hstack([X1[index_up].reshape(-1,1), Y1[index_up].reshape(-1,1)])
    h_train_down = np.hstack([X1[index_down].reshape(-1,1), Y1[index_down].reshape(-1,1)])
    h_train_l = np.hstack([X1[index_left].reshape(-1,1), Y1[index_left].reshape(-1,1)])
    h_train_r = np.hstack([X1[index_right].reshape(-1,1), Y1[index_right].reshape(-1,1)])

    h_train_tot = np.vstack([h_train_l, h_train_r, h_train_up, h_train_down])
    h_train =np.tile(h_train_tot, (N,1,1)).reshape(N, 4*m, 2)    

    # Training data for BC and IC
    u_bc = np.hstack([u_l, u_r, u_up, u_down])
    v_bc = np.hstack([v_l, v_r, v_up, v_down])
       
    u_train = np.hstack([u_tot_l, v_tot_l, vx_tot_l, vy_tot_l, 
                         u_bc, v_bc])

    v_train = np.hstack([u_tot_l, v_tot_l, vx_tot_l, vy_tot_l,
                          u_bc, v_bc])

    s_u_train = u_bc
    s_v_train = v_bc
    
    # Training data for the PDE residual
    u_r_train = np.hstack([u_tot_l, v_tot_l, vx_tot_l, vy_tot_l, u_bc, v_bc])
    v_r_train = np.hstack([u_tot_l, v_tot_l, vx_tot_l, vy_tot_l, u_bc, v_bc])
    h_r_train = np.tile(np.hstack([X1_.flatten().reshape(-1,1), Y1_.flatten().reshape(-1,1)]), (N,1,1))
    s_r_train = np.zeros((N, m_s, 1))  # here we can see the N(s) = u(x), we have m = Nx 

    return u_train, v_train,h_train, s_u_train, \
            s_v_train, u_r_train, v_r_train, h_r_train, s_r_train, h_train_l


# Geneate training data corresponding to N input sample
def generate_training_data(key, N, P, Q):
    config.update("jax_enable_x64", True)
    u_train, v_train, h_train, s_u_train, s_v_train, u_r_train, v_r_train,\
                         h_r_train, s_r_train, h_train_l = generate_one_training_data(key, P, Q ,N)

    u_train = np.float32(u_train.reshape(N ,-1))  #turn to be (N, 4*m1 + 2*4*m)
    v_train = np.float32(v_train.reshape(N ,-1)) 
    h_train = np.float32(h_train.reshape(N ,-1, 2))
    s_u_train = np.float32(s_u_train.reshape(N,-1))
    s_v_train = np.float32(s_v_train.reshape(N,-1))
    u_r_train = np.float32(u_r_train.reshape(N ,-1))
    v_r_train = np.float32(v_r_train.reshape(N ,-1))
    h_r_train = np.float32(h_r_train.reshape(N , -1 ,2))
    
    s_r_train = np.float32(s_r_train.reshape(N ,-1))

    config.update("jax_enable_x64", False)
    return     u_train, v_train,h_train, s_u_train, \
    s_v_train, u_r_train, v_r_train, h_r_train, s_r_train
    
# Used in CNN parts 
nx1 = 82
m = nx1

if __name__ == "__main__":
    # region save path 
    originalDir = os.getcwd()
    os.chdir(os.path.join(originalDir))
    foldername = 'prepare_DeepONet_Elasto_dynamic_square_square_109_119'
    createFolder(foldername)
    os.chdir(os.path.join(originalDir, './' + foldername + '/'))
    originalDir_real = os.path.join(originalDir, './' + foldername + '/')

    os.chdir(os.path.join(originalDir, './' + 'FE_full_elasto_dynamic_ground_truth' + '/'))
    # GRF length scale
    length_scale = [1, 0.4] #0.2 for symetric RBF big length_scale
    import numpy as npr 
    # Resolution of the solution
    center = (0.0, 0.5, 0.0)  # Center of the circle
    radius = 0.3             # Radius of the circle
    X1 = npr.loadtxt('X1.txt')
    Y1 = npr.loadtxt('Y1.txt')
    index_up = np.where(np.isclose(Y1, center[1] + (radius + 0.05)))[0]
    index_down = np.where(np.isclose(Y1, center[1] - (radius + 0.05)))[0]
    index_left = np.where(np.isclose(X1, center[0] - (radius + 0.05)))[0]
    index_right = np.where(np.isclose(X1, center[0] + (radius + 0.05)))[0]

    # resort the sequence of the index (easy for the following application in coupling)
    remix_down = np.argsort(X1[index_down])
    remix_up = np.argsort(X1[index_up])
    remix_left = np.argsort(Y1[index_left])
    remix_right = np.argsort(Y1[index_right])

    index_up = index_up[remix_up]
    index_down = index_down[remix_down]
    index_left = index_left[remix_left]
    index_right = index_right[remix_right]

    nx1 = index_up.shape[0]
    nx2 = index_up.shape[0]

    m_s = nx1 * nx2 # number of sensors in the square
    Nx = nx1
    Ny = nx2
    d = 2
    N = 1000 # number of input samples
    m = nx1 # number of input sensors
    P_train = m # number of output sensors, 100 for each side
    Q_train = 400 #400  # number of collocation points for each input sample


    #nemark method
    beta = 1
    # Time-stepping parameters
    T       = 4.0
    Nsteps  = 1e4
    dt =  T/Nsteps

    # region main 
    # transform the disk to square (disk insert in the square)
    X = np.linspace(center[0]-radius-0.05, center[0]+radius+0.05, nx1)
    Y = np.linspace(center[1]-radius-0.05, center[1]+radius+0.05, nx2)
    X1_, Y1_ = np.meshgrid(X, Y)

    os.chdir(originalDir_real)
    
    ### define the elastic model 
    ela_model = dict()
    ela_model['E'] = 1000e-8 #1000 
    ela_model['nu'] = 0.3 
    ela_model['rho'] = 5e-8 #5
    #os.chdir('/nfsv4/21040463r/PINN/DeepONet_DR_no_ADR_to_ul_ur_vl_vr_test_rerun_0731_uxy_elastic')
    
    key = random.PRNGKey(0)
    #u_bcs_train, v_bcs_train, h_bcs_train, s_u_train, s_v_train, u_res_train, \
    #    v_res_train, h_res_train, s_res_train= generate_training_data(key, N, P_train, Q_train)
   
    # Initialize model
    branch_layers_1 =  [2*4*m, 100, 100, 100, 100, 800]
    trunk_layers =  [d, 100, 100, 100, 100, 800]
    model = PI_DeepONet(branch_layers_1, trunk_layers, **ela_model)
    
    # Create data set
    batch_size =  100 #10000
    #bcs_dataset = DataGenerator(u_bcs_train, v_bcs_train, h_bcs_train, s_u_train, s_v_train, batch_size)
    
    #res_dataset = DataGenerator(u_res_train, v_res_train, h_res_train, s_res_train, s_res_train, batch_size)
    

    # Train
    #model.train(bcs_dataset, res_dataset, nIter=1000000)
    
    # Test data
    #N_test = 100 # number of input samples
    P_test = m   # number of sensors
    
    # region prediction 
    # Predict
    with open('DeepONet_ED_109_119.pkl', 'rb') as f:
        params = pickle.load(f)
        
    '''params = model.get_params(model.opt_state)
    with open('DeepONet_ED_109_119.pkl', 'wb') as f:
        pickle.dump(params, f)


    #Plot for loss function
    plot_loss(model.loss_bcs_log, model.loss_res_log)'''

    #real test   
    os.chdir(os.path.join(originalDir, './' + 'FE_full_elasto_dynamic_ground_truth' + '/'))
    
 
    import numpy as npr # jnp donesn't have loadtxt
    trans = 0.3
    X1_real = npr.loadtxt('X1.txt') #- trans
    Y1_real = npr.loadtxt('Y1.txt') #- trans 

    U1 = npr.loadtxt('U1 ts = 112.txt')
    V1 = npr.loadtxt('V1 ts = 112.txt')
    U1_new = npr.loadtxt('U1 ts = 113.txt')
    V1_new = npr.loadtxt('V1 ts = 113.txt')
    vx = npr.loadtxt('vx ts = 112.txt')/100
    vy = npr.loadtxt('vy ts = 112.txt')/100

    index_right_real = np.concatenate([
    np.where((Y1_real == y_val) & (X1_real == x_val))[0]
    for x_val, y_val in zip(X1[index_right], Y1[index_right])
    ])  
    index_left_real = np.concatenate([
    np.where((Y1_real == y_val) & (X1_real == x_val))[0]
    for x_val, y_val in zip(X1[index_left], Y1[index_left])
    ])
    index_up_real = np.concatenate([
    np.where((Y1_real == y_val) & (X1_real == x_val))[0]
    for x_val, y_val in zip(X1[index_up], Y1[index_up])
    ])
    index_down_real = np.concatenate([
    np.where((Y1_real == y_val) & (X1_real == x_val))[0]
    for x_val, y_val in zip(X1[index_down], Y1[index_down])
    ])
    
    u_r = U1_new[index_right_real]
    u_l = U1_new[index_left_real]
    u_up = U1_new[index_up_real]
    u_down = U1_new[index_down_real]
    v_r = V1_new[index_right_real]
    v_l = V1_new[index_left_real]
    v_up = V1_new[index_up_real]
    v_down = V1_new[index_down_real]

    os.chdir(originalDir_real)

    # interpolate data 
    U1_d = Rbf(X1_real, Y1_real, U1)(X1_, Y1_)
    V1_d = Rbf(X1_real, Y1_real, V1)(X1_, Y1_)
    vx_d = Rbf(X1_real, Y1_real, vx)(X1_, Y1_)
    vy_d = Rbf(X1_real, Y1_real, vy)(X1_, Y1_)

    u_bc = np.hstack([u_l.reshape(1,-1), u_r.reshape(1,-1), u_up.reshape(1,-1), u_down.reshape(1,-1)])
    v_bc = np.hstack([v_l.reshape(1,-1), v_r.reshape(1,-1), v_up.reshape(1,-1), v_down.reshape(1,-1)])

    u_test =np.hstack([U1_d.reshape(1,-1), V1_d.reshape(1,-1), vx_d.reshape(1,-1), vy_d.reshape(1,-1), u_bc.reshape(1,-1), v_bc.reshape(1,-1)]) 
    v_test =np.hstack([U1_d.reshape(1,-1), V1_d.reshape(1,-1), vx_d.reshape(1,-1), vy_d.reshape(1,-1), u_bc.reshape(1,-1), v_bc.reshape(1,-1)])
      
    y_test = np.hstack([X1_real.reshape(-1,1), Y1_real.reshape(-1,1)])
    s_u_pred, s_v_pred = model.predict_s(params, u_test, v_test, y_test)

    # Plot
    plot_disp(X1_real, Y1_real, s_u_pred, 'u_x_pred', rf'$u_{{x, \mathrm{{NO}},\Omega_{{II}}}}^{{{113}}}$')
    plot_disp(X1_real, Y1_real, s_v_pred,'u_y_pred',rf'$u_{{y, \mathrm{{NO}},\Omega_{{II}}}}^{{{113}}}$')
    plot_disp(X1_real, Y1_real,  U1_new.flatten(), 'u_x_truth', rf'$u_{{x, \mathrm{{FE}},\Omega_{{II}}}}^{{{113}}}$')
    plot_disp(X1_real, Y1_real,  V1_new.flatten(), 'u_y_truth', rf'$u_{{y, \mathrm{{FE}},\Omega_{{II}}}}^{{{113}}}$')
    plot_relative_error(X1_real, Y1_real, np.abs(s_u_pred.flatten() - U1_new.flatten()), 'u_x_error',rf'$|u_{{x, \mathrm{{FE}},\Omega_{{II}}}}^{{{113}}} - u_{{x, \mathrm{{NO}},\Omega_{{II}}}}^{{{113}}}|$')
    plot_relative_error(X1_real, Y1_real, np.abs(s_v_pred.flatten() - V1_new.flatten()), 'u_y_error',rf'$|u_{{y, \mathrm{{FE}},\Omega_{{II}}}}^{{{113}}} - u_{{y, \mathrm{{NO}},\Omega_{{II}}}}^{{{113}}}|$')
    














