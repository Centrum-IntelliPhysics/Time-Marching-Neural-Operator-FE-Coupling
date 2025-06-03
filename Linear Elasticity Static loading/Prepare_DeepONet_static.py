"""
=======================================================================
Part of the FEM-DeepONet coupling work
-----------------------------------------------------------------------
DeepONet Model for Linear Elasticity under static loading conditions 
2D plane strain problem
Square + circular domain (overlapping boudnary)
"""

# Commented out IPython magic to ensure Python compatibility.
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
import math  
from scipy.interpolate import griddata
import pickle
import scipy
from scipy.interpolate import Rbf, interp1d, griddata
from utils import plot_loss, plot_bc, createFolder, plot_disp, plot_relative_error
import numpy as npr

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

        
        params_u = (branch_params, trunk_params)
        params_v = (branch_params_v, trunk_params)

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
    def operator_net_u(self, params, u, x, y):
        branch_params, trunk_params = params
        h = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
        B = self.branch_apply(branch_params, u)  
        T = self.trunk_apply(trunk_params, h)
        #print('B.shape=', B.shape, 'T.shape=', T.shape)

        # Compute the final output
        # Input and output shapes:
        # branch: [batch_size, 2m]--> [batch_size, 800]
        # trunk: [p, 2]--> [p, 800]
        # output by Einstein summation: [batch_size, p]
        outputs = np.einsum('ij,kj->ik', B, T)
        
        #print(outputs.shape)
        return  outputs
    
    @partial(jax.jit, static_argnums=(0,))
    def operator_net_v(self, params, v, x, y):
        branch_params, trunk_params = params
        h = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
        B = self.branch_apply(branch_params, v) 
        T = self.trunk_apply(trunk_params, h)
        # Compute the final output
        # Input shapes:
        # Input and output shapes:
        # branch: [batch_size, 2m]--> [batch_size, 800]
        # trunk: [p, 2]--> [p, 800]
        # output by Einstein summation: [batch_size, p]
        outputs = np.einsum('ij,kj->ik', B, T)
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
    

        para1 = self.E/((1 + self.nu)*(1-2*self.nu))
        ###Newton's second law in plane strain 
        res0 = para1*((1-self.nu) * s_u_xx + self.nu * s_v_xy) + para1*(1-2*self.nu)/2*(s_u_yy + s_v_xy)
        res1 = para1*((1-self.nu) * s_v_yy + self.nu * s_u_xy) + para1*(1-2*self.nu)/2*(s_v_xx + s_u_xy)

        return res0, res1

    # region stress
    def stress(self, params, u, v, Y_star):
        params_u, params_v = params
        x, y = Y_star[:,0], Y_star[:,1]
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
        s_u_pred = self.operator_net_u(params_u, u, h[:, 0], h[:, 1])
        s_v_pred = self.operator_net_v(params_v, v, h[:, 0], h[:, 1])
        # Compute loss
        loss =  np.mean((outputs[0] - s_u_pred)**2) + np.mean((outputs[1] - s_v_pred)**2)
        return loss

    # Define residual loss
    def loss_res(self, params, batch):
        # Fetch data
        # inputs: (u1, y), shape = (Nxm, m), (Nxm,1)
        # outputs: u2, shape = (Nxm, 1) zero array
        
        
        inputs, outputs = batch
        u, v, h = inputs
        # Compute forward pass
        res0, res1 = self.residual_net(params, u, v, h[:,0], h[:,1])
        #print(pred.shape, outputs.shape)
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
### symetric RBF (Radial basis function) for the Gaussian random field
def RBF(x1, x2, params): #radial basis function 
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales , 1) - \
            np.expand_dims(x2 / lengthscales , 0)
    r2 = np.abs(np.sum(diffs, axis=2))
    ub = np.ones(r2.shape)*np.max(x1 / lengthscales)
    d = ub -r2
    C  = np.minimum(r2, d)

    return output_scale**2 * np.exp(-0.5 * C**2)* (0.5 * C)


def RBF_data_generation(key, m, length_scale):
    # Generate subkeys
    xmin, xmax = 0, 10
    key, subkey0, subkey1= random.split(key, 3)
    subkeys0 = random.split(subkey0, 2)
    subkeys2 = random.split(subkey1, 2)
    # Generate a GP sample
    N = 512
    length_scale_u = length_scale[0]
    length_scale_v = length_scale[1]
    gp_params_u1 = (0.0008, length_scale_u)
    gp_params_v1 = (0.005, length_scale_v)

    jitter = 1e-20
    X = np.linspace(xmin, xmax, N)[:,None]

    K_u1 = RBF(X, X, gp_params_u1)

    D, V = npr.linalg.eigh(K_u1 + jitter*np.eye(N))
    D = np.maximum(D, 0)
    L_u1 = V @ np.diag(np.sqrt(D))

    K_v1 = RBF(X, X, gp_params_v1)
    D_v1, V_v1 = npr.linalg.eigh(K_v1 + jitter*np.eye(N))
    D_v1 = np.maximum(D_v1, 0)
    L_v1 = V_v1 @ np.diag(np.sqrt(D_v1))
    
    def gp_sample(key, L):
        gp_sample = np.dot(L, random.normal(key, (N,)))
        return gp_sample
    
    gp_sample_u1 = vmap(gp_sample, (0,None))(subkeys0, L_u1)
    gp_sample_v1 = vmap(gp_sample, (0,None))(subkeys2, L_v1)

    # Create a callable interpolation function
    f_fn_u1 = lambda x: vmap(np.interp,(None, None, 0))(x, X.flatten(), gp_sample_u1)
    f_fn_v1 = lambda x: vmap(np.interp,(None, None, 0))(x, X.flatten(), gp_sample_v1)

    x_c = np.linspace(xmin, xmax, m)
    u_c1, _ = f_fn_u1(x_c)
    v_c1, _ = f_fn_v1(x_c)

    return u_c1, v_c1 

# region training data 
def generate_one_training_data(key, m, Q, N):
    keys = random.split(key, N)
    # Generate N samples of boundary conditions
    # shape of u_c and v_c is (N, m)
    u_c, v_c = vmap(RBF_data_generation, (0, None, None))(keys, m, length_scale)

    # The boundary sensors in fixed sequence
    num_points = m  # m is the sensor number 
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

    # traning data for boundary conditions
    h_train_l = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
    h_train =np.tile(h_train_l, (N,1,1)).reshape(N, m, 2)    
    u_train = np.hstack([u_c.reshape(-1,m), v_c.reshape(-1,m)])

    # plot the generated boundary conditions (test RBF's performance)
    plot_bc(np.linspace(0,1,m),u_c.reshape(-1,m)[0,:],'u_l_0')
    plot_bc(np.linspace(0,1,m),u_c.reshape(-1,m)[5,:],'u_l_5')
    plot_bc(np.linspace(0,1,m),v_c.reshape(-1,m)[0,:],'v_l_0')
    plot_bc(np.linspace(0,1,m),v_c.reshape(-1,m)[5,:],'v_l_5')    

    v_train = np.hstack([u_c.reshape(-1,m), v_c.reshape(-1,m)])
    s_u_train = u_c.reshape(-1,1)
    s_v_train = v_c.reshape(-1,1)

    # randomly generate collocation points in the disk domain
    def generate_collocation_points(key, Q):
        subkeys = random.split(key, 2)
        r = random.uniform(subkeys[0], (Q,1), minval=0, maxval=radius)
        theta = random.uniform(subkeys[1], (Q,1), minval=0, maxval=2*np.pi)   
        xc = center[0] + r * np.cos(theta)
        yc = center[1] + r * np.sin(theta)
        
        return np.hstack([xc, yc])
    subkeys_ = random.split(key, N)
   
    
    # Training data for the PDE residual
    u_r_train = np.hstack([u_c, v_c])
    v_r_train = np.hstack([u_c, v_c])
    h_r_train = vmap(generate_collocation_points, (0,None))(subkeys_, Q)
    s_r_train = np.zeros((N,Q,1)) 

    return u_train, v_train,h_train, s_u_train, s_v_train, u_r_train, v_r_train, h_r_train, s_r_train


# Geneate training data corresponding to N input sample
def generate_training_data(key, N, m, Q):
    config.update("jax_enable_x64", True) # RBF_data_generation needs double precision
    u_train, v_train, h_train, s_u_train, s_v_train, u_r_train, v_r_train, h_r_train, s_r_train = generate_one_training_data(key, m, Q ,N)

    u_train = np.float32(u_train.reshape(N ,-1))  #turn to be (N,m)
    v_train = np.float32(v_train.reshape(N ,-1)) 
    h_train = np.float32(h_train.reshape(N , m,-1))
    s_u_train = np.float32(s_u_train.reshape(N,-1))
    s_v_train = np.float32(s_v_train.reshape(N,-1))
    u_r_train = np.float32(u_r_train.reshape(N ,-1))
    v_r_train = np.float32(v_r_train.reshape(N ,-1))
    h_r_train = np.float32(h_r_train.reshape(N , Q ,-1))
    s_r_train = np.float32(s_r_train.reshape(N ,-1))
    config.update("jax_enable_x64", False)
    return     u_train, v_train,h_train, s_u_train,  s_v_train, u_r_train, v_r_train, h_r_train, s_r_train
    


# GRF length scale
length_scale = [2, 2] #length_scale for symetric RBF 

# parameters used in the model
d = 2
N = 1000 # number of input samples
m = 200  # number of input sensors
Q_train = 1600 #400  # number of collocation points for each input sample
center = (0.0, 0.5, 0.0)  # Center of the circle
radius = 0.3             # Radius of the circle
# region main 
if __name__ == "__main__":
    originalDir = os.getcwd()  # Get the current working directory
    os.chdir(os.path.join(originalDir))

    foldername = 'Pretrained_DeepONet_static'  
    createFolder(foldername)
    os.chdir(os.path.join(originalDir, './'+ foldername + '/'))
    # elastic model parameters
    ela_model = dict()
    ela_model['E'] = 0.210e-2 # Young's modulus
    ela_model['nu'] = 0.3  # Poisson's ratio
    os.chdir(os.path.join(originalDir, './'+ foldername + '/'))
    origin_real  = os.path.join(originalDir, './'+ foldername + '/')

    # Generate training data
    key = random.PRNGKey(0)
    u_bcs_train, v_bcs_train,h_bcs_train, s_u_train, s_v_train, u_res_train, v_res_train, h_res_train, s_res_train\
            = generate_training_data(key, N, m, Q_train)
   
    # Initialize model
    branch_layers = [2*m, 100, 100, 100, 100, 800]
    trunk_layers =  [d, 100, 100, 100, 100, 800]
    model = PI_DeepONet(branch_layers, trunk_layers, **ela_model)
    
    # Create data set
    batch_size =  100 

    # dataset for boundary conditions and initial conditions
    bcs_dataset = DataGenerator(u_bcs_train, v_bcs_train, h_bcs_train, s_u_train, s_v_train, batch_size)
    res_dataset = DataGenerator(u_res_train, v_res_train,h_res_train, s_res_train, s_res_train, batch_size)
    
    # Train
    model.train(bcs_dataset, res_dataset, nIter=2000000)
    

    # region prediction 
    # Predict
    '''with open('DeepONet_static.pkl', 'rb') as f:
        params = pickle.load(f)'''

    params = model.get_params(model.opt_state)
    with open('DeepONet_static.pkl', 'wb') as f:
        pickle.dump(params, f)


    #Plot for loss function
    plot_loss(model.loss_bcs_log, model.loss_res_log)
    npr.savetxt('loss_bcs_log.txt', model.loss_bcs_log)
    npr.savetxt('loss_res_log.txt', model.loss_res_log)

    # load the data 
    os.chdir(originalDir + '/'+ 'static_data_ground_truth' + '/')
    x_c = np.array(npr.loadtxt('xc1_out.txt')).reshape(1,-1)
    y_c = np.array(npr.loadtxt('yc1_out.txt')).reshape(1,-1)
    X1 = np.array(npr.loadtxt('X1.txt')).reshape(1,-1)
    Y1 = np.array(npr.loadtxt('Y1.txt')).reshape(1,-1)
    u = np.array(npr.loadtxt('u.txt')).reshape(1,-1) # ground truth from full sqaure 
    v = np.array(npr.loadtxt('v.txt')).reshape(1,-1)
    X = np.array(npr.loadtxt('X.txt')).reshape(1,-1)
    Y = np.array(npr.loadtxt('Y.txt')).reshape(1,-1)
    os.chdir(origin_real)   

    u2_fun = Rbf(X, Y, u)
    v2_fun = Rbf(X, Y, v)

    # interpolate the data to the ground truth
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
    

    y_test = np.hstack([x_c.reshape(-1,1), y_c.reshape(-1,1)])

    u_test = np.hstack([u2c, v2c]).reshape(1,-1) 
    v_test = np.hstack([u2c, v2c]).reshape(1,-1)

    # attention: the boundary points are in a fixed order, so we need to resort the data
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
    

    u2c = u2c[0,index].reshape(1,-1)
    v2c = v2c[0,index].reshape(1,-1)

    u_test = np.hstack([u2c, v2c]).reshape(1,-1) 
    v_test = np.hstack([u2c, v2c]).reshape(1,-1)

    y_test = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
    s_u_pred, s_v_pred= model.predict_s(params, u_test, v_test, y_test)
    y_test_c = np.hstack([x_c.reshape(-1,1), y_c.reshape(-1,1)])
    sigmax_pred0, _, _ = model.stress(params, u_test, v_test, y_test_c)

    #plot the boundary lines 
    plot_bc(np.linspace(0,1,int(s_u_pred[0,:].shape[0])),s_u_pred[0,:int(u_test[0,:].shape[0])],'test u_l_bc')
    plot_bc(np.linspace(0,1,int(u2c.shape[1])),u2c.flatten(),'test u_l_bc1')
    plot_bc(np.linspace(0,1,int(u2c.shape[1])),s_u_pred[0,:int(u_test[0,:].shape[0])] - u2c.flatten(),'test u_l_error')
    
    plot_bc(np.linspace(0,1,int(s_v_pred[0,:].shape[0])),s_v_pred[0,:int(v_test[0,:].shape[0])],'test v_l_bc')
    plot_bc(np.linspace(0,1,int(v2c.shape[1])), v2c.flatten(),'test v_l_bc1')
    plot_bc(np.linspace(0,1,int(v2c.shape[1])),s_v_pred[0,:int(v_test[0,:].shape[0])] - v2c.flatten(),'test v_l_error')


    # Plot the disk domain 
    y_test = np.hstack([X1.reshape(-1,1), Y1.reshape(-1,1)])
    s_u_pred, s_v_pred= model.predict_s(params, u_test, v_test, y_test)
    plot_disp(X1,Y1, s_u_pred, 's_u_test1_0', rf'$u_{{\mathrm{{NO}},\Omega_{{II}}}}$')
    plot_disp(X1,Y1, u2, 's_u_real_0', rf'$u_{{\mathrm{{FE}},\Omega_{{II}}}}$')
    plot_disp(X1,Y1, s_v_pred,'s_v_test1_0',  rf'$v_{{\mathrm{{NO}},\Omega_{{II}}}}$')
    plot_disp(X1,Y1, v2, 's_v_real_0',  rf'$v_{{\mathrm{{FE}},\Omega_{{II}}}}$')

    plot_relative_error(X1,Y1, np.abs(s_u_pred - u2), 's_u_error_01', rf'$|u_{{\mathrm{{FE}},\Omega_{{II}}}} - u_{{\mathrm{{NO}},\Omega_{{II}}}}|$')
    plot_relative_error(X1,Y1, np.abs(s_v_pred - v2), 's_v_error_01', rf'$|v_{{\mathrm{{FE}},\Omega_{{II}}}} - v_{{\mathrm{{NO}},\Omega_{{II}}}}|$')
    













