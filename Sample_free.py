#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
import h5py


# In[2]:


# define a dot product function used for the rotate operation
def random_rotation_matrix():
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    return np.array([
        [cos_theta*cos_phi, -sin_phi, sin_theta*cos_phi],
        [cos_theta*sin_phi, cos_phi, sin_theta*sin_phi],
        [-sin_theta, 0, cos_theta]
    ])
class off_lattice_SAW:
    def __init__(self,N):
        self.N = N
        # initial configuration. Usually we just use a straight chain as inital configuration
        self.init_state = np.dstack((np.arange(N),np.zeros(N),np.zeros(N)))[0]
        self.state = self.init_state.copy()
    
    def pivot_transform(self, pick_pivot, pick_side, thres):
    
        chain = [self.state[0:pick_pivot+1], self.state[pick_pivot+1:]] 
        rotation_matrix = random_rotation_matrix()

        chain[pick_side] = np.dot(chain[pick_side] - self.state[pick_pivot], 
                                  rotation_matrix.T) + self.state[pick_pivot]

        overlap = distance.cdist(chain[0],chain[1])
        overlap = delete_lower_triangle_and_flatten(overlap, int(thres))

        return overlap, np.concatenate(chain, axis=0)
    
    # define pivot algorithm process where t is the number of successful steps
    def walk(self, t, thres):
        acpt = 0
        # while loop until the number of successful step up to t
        while acpt <= t:
            pick_pivot = np.random.randint(1,self.N-1) # pick a pivot site
            pick_side = np.random.choice([0,1]) # pick a side

            overlap, new_chain = self.pivot_transform(pick_pivot, pick_side, thres)

            # determinte whether the new state is accepted or rejected
            if (overlap < thres).sum() != 0:
                continue
            else:
                self.state = new_chain
                acpt += 1

        # place the center of mass of the chain on the origin
        self.state = (self.state - np.int_(np.mean(self.state,axis=0)))


# In[3]:


def delete_lower_triangle_and_flatten(arr, a):
    rows, cols = arr.shape
    mask = np.ones_like(arr, dtype=bool)
    
    for i in range(a):
        mask[rows-1-i, :(a-i)] = False
        
    flattened_array = arr[mask].flatten()
    return flattened_array


# In[5]:


def v_dot(a):return lambda b: np.dot(a,b)
class lattice_SAW:
    def __init__(self,N):
        self.N = N
        # initial configuration. Usually we just use a straight chain as inital configuration
        self.init_state = np.dstack((np.arange(N),np.zeros(N).astype(int),np.zeros(N).astype(int)))[0]
        self.state = self.init_state.copy()

        # define a rotation matrix
        # 9 possible rotations: 3 axes * 3 possible rotate angles(90,180,270)
        self.rotate_matrix = np.array([[[1,0,0],[0,0,-1],[0,1,0]],[[1,0,0],[0,-1,0],[0,0,-1]]
        ,[[1,0,0],[0,0,1],[0,-1,0]],[[0,0,1],[0,1,0],[-1,0,0]]
        ,[[-1,0,0],[0,1,0],[0,0,-1]],[[0,0,-1],[0,1,0],[-1,0,0]]
        ,[[0,-1,0],[1,0,0],[0,0,1]],[[-1,0,0],[0,-1,0],[0,0,1]]
        ,[[0,1,0],[-1,0,0],[0,0,1]]])
    
    def get_state(self):
        return self.state

    # define pivot algorithm process where t is the number of successful steps
    def walk(self, t, v):
        acpt = 0
        # while loop until the number of successful step up to t
        while acpt <= t:
            pick_pivot = np.random.randint(1,self.N-1) # pick a pivot site
            pick_side = np.random.choice([-1,1]) # pick a side

            if pick_side == 1:
                old_chain = self.state[0:pick_pivot+1]
                temp_chain = self.state[pick_pivot+1:]
            else:
                old_chain = self.state[pick_pivot:]
                temp_chain = self.state[0:pick_pivot]

            # pick a symmetry operator
            symtry_oprtr = self.rotate_matrix[np.random.randint(len(self.rotate_matrix))]
            # new chain after symmetry operator
            new_chain = np.apply_along_axis(v_dot(symtry_oprtr),1,temp_chain - self.state[pick_pivot]) + self.state[pick_pivot]

            # use cdist function of scipy package to calculate the pair-pair distance between old_chain and new_chain
            if v == 1:
                overlap = distance.cdist(new_chain,old_chain)
                overlap = overlap.flatten()
                # determinte whether the new state is accepted or rejected
                if len(np.nonzero(overlap)[0]) != len(overlap):
                    continue
            
            if pick_side == 1:
                self.state = np.concatenate((old_chain,new_chain),axis=0)
            elif pick_side == -1:
                self.state = np.concatenate((new_chain,old_chain),axis=0)
            acpt += 1

        # place the center of mass of the chain on the origin
        self.state = self.state - np.int_(np.mean(self.state,axis=0))


# In[7]:


def SAW_sample(chain, n, t, thres):
    config_l = []
    for _ in tqdm(range(int(n/t))):
        chain.walk(t, thres)
        config_l.append(chain.state)
    return config_l


# In[15]:


n = 100000
with h5py.File("free_pol.h5", 'w') as file:
    dataset = file.create_dataset('pol', (7,n, 100, 3))
    config_l=[]; i=0
    for thres in np.linspace(0,1.5,7):
        chain = off_lattice_SAW(100)
        config_l = SAW_sample(chain, n*1000, 1000, thres)
        dataset[i: i+1] = np.array(config_l)
        i += 1

