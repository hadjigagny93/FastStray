import numpy as np 
import pandas as pd
from dataclasses import dataclass  
from  scipy.stats import pearsonr as corr

@dataclass 
class FastStray:
    """A global class that implements faststray algorithms for reduction GPS data point in a given trajectory 
    For more information about the method, you can check the paper on https://arxiv.org/pdf/1608.07338.pdf
    
    attributes
    ----------
    alpha: size of the moving average filter 
    beta: size of the neighborhood to measure the correaltion coef
    gamma: size of the neighboorhood to perform the non maximum suppression
    position: lon, lat, timestamp data dim = (n_samples, 3)
    spatial_position: lon, lat dim = (n_samples, 2)
    temporal_position: timestamp dim = (n_samples, )
    filtering_spatial_data: moving average on spatial_position
    filtering_temporal_position: moving average on temporal_position data 

    methods:
    --------
    linear_correlation_based_coef: static method, evaluate the correlation between space and time informations 
    ksi_function: more larger is ksi the more information we can extract from the related point 
    moving_average_smooth_trajectory: moving average 
    update_filtering_position: return of MA methods 
    get_params_idx: return intervall which one apply MA algorithm on -- this allows specifying windows for inference

    """
    test_coeff: int = 1
    alpha: float = 20
    beta: float = 20
    gamma: float = 20
    error: np.ndarray = np.random.exponential(0.1, (1000, 3)) * test_coeff
    position: np.ndarray = np.vstack([np.linspace(-3, 3, 1000), np.sin(3*np.linspace(-3, 3, 1000)), np.linspace(0, 1, 1000)]).T + error 
    sample_size: int = position.shape[0]
    coeff: np.ndarray = np.zeros(sample_size)
    max_coeff: np.ndarray = np.zeros(sample_size)
    spatial_dim: tuple = (sample_size, 2)
    temporal_dim: tuple = (sample_size, 1)

    def split_position(self):
        self.spatial_position = self.position[:,0:2]
        self.temporal_position = self.position[:, 2].reshape(-1, 1)
        return True 

    def moving_average(self):
        _ = self.split_position()
        """Calculate the trajectory 𝑇-1 (composed by a list of points 𝑃1 and time stamps 𝑆1) using moving average filter 
        -- alpha param defining the window of the filter -- the filter is computing on the whole trajectory """
        new_spatial_position = np.zeros(self.spatial_dim) # init positions to zero
        for i in range(self.sample_size):
            j_index = self.get_params_idx(idx=i, value=self.alpha, sample_size=self.sample_size)
            new_spatial_position[i] = self.spatial_position[j_index,:].mean(axis=0)
        return new_spatial_position, self.temporal_position

    def update_filtering_position(self):
        self.filtering_spatial_position, self.filtering_temporal_position = self.moving_average()

    def update_coeff(self):
        for i in range(self.sample_size):
            j_index = self.get_params_idx(idx=i, value=self.beta, sample_size=self.sample_size)
            P_mu = self.filtering_spatial_position[j_index,:]
            t_mu = self.filtering_temporal_position[j_index,:]
            self.coeff[i] = self.ksi_func(P_mu, t_mu)

    def update_max_coeff(self):
        for i in range(self.sample_size):
            j_index = self.get_params_idx(idx=i, value=self.gamma, sample_size=self.sample_size)
            self.max_coeff[i] = max(map(self.coeff.__getitem__, j_index))

    def simplified_trajectory(self):
        final_index = np.where(self.max_coeff == self.coeff)[0]
        self.simplified_spatial_position = self.filtering_spatial_position[final_index,:]
        self.simplified_temporal_position = self.filtering_temporal_position[final_index,:]

    def run(self):
        self.update_filtering_position()
        self.update_coeff()
        self.update_max_coeff()
        self.simplified_trajectory()

    @staticmethod 
    def get_params_idx(idx, value, sample_size):
        inf, sup = max(0, idx - value), min(idx + value, sample_size)
        return np.arange(inf, sup)
        
    @staticmethod
    def ksi_func(pp, tt):
        ppx, ppy = pp.T
        rat_x = corr(ppx, tt.flatten())[0] ** 2
        rat_y = corr(ppy, tt.flatten())[0] ** 2
        return 1./ rat_x + 1./ rat_y
