

import numpy as np 
import pandas as pd
from dataclasses import dataclass 
import matplotlib.pyplot as plt 
from  scipy.stats import pearsonr as corr

@dataclass 
class FastStray:
    """A global class that implements faststray algorithms for reduction GPS data point in a given trajectory 
    For more information about the method, you can check the paper on https://arxiv.org/pdf/1608.07338.pdf
    
    attributes
    ----------
    test: used for introducing error during test step, must be set to 0 when position is given as argument
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
    params: dict
    space: int = 2
    time: int = 1
    test_coeff: int = 1
    error: np.ndarray = np.random.exponential(0.1, (1000, 3)) * test_coeff
    position: np.ndarray = np.vstack([np.linspace(-3, 3, 1000), np.sin(3*np.linspace(-3, 3, 1000)), np.linspace(0, 1, 1000)]).T + error 
    sample_size: int = position.shape[0]
    spatial_dim: tuple = (sample_size, space)
    temporal_dim: tuple = (sample_size, time)

    def moving_average(self):
        """Calculate the trajectory 𝑇-1 (composed by a list of points 𝑃1 and time
         stamps 𝑆1) using moving average filter -- alpha param defining the window 
         of the filter -- the filter is computing on the whole trajectory 
        """
        self.spatial_position = self.position[:,:self.space]
        self.temporal_position = self.position[:,self.space].reshape(-1, 1)
        idx = self.get_index(param='alpha')
        new_spatial_position = np.array([*map(self.mean_position, idx)])
        self.filtering_spatial_position, self.filtering_temporal_position = new_spatial_position, self.temporal_position

    def mean_position(self, index):
        """return average spatial position"""
        return self.spatial_position[index,:].mean(axis=0)
    
    def sub_spatial_array(self, index):
        return self.filtering_spatial_position[index,:]
    
    def sub_temporal_array(self, index):
        return self.filtering_temporal_position[index,:]

    def sub_coeff(self, index):
        return max(map(self.coeff.__getitem__, index))

    def get_index(self, param):
        return [*map(self.get_params_idx, range(self.sample_size), [self.params[param]]*self.sample_size, [self.sample_size]*self.sample_size)]

    def update_coeff(self):
        idx = self.get_index(param='beta')
        p_mu = map(self.sub_spatial_array, idx)
        t_mu = map(self.sub_temporal_array, idx)
        self.coeff = [*map(self.ksi_func, p_mu, t_mu)]

    def update_max_coeff(self):
        idx = self.get_index(param='gamma')
        self.max_coeff = [*map(self.sub_coeff, idx)]

    def simplified_trajectory(self):
        compress_index = np.where(np.array(self.max_coeff) == np.array(self.coeff))[0]
        self.simplified_spatial_position = self.filtering_spatial_position[compress_index,:]
        self.simplified_temporal_position = self.filtering_temporal_position[compress_index,:]

    def run(self):
        self.moving_average()
        self.update_coeff()
        self.update_max_coeff()
        self.simplified_trajectory()

    @staticmethod 
    def get_params_idx(idx, value, sample_size):
        inf, sup = max(0, idx - value), min(idx + value, sample_size)
        return np.arange(inf, sup)

    @staticmethod
    def plot_average_filtering_deviation(init, avg):
        """ init is the noisy trajectory, avg the filter one """
        plt.scatter(avg[:, 0], avg[:, 1])
        plt.scatter(init[:, 0], init[:, 1], color='r')

    @staticmethod
    def ksi_func(pp, tt):
        ppx, ppy = pp.T
        rat_x = corr(ppx, tt.flatten())[0] ** 2
        rat_y = corr(ppy, tt.flatten())[0] ** 2
        return 1./ rat_x + 1./ rat_y
    
  
    def compression_rate(self):
        return 1. - (len(self.simplified_temporal_position) / len(self.temporal_position))

        

def main():
    fst = FastStray(params={'alpha':10, 'beta': 10, 'gamma': 10})
    fst.run()
    print(f"compression rate is about: {fst.compression_rate()}")
    #plt.scatter(fst.simplified_spatial_position[:, 0], fst.simplified_spatial_position[:, 1])


if __name__ == "__main__":
    main()ython 