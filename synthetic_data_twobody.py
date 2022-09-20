import os
import sys
import pickle

import numpy as np
from scipy import integrate

import matplotlib.pyplot as plt
import torch

from synthetic_utils import update, total_energy

solve_ivp = integrate.solve_ivp


##### HELPER FUNCTION #####
def coords2state(coords, nbodies=2, mass=1):
    timesteps = coords.shape[0]
    state = coords.T
    state = state.reshape(-1, nbodies, timesteps).transpose(1, 0, 2)
    mass_vec = mass * np.ones((nbodies, 1, timesteps))
    state = np.concatenate([mass_vec, state], axis=1)
    return state


class two_body_dataset_creator(object):
    def __init__(self, CLS=20, save=True):
        path = 'datasets/synthetic_twobody'
        if not os.path.exists(path):
            os.makedirs(path)

        def good_data_1():
            data = self.two_pair_setting_randomtime(time_select=10)
            np.savez(os.path.join(path, 'test7'), data=data)
            '''test6 is with time_select=20, test7 is with time_select=10'''

        data = self.two_pair_setting_randomtime_diffmass(time_select=10)
        if save:
            np.savez(os.path.join(path, 'test8'), data=data)

        #a_file = open(os.path.join(path, 'test2.pkl'), "wb")
        #pickle.dump(data, a_file)
        #a_file.close()

    def two_pair_setting_randomtime_diffmass(self, trials=500, timesteps=50,
                                             t_span=[0, 10], orbit_noise=5e-2, time_select=10):
        data = []

        for i in range(trials):
            orbits = []
            for pair in range(2):
                mass1 = torch.rand(1) * 0.4 + 0.8  # range [0.8, 1.2]
                mass2 = torch.rand(1) * 0.4 + 1.2  # range [1.2, 1.6]

                state = self.random_config(orbit_noise, weight_ratio=None, mass1=mass1, mass2=mass2)
                orbit, settings = self.get_orbit(state, t_points=timesteps,
                                                 t_span=t_span)  # orbit shape [2, 5, timesteps]

                start_point = torch.randint(low=0, high=int(timesteps - time_select), size=(1,)).item()
                orbits.append(orbit[:, :, start_point:int(start_point + time_select)])

            orbits = np.concatenate(orbits, axis=0)  # [6, 5, timestamp]
            data.append(orbits)

        data = np.stack(data, axis=0)  # [trial, 6, 5, timestamp]

        return data

    def two_pair_setting_randomtime(self, trials=500, timesteps=50, t_span=[0, 10], orbit_noise=5e-2, time_select=20):
        """with different mass, but the same others"""
        data = []

        for i in range(trials):
            orbits = []
            for pair in range(2):
                mass1 = torch.rand(1) * 0.8 + 0.8  # range 0.8 -> 1.6
                mass2 = torch.rand(1) * 0.8 + 0.8

                state = self.random_config(orbit_noise, weight_ratio=None, mass1=mass1, mass2=mass2)
                orbit, settings = self.get_orbit(state, t_points=timesteps,
                                                 t_span=t_span)  # orbit shape [2, 5, timesteps]

                start_point = torch.randint(low=0, high=int(timesteps-time_select), size=(1,)).item()
                orbits.append(orbit[:, :, start_point:int(start_point+time_select)])

            orbits = np.concatenate(orbits, axis=0)  # [6, 5, timestamp]
            data.append(orbits)

        data = np.stack(data, axis=0)  # [trial, 6, 5, timestamp]

        return data

    def two_pair_setting(self, trials=500, timesteps=50, t_span=[0, 10], orbit_noise=5e-2):
        """with different mass, but the same others"""
        data = []

        for i in range(trials):
            orbits = []
            for pair in range(2):
                mass1 = torch.rand(1) * 0.8 + 0.8  # range 0.8 -> 1.6
                mass2 = torch.rand(1) * 0.8 + 0.8

                state = self.random_config(orbit_noise, weight_ratio=None, mass1=mass1, mass2=mass2)
                orbit, settings = self.get_orbit(state, t_points=timesteps,
                                                 t_span=t_span)  # orbit shape [2, 5, timesteps]

                orbits.append(orbit)
            orbits = np.concatenate(orbits, axis=0)  # [6, 5, timestamp]
            data.append(orbits)

        data = np.stack(data, axis=0)  # [trial, 6, 5, timestamp]

        return data

    def three_pair_setting3(self, trials=500, timesteps=50, t_span=[0, 10], orbit_noise=5e-2):
        data = []

        for i in range(trials):
            ratio_list = torch.rand(3) * 0.3 + 1.1
            orbits = []
            for ratio in ratio_list.tolist():
                state = self.random_config(orbit_noise, weight_ratio=ratio)
                orbit, settings = self.get_orbit(state, t_points=timesteps,
                                                 t_span=t_span)  # orbit shape [2, 5, timesteps]

                orbits.append(orbit)
            orbits = np.concatenate(orbits, axis=0)  # [6, 5, timestamp]
            data.append(orbits)

        data = np.stack(data, axis=0)  # [trial, 6, 5, timestamp]

        return data

    def three_pair_setting2(self, trials=200, timesteps=50, t_span=[0, 10], orbit_noise=5e-2, id=20):
        data = {}
        for id_i in range(id):
            ratio_list = torch.rand(3) * 0.3 + 1.1
            data['data{}'.format(id_i)] = self.three_pair_i(ratio_list, trials, timesteps, t_span, orbit_noise)

        return data

    def three_pair_setting(self, trials=500, timesteps=50, t_span=[0, 10], orbit_noise=5e-2):
        ratio_list1 = torch.rand(3) * 0.3 + 1.1
        data1 = self.three_pair_i(ratio_list1, trials, timesteps, t_span, orbit_noise)

        ratio_list2 = torch.rand(3) * 0.3 + 1.1 # 1.6 for balancing prob
        data2 = self.three_pair_i(ratio_list2, trials, timesteps, t_span, orbit_noise)

        return {'data1': data1,
                'data2': data2}

    def three_pair_i(self, ratio_list, trials=1000, timesteps=50, t_span=[0, 10], orbit_noise=5e-2):
        """three pair setting considers 3 balanced, 2b+1unb, 2unb+1b, 3 unb cases"""

        data = []

        for i in range(trials):
            orbits = []
            for ratio in ratio_list.tolist():
                state = self.random_config(orbit_noise, weight_ratio=ratio)
                orbit, settings = self.get_orbit(state, t_points=timesteps, t_span=t_span)  # orbit shape [2, 5, timesteps]

                orbits.append(orbit)
            orbits = np.concatenate(orbits, axis=0)  # [6, 5, timestamp]
            data.append(orbits)

        data = np.stack(data, axis=0) # [trial, 6, 5, timestamp]

        return data

    def random_config(self, orbit_noise=5e-2, min_radius=0.5, max_radius=1.5, weight_ratio=1, mass1=None, mass2=None):
        state = np.zeros((2, 5))
        if weight_ratio is not None:
            state[0, 0] = 1
            state[1, 0] = 1 * weight_ratio
        else:
            assert (mass1 is not None) and (mass2 is not None)
            state[0, 0] = mass1
            state[1, 0] = mass2

        pos = np.random.rand(2) * (max_radius - min_radius) + min_radius  # uniform distribution
        r = np.sqrt(np.sum((pos ** 2)))

        # velocity that yields a circular orbit
        vel = np.flipud(pos) / (2 * r ** 1.5)
        vel[0] *= -1
        vel *= 1 + orbit_noise * np.random.randn()

        # make the circular orbits SLIGHTLY elliptical
        state[:, 1:3] = pos
        state[:, 3:5] = vel
        state[1, 1:] *= -1
        return state

    def get_orbit(self, state, update_fn=update, t_points=100, t_span=[0, 2], **kwargs):
        if not 'rtol' in kwargs.keys():
            kwargs['rtol'] = 1e-9

        orbit_settings = locals()

        nbodies = state.shape[0]
        t_eval = np.linspace(t_span[0], t_span[1], t_points)
        orbit_settings['t_eval'] = t_eval

        path = solve_ivp(fun=update_fn, t_span=t_span, y0=state.flatten(), t_eval=t_eval, **kwargs)
        orbit = path['y'].reshape(nbodies, 5, t_points)
        return orbit, orbit_settings

    def get_one_visual(self, name='1'):
        orbit_noise = 5e-2
        timesteps = 50
        t_span = [0, 10]
        time_select = 10

        mass1 = torch.rand(1) * 0.8 + 0.8
        mass2 = torch.rand(1) * 0.8 + 0.8

        state = self.random_config(orbit_noise, weight_ratio=None, mass1=mass1, mass2=mass2)
        orbit, settings = self.get_orbit(state, t_points=timesteps, t_span=t_span)

        start_point = torch.randint(low=0, high=int(timesteps - time_select), size=(1,)).item()
        select_orbit = orbit[:, :, start_point:int(start_point + time_select)]

        """orbits = []
            for pair in range(2):
                mass1 = torch.rand(1) * 0.8 + 0.8  # range 0.8 -> 1.6
                mass2 = torch.rand(1) * 0.8 + 0.8

                state = self.random_config(orbit_noise, weight_ratio=None, mass1=mass1, mass2=mass2)
                orbit, settings = self.get_orbit(state, t_points=timesteps,
                                                 t_span=t_span)  # orbit shape [2, 5, timesteps]

                start_point = torch.randint(low=0, high=int(timesteps-time_select), size=(1,)).item()
                orbits.append(orbit[:, :, start_point:int(start_point+time_select)])"""

        print(orbit.shape)  # (2, 5, 1000) -- two body, 5 properties [mass, pos, vel], timesteps

        shaped_state = state.copy().reshape(2, 5, 1)
        print('energy', total_energy(shaped_state))

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        axes[0].set_title('position')
        axes[0].plot(orbit[0, 1, :], orbit[0, 2, :], c='r', label='object 1', linestyle='dashed')
        axes[0].plot(orbit[1, 1, :], orbit[1, 2, :], c='b', label='object 2', linestyle='dashed')

        axes[0].scatter(orbit[0, 1, 0], orbit[0, 2, 0], c='k', marker='x')
        axes[0].scatter(orbit[1, 1, 0], orbit[1, 2, 0], c='k', marker='x')

        axes[0].plot(select_orbit[0, 1, :], select_orbit[0, 2, :], c='r', linewidth=4)
        axes[0].plot(select_orbit[1, 1, :], select_orbit[1, 2, :], c='b', linewidth=4)

        axes[0].set_xlim([-2, 2])
        axes[0].set_ylim([-2, 2])
        axes[0].legend()


        axes[1].set_title('velocity')
        axes[1].plot(orbit[0, 3, :], orbit[0, 4, :], c='r', label='object 1')
        axes[1].plot(orbit[1, 3, :], orbit[1, 4, :], c='b', label='object 2')

        #plt.show()
        plt.savefig('2body_example{}.eps'.format(name))

    def sample_orbits(self,
                      timesteps=50,
                      trials=1000,
                      nbodies=2,
                      orbit_noise=5e-2,
                      min_radius=0.5,
                      max_radius=1.5,
                      t_span=[0, 20],
                      verbose=False,
                      **kwargs):
        orbit_settings = locals()
        print("Making a dataset of near-circular 2-body orbits:")

        x, dx, e = [], [], []
        N = timesteps * trials

        while len(x) < N:
            state = self.random_config(orbit_noise, min_radius, max_radius)
            orbit, settings = self.get_orbit(state, t_points=timesteps, t_span=t_span, **kwargs)
            batch = orbit.transpose(2, 0, 1).reshape(-1, 10)

            for state in batch:
                dstate = update(None, state)

                # reshape from [nbodies, state] where state=[m, qx, qy, px, py]
                # to [canonical_coords] = [qx1, qx2, qy1, qy2, px1,px2,....]
                coords = state.reshape(nbodies, 5).T[1:].flatten()
                dcoords = dstate.reshape(nbodies, 5).T[1:].flatten()
                x.append(coords)
                dx.append(dcoords)

                shaped_state = state.copy().reshape(2, 5, 1)
                e.append(total_energy(shaped_state))

        data = {'coords': np.stack(x)[:N],
                'dcoords': np.stack(dx)[:N],
                'energy': np.stack(e)[:N]}
        return data, orbit_settings


CT = two_body_dataset_creator(save=False)
for i in range(10):
    CT.get_one_visual(name='{}'.format(i))
# CT.get_one_visual()  # (2, 5, 50)