import os
import sys
import numpy as np
import torch
from scipy import integrate
import matplotlib.pyplot as plt

solve_ivp = integrate.solve_ivp

from synthetic_utils import update, total_energy

class three_body_dataset_creator(object):
    def __init__(self, save=True):
        path = 'datasets/synthetic_threebody'
        if not os.path.exists(path):
            os.makedirs(path)

        # print(data.shape) # (500, 6, 5, 50)
        if save:
            data = self.two_pair_setting_randomtime(time_select=50)
            np.savez(os.path.join(path, 'test2'), data=data)

    def two_pair_setting_randomtime(self,
                                    trials=500,
                                    timesteps=200,
                                    t_span=[0, 10],
                                    orbit_noise=1e-1,
                                    time_select=50):
        data = []

        for i in range(trials):
            orbits = []
            for pair in range(2):
                state = self.random_config(nu=orbit_noise)
                orbit, settings = self.get_orbit(state, t_points=timesteps,
                                                 t_span=t_span, nbodies=3)  # orbit shape [2, 5, timesteps]

                start_point = torch.randint(low=0, high=int(timesteps - time_select), size=(1,)).item()
                orbits.append(orbit[:, :, start_point:int(start_point + time_select)])

            orbits = np.concatenate(orbits, axis=0)  # [6, 5, timestamp]
            data.append(orbits)

        data = np.stack(data, axis=0)  # [trial, 6, 5, timestamp]
        return data

    def get_one_visual(self, name='1'):

        timesteps = 200
        orbit_noise = 1e-1
        t_span = [0, 10]
        time_select = 50

        state = self.random_config(nu=orbit_noise)
        orbit, settings = self.get_orbit(state, t_points=timesteps, t_span=t_span, nbodies=3)

        start_point = torch.randint(low=0, high=int(timesteps - time_select), size=(1,)).item()
        orbit_select = orbit[:, :, start_point:int(start_point + time_select)]

        print(orbit.shape)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        axes[0].set_title('position')
        axes[0].plot(orbit[0, 1, :], orbit[0, 2, :], c='r', label='object 1', linestyle='dashed')
        axes[0].plot(orbit[1, 1, :], orbit[1, 2, :], c='b', label='object 2', linestyle='dashed')
        axes[0].plot(orbit[2, 1, :], orbit[2, 2, :], c='g', label='object 3', linestyle='dashed')
        axes[0].scatter(orbit[:, 1, 0], orbit[:, 2, 0], marker='x', c='k', label='start')

        axes[0].plot(orbit_select[0, 1, :], orbit_select[0, 2, :], c='r', linewidth=4)
        axes[0].plot(orbit_select[1, 1, :], orbit_select[1, 2, :], c='b', linewidth=4)
        axes[0].plot(orbit_select[2, 1, :], orbit_select[2, 2, :], c='g', linewidth=4)

        axes[0].set_xlim([-2.5, 2.5])
        axes[0].set_ylim([-2.5, 2.5])
        axes[0].legend()

        axes[1].set_title('velocity')
        axes[1].plot(orbit[0, 3, :], orbit[0, 4, :], c='r', label='object 1')
        axes[1].plot(orbit[1, 3, :], orbit[1, 4, :], c='b', label='object 2')
        axes[1].plot(orbit[2, 3, :], orbit[2, 4, :], c='g', label='object 3')
        axes[1].scatter(orbit[:, 3, 0], orbit[:, 4, 0], marker='x', c='k', label='start')

        #plt.show()
        plt.savefig('3body_example{}.eps'.format(name))

    def random_config(self, nu=2e-1, min_radius=0.9, max_radius=1.2):
        '''This is not principled at all yet'''
        state = np.zeros((3, 5))
        state[:, 0] = 1

        # state[0, 0] = 1
        # state[1, 0] = 1.2
        # state[2, 0] = 1.2

        # p1 = 2 * np.random.rand(2) - 1 # [from -1 to 1 for x and y?]
        p1 = np.random.rand(2) * (max_radius - min_radius) + min_radius  # [0.9, 1.2] for both]
        r = np.random.rand() * (max_radius - min_radius) + min_radius

        p1 *= r / np.sqrt(np.sum((p1 ** 2)))
        p2 = self.rotate2d(p1, theta=2 * np.pi / 3)
        p3 = self.rotate2d(p2, theta=2 * np.pi / 3)

        # # velocity that yields a circular orbit
        v1 = self.rotate2d(p1, theta=np.pi / 2)
        v1 = v1 / r ** 1.5
        v1 = v1 * np.sqrt(np.sin(np.pi / 3) / (2 * np.cos(np.pi / 6) ** 2))  # scale factor to get circular trajectories
        v2 = self.rotate2d(v1, theta=2 * np.pi / 3)
        v3 = self.rotate2d(v2, theta=2 * np.pi / 3)

        # make the circular orbits slightly chaotic
        v1 *= 1 + nu * (2 * np.random.rand(2) - 1)
        v2 *= 1 + nu * (2 * np.random.rand(2) - 1)
        v3 *= 1 + nu * (2 * np.random.rand(2) - 1)

        state[0, 1:3], state[0, 3:5] = p1, v1
        state[1, 1:3], state[1, 3:5] = p2, v2
        state[2, 1:3], state[2, 3:5] = p3, v3
        return state

    def get_orbit(self, state, update_fn=update,
                  t_points=100, t_span=[0, 2], nbodies=3, **kwargs):
        if not 'rtol' in kwargs.keys():
            kwargs['rtol'] = 1e-9

        orbit_settings = locals()

        nbodies = state.shape[0]
        t_eval = np.linspace(t_span[0], t_span[1], t_points)
        orbit_settings['t_eval'] = t_eval

        path = solve_ivp(fun=update_fn, t_span=t_span, y0=state.flatten(),
                         t_eval=t_eval, **kwargs)
        orbit = path['y'].reshape(nbodies, 5, t_points)
        return orbit, orbit_settings

    @staticmethod
    def rotate2d(p, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return (R @ p.reshape(2, 1)).squeeze()

    def sample_orbits(self, timesteps=20, trials=5000, nbodies=3, orbit_noise=2e-1,
                      min_radius=0.9, max_radius=1.2, t_span=[0, 5], verbose=False, **kwargs):
        orbit_settings = locals()
        if verbose:
            print("Making a dataset of near-circular 3-body orbits:")

        x, dx, e = [], [], []
        N = timesteps * trials
        while len(x) < N:

            state = self.random_config(nu=orbit_noise, min_radius=min_radius, max_radius=max_radius)
            orbit, settings = self.get_orbit(state, t_points=timesteps, t_span=t_span, nbodies=nbodies, **kwargs)
            batch = orbit.transpose(2, 0, 1).reshape(-1, nbodies * 5)

            for state in batch:
                dstate = update(None, state)

                # reshape from [nbodies, state] where state=[m, qx, qy, px, py]
                # to [canonical_coords] = [qx1, qx2, qy1, qy2, px1,px2,....]
                coords = state.reshape(nbodies, 5).T[1:].flatten()
                dcoords = dstate.reshape(nbodies, 5).T[1:].flatten()
                x.append(coords)
                dx.append(dcoords)

                shaped_state = state.copy().reshape(nbodies, 5, 1)
                e.append(total_energy(shaped_state))

        data = {'coords': np.stack(x)[:N],
                'dcoords': np.stack(dx)[:N],
                'energy': np.stack(e)[:N]}
        return data, orbit_settings


creater = three_body_dataset_creator(save=False)
for name in range(20):
    creater.get_one_visual(name='{}'.format(name))