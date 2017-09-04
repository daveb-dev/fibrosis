import numpy as np
import pickle

path = 'experiments/transpiration_sensitivity/results/'


def average_noise(case, seeds, out=''):
    err_u = []
    err_dP = []
    err_dP_ppe = []
    err_dP_ste = []
    err_dP_stei = []
    x = []
    for s in seeds:
        fstr = path + case.format(s=s) + '/estim_results.dat'
        print('reading ' + fstr)
        with open(fstr, 'rb') as fin:
            data = pickle.load(fin, encoding='latin1')
        err_u.append(data['err_u'])
        err_dP.append(data['err_dP'])
        err_dP_ppe.append(data['err_dP_ppe'])
        err_dP_ste.append(data['err_dP_ste'])
        err_dP_stei.append(data['err_dP_stei'])
        x.append(data['x_opt'])

    u_mean = np.array(err_u).mean()
    u_std = np.array(err_u).std()
    err_dP_mean = np.array(err_dP).mean()
    err_dP_std = np.array(err_dP).std()
    err_dP_ppe_mean = np.array(err_dP_ppe).mean()
    err_dP_ppe_std = np.array(err_dP_ppe).std()
    err_dP_ste_mean = np.array(err_dP_ste).mean()
    err_dP_ste_std = np.array(err_dP_ste).std()
    err_dP_stei_mean = np.array(err_dP_stei).mean()
    err_dP_stei_std = np.array(err_dP_stei).std()
    x_mean = np.array(x).mean(axis=0)
    x_std = np.array(x).std(axis=0)

    #  save_pickle = {
    #      'x': x_mean,
    #      'err_u': u_mean,
    #      'err_dP': dP_mean,
    #      'err_dP_stei': dP_stei_mean,
    #      'err_dP_ste': dP_ste_mean,
    #      'err_dP_ppe': dP_ppe_mean,
    #      'x_std': x_std,
    #      'err_u_std': u_std,
    #      'err_dP_std': dP_std,
    #      'err_dP_stei_std': dP_stei_std,
    #      'err_dP_ste_std': dP_ste_std,
    #      'err_dP_ppe_std': dP_ppe_std,
    #  }

    outfile = out if out else (case.format(s='') +
                               '{0}-{1}'.format(min(seeds), max(seeds)) +
                               '_statistics.npz')
    np.savez(path + outfile,
             x=x_mean, x_std=x_std,
             err_u=u_mean, err_u_std=u_std,
             err_dP=err_dP_mean, err_dP_std=err_dP_std,
             err_dP_ppe=err_dP_ppe_mean, err_dP_ppe_std=err_dP_ppe_std,
             err_dP_ste=err_dP_ste_mean, err_dP_ste_std=err_dP_ste_std,
             err_dP_stei=err_dP_stei_mean, err_dP_stei_std=err_dP_stei_std)

    print('writing ' + path + outfile)

    pass

if __name__ == '__main__':
    cases = ['s1_f0.6_noise_seed{s}', 's1_f0.6_noslip_noise_seed{s}']
    seeds = range(100)
    for c in cases:
        average_noise(c, seeds)
