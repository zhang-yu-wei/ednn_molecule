import numpy as np
import argparse
import h5py
import os
from progress import progress_timer

"""
This is the file which generates the data. The data contains the grid, the total 
energy, the ising model energy, the coulomb energy and the coordinates of the 
center of the chemical molecules
"""


def H(arr):
   #shift one to the right elements
   x = np.roll(arr,1,axis=1)
   #shift elements one down
   y = np.roll(arr,1,axis=0)
   #multiply original with transformations and sum each arr
   x = np.sum(np.multiply(arr,x))
   y = np.sum(np.multiply(arr,y))
   return -float(x+y)


def avg_grid(grid, original_size, new_size):
    avg_size = int(original_size / new_size)
    avg_list = []
    for i in range(new_size):
        for j in range(new_size):
            avg_list.append(grid[i*avg_size:(i+1)*avg_size, j*avg_size:(j+1)*avg_size])
    avg_list = [np.mean(avg_item) for avg_item in avg_list]
    data = np.reshape(avg_list, [new_size, new_size])
    return data


def gen_data(num_exm, total_grid_size=256, max_mole_dist=32, max_mole_size=30):
    """
    generate data
    :param num_exm: number of exmaples
    :param total_grid_size: size of studying area
    :param max_mole_size: maximum size of molecules
    """
    # initialize timer
    pt = progress_timer(description= 'creating examples', n_iter=num_exm)
    # set up a dict to memorize the data
    data = {'grid': [], 'total': [], 'ising': [],
            'elec': []}
    # compute maximum molecule number
    max_mole_num = ((total_grid_size // max_mole_dist) / 2) ** 2
    for z in range(num_exm):
        # initialize grid
        grid = np.zeros(shape=[total_grid_size, total_grid_size])
        total_energy = 0
        total_ising_energy = 0
        total_electric_energy = 0
        centers = []
        charges = []
        mole_num = np.random.randint(2, max_mole_num)
        while len(centers) < mole_num:
            coor = [np.random.randint(total_grid_size),
                    np.random.randint(total_grid_size)]
            real_coor = True
            for pre_coor in centers:
                if (abs(pre_coor[0] - coor[0]) < max_mole_dist) & (abs(
                        pre_coor[1] - coor[1]) < max_mole_dist):
                    real_coor = False
                    break
            if not real_coor:
                continue
            x_range_l = np.random.randint(max_mole_size / 2)
            if x_range_l > coor[0]:
                x_range_l = coor[0]
            y_range_d = np.random.randint(max_mole_size / 2)
            if y_range_d > coor[1]:
                y_range_d = coor[1]
            x_range_r = np.random.randint(max_mole_size / 2)
            if x_range_r > total_grid_size - coor[0] - 1:
                x_range_r = total_grid_size - coor[0] - 1
            y_range_u = np.random.randint(max_mole_size / 2)
            if y_range_u > total_grid_size - coor[1] - 1:
                y_range_u = total_grid_size - coor[1] - 1
            mole = np.round(np.random.uniform(size=[x_range_l +
                                        x_range_r + 1, y_range_d + y_range_u + 1]))
            mole = mole * 2 -1
            for m in range(x_range_l + x_range_r + 1):
                for n in range(y_range_d + y_range_u + 1):
                    grid.itemset((coor[0] - x_range_l + m, coor[1] - y_range_d + n),
                                 mole[m][n])
            ising_energy = H(mole)
            total_ising_energy += ising_energy
            charge = np.sum(mole)
            center = [(2*coor[0] - x_range_l + x_range_r)/2, (2*coor[1] - y_range_d + y_range_u)/2]
            for i in range(len(centers)):
                dist = np.sqrt(np.square(centers[i][0] - center[0]) +
                               np.square(centers[i][1] - center[1]))
                total_electric_energy += 10*charge*charges[i]/dist
            charges.append(charge)
            centers.append(center)
        total_energy = total_electric_energy + total_ising_energy
        data['grid'].append(grid)
        data['total'].append(total_energy)
        data['ising'].append(total_ising_energy)
        data['elec'].append(total_electric_energy)
        p_str = "generating done " + str(z)
        pt.update()
    pt.finish()
    return data


def compute_new_size(avg_size, focus):
    return (avg_size - focus) * 2 + avg_size


def padding(data, focus):
    data_size = np.shape(data)[0]
    new_size = compute_new_size(data_size, focus)
    new_data = np.zeros(shape=[new_size, new_size])
    start_ind = int((new_size - data_size) / 2)
    for i in range(data_size):
        for j in range(data_size):
            new_data.itemset((start_ind + i, start_ind + j), data[i][j])
    return new_data


def modify(data, original_size, new_size):
    return padding(avg_grid(data, original_size, new_size), focus)
    

def save_data(data_list, path, t):
    open_path = path + '/ori.hdf5'
    f = h5py.File(open_path, 'w')
    dset = f.create_dataset("ori_data", (len(data_list), total_grid_size, total_grid_size))
    d = 'saving ' + t + ' data'
    pt = progress_timer(description=d, n_iter=len(data_list))
    for i in range(len(data_list)):
        dset[i, ...] = data_list[i]
        pt.update()
    f.close()
    pt.finish()


def save_mo_data(data_list, path, t, new_size):
    open_path = path + '/avg.hdf5'
    f = h5py.File(open_path, 'w')
    size = compute_new_size(new_size, focus)
    dset = f.create_dataset("avg_data", (len(data_list), size, size))
    d = 'modifying ' + t + ' data'
    pt = progress_timer(description=d, n_iter=len(data_list))
    for i in range(len(data_list)):
        dset[i, ...] = modify(data_list[i], total_grid_size, new_size)
        pt.update()
    f.close()
    pt.finish()


def save_to_file(path, name, data_dict, new_size):
    save_data(data_dict['grid'], path, name)
    save_mo_data(data_dict['grid'], path, name, new_size)
    open_path = path + '/energy.hdf5'
    f = h5py.File(open_path, 'w')
    f['total'] = data_dict['total']
    f['ising'] = data_dict['ising']
    f['elec'] = data_dict['elec']
    f.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('save', help='which directory to save data')

    parser.add_argument('-ntr', help='number of training examples(default: 30000)', default=30000,
                        dest='num_tr', type=int)
    parser.add_argument('-nv', help='number of validation examples(default: 200)', default=200,
                        dest='num_v', type=int)
    parser.add_argument('-nte', help='number of test examples(default: 500)', default=500,
                        dest='num_te', type=int)
    parser.add_argument('-new', help='squeezed size(default: 16)', default=16,
                        dest='new_size', type=int)
    parser.add_argument('-f', help='focus size(default: 8)', default=8,
                        dest='focus', type=int)
    args = parser.parse_args()

    total_grid_size = 256
    focus = args.focus

    # seperate the train and test data
    print("generate training example")
    train_data = gen_data(args.num_tr)
    print("genrate validation example")
    valid_data = gen_data(args.num_v)
    print("generate test example")
    test_data = gen_data(args.num_te)

    path = args.save + '/train-data'
    save_to_file(path, 'training', train_data, args.new_size)
    path = args.save + '/valid-data'
    save_to_file(path, 'validation', valid_data, args.new_size)
    path = args.save + '/test-data'
    save_to_file(path, 'test', test_data, args.new_size)
