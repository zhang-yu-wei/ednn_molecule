import numpy as np
import argparse
import h5py

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
    # set up a dict to memorize the data
    data = {'grid': [], 'energy': [], 'isenergy': [],
            'elecenergy': []}
    # compute maximum molecule number
    max_mole_num = ((total_grid_size // max_mole_dist) / 2) ** 2
    for z in range(num_exm):
        # initialize grid
        grid = np.zeros(shape=[total_grid_size, total_grid_size])
        total_energy = 0
        total_ising_energy = 0
        total_electric_energy = 0
        coors = []
        charges = []
        mole_num = np.random.randint(2, max_mole_num)
        while len(coors) < mole_num:
            coor = [np.random.randint(total_grid_size),
                    np.random.randint(total_grid_size)]
            real_coor = True
            for pre_coor in coors:
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
            if x_range_r > total_grid_size - coor[0]:
                x_range_r = total_grid_size - coor[0]
            y_range_u = np.random.randint(max_mole_size / 2)
            if y_range_u > total_grid_size - coor[1]:
                y_range_u = total_grid_size - coor[1]
            mole = np.round(np.random.uniform(size=[x_range_l +
                                        x_range_r, y_range_d + y_range_u]))
            mole = mole * 2 -1
            for m in range(x_range_l + x_range_r):
                for n in range(y_range_d + y_range_u):
                    grid.itemset((coor[0] - x_range_l + m, coor[1] - y_range_d + n),
                                 mole[m][n])
            ising_energy = H(mole)
            total_ising_energy += ising_energy
            charge = np.sum(mole)
            for i in range(len(coors)):
                dist = np.sqrt(np.square(coors[i][0] - coor[0]) +
                               np.square(coors[i][1] - coor[1]))
                total_electric_energy += 10*charge*charges[i]/dist
            charges.append(charge)
            coors.append(coor)
        total_energy = total_electric_energy + total_ising_energy
        data['grid'].append(grid)
        data['energy'].append(total_energy)
        data['isenergy'].append(total_ising_energy)
        data['elecenergy'].append(total_electric_energy)
        p_str = "generating done " + str(z)
        print(p_str)
    return data


def padding(data, focus):
    data_size = np.shape(data)[0]
    new_size = (data_size - focus) * 2 + data_size
    new_data = np.zeros(shape=[new_size, new_size])
    start_ind = int((new_size - data_size) / 2)
    for i in range(data_size):
        for j in range(data_size):
            new_data.itemset((start_ind + i, start_ind + j), data[i][j])
    return new_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-ntr', help='number of training examples', default=50000,
                        dest='num_tr', type=int)
    parser.add_argument('-nv', help='number of validation examples', default=200,
                        dest='num_v', type=int)
    parser.add_argument('-nte', help='number of test examples', default=500,
                        dest='num_te', type=int)
    parser.add_argument('-new', help='what size you want the data to be', default=16,
                        dest='new_size', type=int)
    parser.add_argument('-f', help='focus size', default=8,
                        dest='focus', type=int)
    args = parser.parse_args()

    total_grid_size = 256
    focus = args.focus

    # seperate the train and test data
    train_data = gen_data(args.num_tr, args.new_size)
    valid_data = gen_data(args.num_v, args.new_size)
    test_data = gen_data(args.num_te, args.new_size)

    path = args.save + '/train-data.hdf5'
    f1 = h5py.File(path, "w")
    f1['ori_data'] = train_data['grid']
    f1['avg_data'] = [padding(avg_grid(data, total_grid_size, args.new_size)
                            , focus) for data in train_data['grid']]
    f1['energy'] = train_data['energy']
    f1['isenergy'] = train_data['isenergy']
    f1['elecenergy'] = train_data['elecenergy']

    path = args.save + '/valid-data.hdf5'
    f2 = h5py.File(path, "w")
    f2['ori_data'] = valid_data['grid']
    f2['avg_data'] = [padding(avg_grid(data, total_grid_size, args.new_size)
                            , focus) for data in valid_data['grid']]
    f2['energy'] = valid_data['energy']
    f2['isenergy'] = valid_data['isenergy']
    f2['elecenergy'] = valid_data['elecenergy']

    path = args.save + '/test-data.hdf5'
    f3 = h5py.File(path, "w")
    f3['ori_data'] = test_data['grid']
    f3['avg_data'] = [padding(avg_grid(data, total_grid_size, args.new_size)
                            , focus) for data in test_data['grid']]
    f3['energy'] = test_data['energy']
    f3['isenergy'] = test_data['isenergy']
    f3['elecenergy'] = test_data['elecenergy']
 
