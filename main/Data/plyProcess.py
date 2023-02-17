from scipy.io import loadmat
def matToPly():
    pos = loadmat('./plyOutput-14/pos.mat')['pos']
    #pos = loadmat('/main/Data/plyOutput-14/pos.mat')['pos']
    T = pos.shape[0]
    N=  pos.shape[1]
    print(pos.shape)

    for t in range(1,T+1):
        write_ply('./plyFromMat+"/plyData_Frame', t, 3, N, pos[t-1],[], needVel=False)
def velpredToPly(posMat ,velMat):#根据vel生成粒子路径
    vel = loadmat(velMat)['vel_pred']# T * N * 3
    pos = loadmat('plyOutput-14/pos.mat')['pos']# T * N * 3 需要知道粒子的初始位置

    T=vel.shape[0]
    N=vel.shape[1]
    pos2=[]
    pos2.append(pos[0])
    deltaTime=0.1012
    print(pos2)
    for x in range(0,T):
        pos2.append(pos2[x]+vel[x]*deltaTime)
        
def write_ply(path, frame_num, dim, num, pos, vel, needVel=True):
    # 文件路径，当前帧数，维度，粒子数，位置,速度
    if(type(pos) is'numpy.ndarray'):
        print(type(pos))
        pos = pos.to_numpy()  # taichi don't support slice
    if(needVel):
        vel = vel.to_numpy()



    if dim == 3:
        if (needVel):
            list_pos = [(pos[i, 0], pos[i, 1], pos[i, 2], vel[i, 0], vel[i, 1], vel[i, 2]) for i in range(num)]
        else:
            list_pos = [(pos[i, 0], pos[i, 1], pos[i, 2]) for i in range(num)]
        # list_pos = [(pos[i, 0], pos[i, 1], pos[i, 2]) for i in range(num)]
        # list_vel = [(vel[i, 0], vel[i, 1], vel[i, 2]) for i in range(num)]
    elif dim == 2:
        list_pos = [(pos[i, 0], pos[i, 1], 0) for i in range(num)]
        list_vel = [(vel[i, 0], vel[i, 1], 0) for i in range(num)]
    else:
        print('write_ply(): dim exceeds default values')
        return
    # np_pos = np.array(list_pos, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    # np_vel = np.array(list_vel, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    # el_pos = PlyElement.describe(np_pos, 'vertex')
    # el_vel = PlyElement.describe(np_vel, 'velocity')
    if (needVel):
        np_pos = np.array(list_pos,
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4')])
    else:
        np_pos = np.array(list_pos, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    el_pos = PlyElement.describe(np_pos, 'vertex')
    PlyData(
        [el_pos],
        text=True  # ASCII
    ).write(str(path) + '_' + str(frame_num) + '.ply')
matToPly()