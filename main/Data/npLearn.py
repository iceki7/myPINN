import numpy as np
import scipy
import time

# idx = np.random.choice(20,7, replace=False)
# print(idx)

# snap = np.array([3])
# # print(snap)

# P_star=np.array([[3,5,7,4,2],[2,6,1,5,9]])
# p_star = P_star[:,snap]
# # print(p_star)


# data_1d_x = np.linspace(0, 5, 4, endpoint=True)
# data_1d_y = np.linspace(7, 9, 3, endpoint=True)
# data_1d_nu = np.linspace(10, 12,2, endpoint=True)
# # print(data_1d_x)
# # print(data_1d_y)
# # print(data_1d_nu)
# data_2d_xy_before = np.array(np.meshgrid(data_1d_x, data_1d_y, data_1d_nu))
# data_2d_xy_before_reshape = data_2d_xy_before.reshape(3, -1)
# data_2d_xy = data_2d_xy_before_reshape.T

# idx = np.random.choice(10, 10, replace=False)
# print(idx)
# print('\n')
# print(data_2d_xy)

# print('predict')
# print(data_2d_xy[:,0:1])
# print(data_2d_xy[:,1:2])
# print(data_2d_xy[:,2:3])

# t_star=scipy.io.loadmat('main/Data/time.mat')['time'] #1 T
# N = t_star.shape
# TT = np.tile(t_star.T, (1,5)) # T N
# print(N)
# print(t_star)
# print(TT)  #T N


pos_star=scipy.io.loadmat('main/Data/test2/pos.mat')['pos'] #T N 3
vel_star=scipy.io.loadmat('main/Data/test2/vel.mat')['vel'] #T N 3
t_star=scipy.io.loadmat('main/Data/test2/time.mat')['time'] #1 T
#mat 内容：
# N=5000 T=200
#时间等间距

N = pos_star.shape[1]
T = t_star.shape[1]

# Rearrange Data 

XX=pos_star[:,:,0]  # T N
YY=pos_star[:,:,1]
ZZ=pos_star[:,:,2]

print('XX shape')
print(XX.shape)
# XX = np.tile(X_star[:,0:1], (1,T)) # N x T  #位置分量
# YY = np.tile(X_star[:,1:2], (1,T)) # N x T
# ZZ = np.tile(X_star[:,2:3], (1,T)) # N x T
TT = np.tile(t_star.T, (1,N)) # T N
print('TT shape')
print(TT.shape)

UU = vel_star[:,:,0] # T x N    
VV = vel_star[:,:,1] # 
WW = vel_star[:,:,2] # 
print('UU shape')
print(UU.shape)
    
x = XX.flatten()[:,None] # TN x 1
y = YY.flatten()[:,None] #
z = ZZ.flatten()[:,None] #
t = TT.flatten()[:,None] #

u = UU.flatten()[:,None] # TN 1 
v = VV.flatten()[:,None] #
w = WW.flatten()[:,None] #
#p = PP.flatten()[:,None] #

######################################################################
######################## Noiseles Data ###############################
######################################################################
# Training Data    

#随机在整个N×T 时空域上取5k点作为训练集
N_train=32
idx = np.random.choice(N*T, N_train, replace=False)
x_train = x[idx,:]
y_train = y[idx,:]
z_train = z[idx,:]

t_train = t[idx,:]
u_train = u[idx,:]
v_train = v[idx,:]
w_train = w[idx,:]
print('u_train shape')
print(u_train.shape)


#batch 划分
batch=10
now=0
print()
while(True):
    
    if(now+batch<=N_train):
        ub=u_train[now:now+batch,:] #选择batch个数据  
        now+=batch
    else:
        ub=u_train[now:N_train,:] #选择batch个数据
        now=0  
    print('ub shape')
    print(ub.shape)
    time.sleep(5)
