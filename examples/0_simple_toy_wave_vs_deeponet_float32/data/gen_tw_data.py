import numpy as np 
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

# plt.style.use('siads')


NT=20 # 20
NX=300

x = np.linspace(0,1,NX,endpoint=False)
t = np.linspace(0,70,NT,endpoint=False)


xx,tt=np.meshgrid(x,t)

omega = 70
c = 0.12/10
x0 = 0.1

# u = np.exp()
u = np.exp(-1000*(xx-x0-c*tt)**2)*np.sin(omega*(xx-x0-c*tt))



# print(tt.shape)
# print(u.shape)


# vis
plt.figure()
for i in range(NT):
    plt.plot(x,u[i,:],'-',label=str(i) + '-th time')

plt.savefig('u_at_diff_t.png')


# vis iso
plt.figure(figsize=(4,4))
ax = plt.axes(projection='3d')
ax.plot_surface(xx,tt,u,cmap="rainbow", lw=2)#,rstride=1, cstride=1)
ax.view_init(57, -80)
ax.set_xlabel(r'$x$',fontsize=25)
ax.set_ylabel(r'$t$',fontsize=25)
ax.set_zlabel(r'$u$',fontsize=25)

plt.tight_layout()
# plt.axis('off')
plt.savefig('3d.png')


# prepare tensor product data
# (20,300) # t,x
print(xx.shape)
# I need x,t,mu
xx_flatten = xx.reshape(-1,1)
tt_flatten = tt.reshape(-1,1)
u_flatten = u.reshape(-1,1)

raw_data = np.hstack((tt_flatten, xx_flatten, u_flatten))
raw_data = raw_data.astype('float32')

# print(raw_data.shape)

# normalize the data
# just simply all 0-1
mean = raw_data.mean(axis=0)
std = raw_data.std(axis=0)

# ok. we change our mind, try -1,1 for
# [t,x,u]
# for i in range(2):
#     mean[i] = 0.5*(np.min(raw_data[:,i])+np.max(raw_data[:,i]))
#     std[i] = 0.5*(-np.min(raw_data[:,i])+np.max(raw_data[:,i]))
#
# std[-1] = std[-1] * 5

normalized_data = (raw_data - mean)/std


print('normalized data mean = ',normalized_data.mean(axis=0))
print('normalized data std = ',normalized_data.std(axis=0))

print('normalized data min = ',normalized_data.min(axis=0))
print('normalized data max = ',normalized_data.max(axis=0))

np.savez('train.npz',data=normalized_data, std=std, mean=mean)
