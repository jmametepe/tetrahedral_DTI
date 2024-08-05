import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import time

def rotx(ang):
    return R.from_euler('x', ang, degrees=True).as_matrix()

def roty(ang):
    return R.from_euler('y', ang, degrees=True).as_matrix()

def rotz(ang):
    return R.from_euler('z', ang, degrees=True).as_matrix()

iter = 14000000

tit = np.zeros((iter, 3))
ll = np.zeros((iter, 3))
diffs = np.zeros((iter, 3))
final = np.zeros((iter, 12))
nfinal = np.zeros((iter, 12))
FA = np.zeros(iter)
evecs = np.zeros((iter, 3))

ta = 0
tb = 360
la = 0.003
lb = 0.0001

np.random.seed(int(time.time()))

t1 = np.linspace(ta, tb, iter)
t2 = np.linspace(ta, tb, iter)
t3 = np.linspace(ta, tb, iter)
np.random.shuffle(t1)
np.random.shuffle(t2)
np.random.shuffle(t3)
tit[:,0] = t1
tit[:,1]=t2
tit[:,2]=t3

#for n in range(3):
#    a = np.linspace(ta, tb, iter)
#    np.random.shuffle(a)
#    tit[:, n] = a

#for n in range(3):
  #  a = np.linspace(la, lb, iter)
 #   np.random.shuffle(a)
#    ll[:, n] = a



a1 = np.linspace(la, lb, iter)
a2 = np.linspace(la, lb, iter)
a3 = np.linspace(la, lb, iter)
np.random.shuffle(a1)
np.random.shuffle(a2)
np.random.shuffle(a3)
ll[:, 0] = a1
ll[:, 1] = a2
ll[:, 2] = a3
#print (ll)

bval = np.linspace(500, 1000, iter)
np.random.shuffle(bval)

for i in range(iter):
    vecs = rotx(tit[i, 0]) @ roty(tit[i, 1]) @ rotz(tit[i, 2])
    l = np.sort(ll[i, :])[-3:]
    #l = np.sort(ll[i, :])[-2:]
    #print(l)
    l = [l[0],l[1],l[2]]
    
   
    va = np.diag(l)
    D = vecs @ va @ vecs.T
    diffs[i, :] = l
    #evecs[i, :] = vecs[:, 2]

    #print('val') 
    #print(va)
    #print('vec')
    #print(vecs)
    #print(evecs)

    nu = 1 / np.sqrt(3)
    ang = np.random.randint(1, 21, 3)
    vec = rotx(ang[0]) @ roty(ang[1]) @ rotz(ang[2])
    initial_vectors = np.array([[nu, nu, nu],
                            [-nu, -nu, nu],
                            [nu, -nu, -nu],
                            [-nu, nu, -nu]]).T  # Transposed to make them column vectors

    # Rotate each vector using the rotation matrix
    n1 = vec @ initial_vectors[:, 0]
    n2 = vec @ initial_vectors[:, 1]
    n3 = vec @ initial_vectors[:, 2]
    n4 = vec @ initial_vectors[:, 3]

# n now holds the rotated vectors
    n = np.array([n1, n2, n3, n4])
    #print(n)
    np.random.shuffle(n)
    #print(n)

    S0 = np.abs((5000 - 1) * np.random.rand() + 1)
    b = bval[i]
    SS = np.zeros(5)
    SS[0] = S0

    for j in range(4):
        g = n[j, :]
        #print(g)
        SS[j + 1] = S0 * np.exp(-b * g @ D @ g.T)

    Df = np.log(SS[1:5] / SS[0]) / -b
    gg = Df[:, None] * n
    #print(gg)

    snr = 5
    noise = (SS[0] / snr) * (np.random.randn(5) + (1j * np.random.randn(5)) )

    hsig = SS*np.cos(np.pi/4)
    fsignal = np.abs(hsig + noise + (1j * hsig))

    nf = np.log(fsignal[1:5] / fsignal[0]) / -b
    ng = nf[:, None] * n

    final[i, :] = gg.flatten()
    #print(final)
    nfinal[i, :] = ng.flatten()
    FA[i] = np.sqrt(((l[0] - l[1])**2 + (l[0] - l[2])**2 + (l[1] - l[2])**2) / (2 * (l[0]**2 + l[1]**2 + l[2]**2)))

column_names = ['g1x', 'g1y', 'g1z', 'g2x', 'g2y', 'g2z', 'g3x', 'g3y', 'g3z', 'g4x', 'g4y', 'g4z']
final_df = pd.DataFrame(final, columns=column_names)
nfinal_df = pd.DataFrame(nfinal, columns=column_names)
diffs_df = pd.DataFrame(diffs, columns=['l1', 'l2', 'l3'])
#peig = pd.DataFrame(evecs, columns=['x', 'y', 'z'])
FA_df = pd.DataFrame(FA, columns=['FA'])

final_df.to_csv('4din.csv', index=False)
nfinal_df.to_csv('n4din.csv', index=False)

diffs_df.to_csv('4ddiff.csv', index=False)
FA_df.to_csv('4dFA.csv', index=False)

#peig.to_csv('vecs.csv', index=False)
