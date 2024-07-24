import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pyvista as pv
import copy
import os
import meshio
import point_cloud_utils as pcu


def write_vtk(save_dir,file_name,data_array):
    ## Write a VTK files
    ## Head content: 
    ##     # vtk DataFile Version 2.0
    ##     rawdata, Created by Gmsh
    ##     ASCII
    ##     DATASET POLYDATA
    ##     POINTS 2774 double   
    num_point = data_array.shape[0] 
    text_file = open(save_dir+file_name, "w")
    text_file.write('# vtk DataFile Version 2.0\n')
    text_file.write('rawdata, Created by Gmsh\n')
    text_file.write('ASCII\n')
    text_file.write('DATASET POLYDATA\n')
    text_file.write('POINTS '+str(num_point)+' double\n')
    for i in range(num_point):
        text_file.write(str(data_array[i,0])+'\t'+str(data_array[i,1])+'\t'+str(data_array[i,2])+'\n')

    text_file.close()
    print('File saved')


### compare deformed mesh with reference mesh
### point_cor: if meshes have point to point correspondence
def compare_mesh(mesh1_dir,mesh2_dir,num_block,point_cor): 
    mesh1 = pv.read(mesh1_dir)
    mesh2 = pv.read(mesh2_dir)
    if point_cor == True:
        error_list=[]
        for block in range(num_block):
            points_mesh1 = mesh1[block].points
            points_mesh2 = mesh2[block].points
            error_list.append(np.mean(np.sqrt(np.sum(np.square(points_mesh1-points_mesh2),axis=1))))
        err = np.mean(error_list)
    else:
        points_mesh1 = mesh1.points
        points_mesh2 = mesh2[0].points
        for block in range(1,num_block):
            # points_mesh1 = np.vstack((points_mesh1,mesh1[block].points))
            points_mesh2 = np.vstack((points_mesh2,mesh2[block].points))
        points_mesh1 = np.unique(points_mesh1, axis=0)
        points_mesh2 = np.unique(points_mesh2, axis=0)
        # print(points_mesh1.shape)

        # write_vtk('./','mesh1.vtk',points_mesh1)
        # write_vtk('./','mesh2.vtk',points_mesh2)
        meshio.write("mesh1.ply", mesh=meshio.Mesh(points=points_mesh1, cells = []), binary=False)
        meshio.write("mesh2.ply", mesh=meshio.Mesh(points=points_mesh2, cells = []), binary=False)

        ### https://www.fwilliams.info/point-cloud-utils/sections/shape_metrics/
        p1 = pcu.load_mesh_v("mesh1.ply")
        p2 = pcu.load_mesh_v("mesh2.ply")

        ### Compute the chamfer distance between p1 and p2
        ### https://www.fwilliams.info/point-cloud-utils/sections/shape_metrics/
        err = pcu.chamfer_distance(p1, p2) 
    return err

num_core = 128
prior_mean = 5000
prior_variance = np.square(10000)
proposed_sample_current = copy.deepcopy(prior_mean)
assumption_variance = 1000
noise_variance = 0.1

posterior_list = []
difference_list = []
timestep = 50

### save initial guess of shear modulus and run in FreeFEM
np.savetxt("muguess.txt",[proposed_sample_current])
os.system("/home/yuehao/freefem/bin/ff-mpirun -np "+str(num_core)+" fsi.edp -v 0 ")
mesh1dir = 'deform/comparison.vtu' ### data
mesh2dir = 'trial_deform/deform_'+str(num_core)+'.pvd' ### trial
difference_old = compare_mesh(mesh1dir,mesh2dir,num_core,False)
print('Difference: ',difference_old)

posterior_list.append(proposed_sample_current)
difference_list.append(difference_old)

for t in range(timestep):
    print('Timestep:',t)
    ### Important! The workflow below this is now univaraite!!! output [sample_size->1,num_vague]
    proposed_sample_new = np.random.normal(proposed_sample_current,assumption_variance,1)
    print('Guess mu: ', proposed_sample_new)

    np.savetxt("muguess.txt",[proposed_sample_new])
    os.system("/home/yuehao/freefem/bin/ff-mpirun -np "+str(num_core)+" fsi.edp -v 0 ")

    difference_new = compare_mesh(mesh1dir,mesh2dir,num_core,False)
    print('Difference: ',difference_new)
    if difference_new <= difference_old:
        proposed_sample_current = proposed_sample_new
        posterior_list.append(proposed_sample_current.item())
        difference_old = difference_new
        difference_list.append(difference_old)
        print('Accept sample: ',proposed_sample_current)
        print(posterior_list)
        print(difference_list)
    else:
        print('Not accept')
        pass
        # print('Accept ratio: ',accept_ratio,'; Xnew: ',x_new,'; Reject')
    
    
# truncate_num = int(len(posterior_list)/4)
# posterior_list = np.asarray(posterior_list[truncate_num:])
np.save('mu.npy',np.array(posterior_list))
np.save('difference.npy',np.array(difference_list))

# sns.histplot(a_list)
# plt.show()