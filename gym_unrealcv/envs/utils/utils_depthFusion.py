from __future__ import division, absolute_import, print_function
import os, sys, time, re, json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from math import sin, cos, radians
# def write_pose(x, y, z, roll, pitch, yaw, num=2, filename='pose.txt'):
#
#     f = open(filename, 'w+')
#     for i in range(len(x)):
#         # f.write(filename,'w+')
#         f.write('%6.4f %6.4f %6.4f\n' % (x[i], y[i], z[i]))
#         f.write('%6.4f %6.4f %6.4f\n' % (roll[i], pitch[i], yaw[i]))
#
#     f.close()

def poseOrigin(pose_filename):

    # pose = [0.0,0.0,0.0,180.0,0.0,-90.0]
    # pose = [0.0,0.0,0.0,-90.0,0.0,180.0]
    pose = [0.0,0.0,0.0,0.0,-90.0,180.0]
    write_pose(pose, pose_filename)

    return pose


def poseRelToAbs(azimuth, elevation,distance):

    p = 90
    yaw_exp = 270 - azimuth
    pitch = -1 * elevation
    y = distance*sin(radians(p-elevation))*cos(radians(azimuth))
    x = distance*sin(radians(p-elevation))*sin(radians(azimuth))
    z = distance*cos(radians(p-elevation))
    # rotation = [0, yaw_exp, pitch]
    pose = [x, y, z, 0, yaw_exp,pitch]
    return pose

def write_pose(pose, filename='pose.txt'):

    x, y, z, pitch, yaw, roll = pose
    print('pose.... ', pose)
    rot_mat = eulerAnglesToRotationMatrix([math.radians(pitch),math.radians(yaw),math.radians(roll)]) # worked on pure elevation trans3
    # rot_mat = eulerAnglesToRotationMatrix([math.radians(roll),math.radians(yaw),math.radians(pitch)])
    # rot_mat = eulerAnglesToRotationMatrix([math.radians(roll),math.radians(pitch),math.radians(yaw)])
    # rot_mat = np.linalg.inv(rot_mat)
    # trans_1 = np.array(loc['x'],loc['y'],loc['z'])
    trans = np.zeros((3,1))

    # loc['y'] = -1 * loc['y']
    # print('loc',loc['x'],loc['y'],loc['z'] )
    # trans[:,0] = np.transpose(np.array((loc['x'],loc['y'],loc['z'])))
    trans[:,0] = np.transpose(np.array((-1*y,z,-1*x)))
    # trans[:,0] = np.transpose(np.array((loc['x'],-1*loc['y'],loc['z'])))
    trans[:,0] = -1*trans[:,0]

    # print('rotation')
    # print(roll)
    # print(pitch)
    # print(yaw)
    # print(rot_mat)
    # print(rot_mat.shape)
    # print(trans)
    trans = trans/100 # 1/1 meters
    # print(trans.shape)
    pose = np.hstack((rot_mat,trans))
    last_row = np.zeros((1,4))
    last_row[:,3] = 1
    pose = np.vstack((pose,last_row))
    # pose = np.linalg.inv(pose)
    # print('pose')
    # print(pose)
    # np.savetxt('pose_out.txt', pose)

    np.savetxt(filename, pose)


    # f = open(filename, 'w+')
    #
    # f.write('%6.4f %6.4f %6.4f\n' % (x, y, z))
    # f.write('%6.4f %6.4f %6.4f\n' % (roll, pitch, yaw))
    #
    # f.close()

def write_depth(depth_np_filename, depth):
    np.save(depth_np_filename, depth)


def eulerAnglesToRotationMatrix(theta):
    print('theta 0 ', theta[0], theta[1], theta[2])

    R_x = np.array([[1.,         0.,                  0.                   ],
                    [0.,         math.cos(theta[0]), -1*math.sin(theta[0]) ],
                    [0.,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])



    R_y = np.array([[math.cos(theta[1]),    0.,      math.sin(theta[1])  ],
                    [0.,                     1.,      0.                   ],
                    [-math.sin(theta[1]),   0.,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -1*math.sin(theta[2]),    0.],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0.],
                    [0.,                     0.,                      1.]

                    ])

    # R = np.dot(R_z, np.dot( R_y, R_x ))
    R = np.matmul(np.matmul( R_z, R_y ), R_x)

    return R

def depth_conversion(PointDepth, f):
    H = PointDepth.shape[0]
    W = PointDepth.shape[1]
    i_c = np.float(H) / 2 - 1
    j_c = np.float(W) / 2 - 1
    columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
    DistanceFromCenter = ((rows - i_c)**2 + (columns - j_c)**2)**(0.5)
    # PlaneDepth = (PointDepth-1) / (1 + (DistanceFromCenter / f)**2)**(0.5)
    PlaneDepth = (PointDepth) / (1 + (DistanceFromCenter / f)**2)**(0.5)
    return PlaneDepth



mod = SourceModule("""
    __global__ void Integrate(float * cam_K, float * cam2base, float * depth_im,
                   int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                   float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
                   float * voxel_grid_TSDF, float * voxel_grid_weight) {

      int pt_grid_z = blockIdx.x;
      int pt_grid_y = threadIdx.x;

      for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x) {

        // Convert voxel center from grid coordinates to base frame camera coordinates
        float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
        float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
        float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

        // Convert from base frame camera coordinates to current frame camera coordinates
        float tmp_pt[3] = {0};
        tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
        tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
        tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
        float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
        float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
        float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

        if (pt_cam_z <= 0)
          continue;

        int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
        int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
        if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
          continue;

        float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];

        if (depth_val <= 0 || depth_val > 6)
          continue;

        float diff = depth_val - pt_cam_z;

        if (diff <= -trunc_margin)
          continue;

        // Integrate
        int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
        float dist = fmin(1.0f, diff / trunc_margin);
        float weight_old = voxel_grid_weight[volume_idx];
        float weight_new = weight_old + 1.0f;
        voxel_grid_weight[volume_idx] = weight_new;
        voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
      }
    }

    """)


def LoadMatrixFromFile(filename,M=3,N=3):
    #tmp = np.array((1,2), dtype=np.float32)
    #i = 0;
    tmp = []
    with open(filename) as f:
        for line in f:
            line_str = line.split()
            #line_str = float(line_str)
            #tmp.append(x)
            #tmp.append(y)
            #tmp.append(z)
            #print("type of line")
            #print(type(line_str))
            #print(line_str)
            #for i in range(N):
            #    line_str.pop()
            line_str.reverse()
            while(line_str):
                tmp.append(float(line_str.pop()))

            #element = line_str.pop()
            #print
    return tmp

def SaveVoxelGrid2SurfacePointCloud(filename,voxel_grid_dim_x,voxel_grid_dim_y,voxel_grid_dim_z, voxel_size,voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
    voxel_grid_TSDF,voxel_grid_weight,tsdf_thresh, weight_thresh, save_pcd = False):

    '''count total number of points in point cloud'''

    ply_header = '''ply
format ascii 1.0
element vertex %(num_pts)d
property float x
property float y
property float z
end_header
'''

    num_pts = 0
    '''
    for i in range (0, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z):
        if (abs(voxel_grid_TSDF[i])< tsdf_thresh and voxel_grid_weight[i] > weight_thresh):
            num_pts= num_pts+1
    print("Array addition     ")
    print(num_pts)
    '''
    mask = (abs(voxel_grid_TSDF)<tsdf_thresh) & (voxel_grid_weight>weight_thresh)
    num_pts = np.sum(mask)
    print("SDFasdf NUMPTS DDFS     ")
    print(num_pts)
    with open(filename, 'wb') as f:
        f.write((ply_header % dict(num_pts=num_pts)).encode('utf-8'))
        #np.savetxt(f, coordinate, fmt='%f %f %f ')


        mask = (abs(voxel_grid_TSDF)<tsdf_thresh) & (voxel_grid_weight>weight_thresh)
        num_pts = np.sum(mask)

        i = np.arange(voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z)

        z = np.zeros((num_pts),dtype='int32')
        y = np.zeros((num_pts),dtype='int32')
        x = np.zeros((num_pts),dtype='int32')

        z = np.floor(i[mask]/(voxel_grid_dim_x*voxel_grid_dim_y))
        y = np.floor((i[mask]-(z*voxel_grid_dim_z*voxel_grid_dim_y))/voxel_grid_dim_x)
        x = np.int32(i[mask] -(z*voxel_grid_dim_x*voxel_grid_dim_y)-(y*voxel_grid_dim_x))

        pt_base_x = np.float32(voxel_grid_origin_x + np.float32(x) * voxel_size).reshape(num_pts,1)
        pt_base_y = np.float32(voxel_grid_origin_y + np.float32(y) * voxel_size).reshape(num_pts,1)
        pt_base_z = np.float32(voxel_grid_origin_z + np.float32(z) * voxel_size).reshape(num_pts,1)

        #coordinates = np.zeros((num_pts,3),dtype='float32')
        coordinates = np.hstack((pt_base_x,pt_base_y,pt_base_z))
        if save_pcd == True:
            np.savetxt(f, coordinates, fmt='%f %f %f ')
                #f.write(pt_base_x)
                #f.write(pt_base_y)
                #f.write(pt_base_z)


def depth_fusion(data_path = 'log', cam_K_file = 'camera-intrinsics.txt', output_pts = 'log/tsdf_house.ply', first_frame_idx =1,base_frame_idx=1,num_frames=2,npy=True, max_depth = 3.0, save_pcd =False):

    im_width = 640
    im_height = 480

    voxel_grid_origin_x = -1.5
    voxel_grid_origin_y = -1.5
    voxel_grid_origin_z = -0.5
    voxel_size = 0.006

    voxel_grid_origin_x = -0.5
    voxel_grid_origin_y = -0.5
    voxel_grid_origin_z = -0.5
    # voxel_size = 0.005
    voxel_size = 0.010
    # voxel_size = 0.03
    # voxel_size = 0.12

    trunc_margin = voxel_size * 5
    # voxel_grid_dim_x = 500
    # voxel_grid_dim_y = 500
    # voxel_grid_dim_z = 500
    # voxel_grid_dim_x = 600
    # voxel_grid_dim_y = 600
    # voxel_grid_dim_z = 600
    # voxel_grid_dim_x = 200
    # voxel_grid_dim_y = 200
    # voxel_grid_dim_z = 200
    voxel_grid_dim_x = 100
    voxel_grid_dim_y = 100
    voxel_grid_dim_z = 100


    # voxel_size = 0.006
    # voxel_grid_dim_x = 500
    # voxel_grid_dim_y = 500
    # voxel_grid_dim_z = 500


    '''read camera intrinsics'''
    cam_K = LoadMatrixFromFile(cam_K_file,3,3)
    # print('camk')
    # print(cam_K)
    #cam_K_np = np.array(cam_K).reshape(3,3)
    cam_K_np = np.array(cam_K,dtype='float32')
    #cam_K_np.reshape(3,3)
    # print(cam_K_np)

    '''read base frame camera pose'''
    base2world_file = data_path+'/frame-'+'{:06}'.format(base_frame_idx)+'.pose.txt'
    base2world = LoadMatrixFromFile(base2world_file)
    base2world_np = np.array(base2world,dtype='float32').reshape(4,4)
    # print(base2world_np)
    # print(base2world_np.dtype)
    # print("asdfasfd")

    '''invert base frame camera pose to get world-to-base frame transform'''
    base2world_inv = np.linalg.inv(base2world_np)
    # print('inverse')
    # print(base2world_inv)

    '''flatten again the camera poses'''
    base2world_np_flat = base2world_np.flatten()
    base2world_inv_flat = base2world_inv.flatten()

    '''initialize voxel grid'''
    voxel_grid_TSDF = np.ones((voxel_grid_dim_x*voxel_grid_dim_y*voxel_grid_dim_z),dtype='float32')
    voxel_grid_weight = np.zeros((voxel_grid_dim_x*voxel_grid_dim_y*voxel_grid_dim_z),dtype='float32')


    # print('voxel_grid_TSDF')
    # print(voxel_grid_TSDF)
    # print(voxel_grid_TSDF.shape)

    # print('voxel_grid_TSDF')
    # print(voxel_grid_weight)
    # print(voxel_grid_weight.shape)

    '''Load variables to GPU memory'''
    gpu_voxel_grid_TSDF = cuda.mem_alloc(voxel_grid_TSDF.nbytes)
    gpu_voxel_grid_weight = cuda.mem_alloc(voxel_grid_weight.nbytes)
    cuda.memcpy_htod(gpu_voxel_grid_TSDF,voxel_grid_TSDF)
    cuda.memcpy_htod(gpu_voxel_grid_weight,voxel_grid_weight)

    '''add CUDA error check'''

    gpu_cam_K = cuda.mem_alloc(cam_K_np.nbytes)
    cuda.memcpy_htod(gpu_cam_K,cam_K_np)
    gpu_cam2base = cuda.mem_alloc(base2world_np_flat.nbytes)
    '''initialize depth image'''
    depth_im = np.zeros((im_height,im_width),dtype='float32')
    depth_norm_flat = depth_im.flatten()
    gpu_depth_im = cuda.mem_alloc(depth_norm_flat.nbytes)

    '''add CUDA error check'''

    '''Loop through each depth frame and integrate TSDF voxel grid'''
    print('Loop through each depth frame and integrate TSDF voxel grid')
    for frame_idx in range(first_frame_idx,first_frame_idx+num_frames):

        curr_frame_prefix = '{:06}'.format(frame_idx)
        #print(curr_frame_prefix)
        # print('current frame prefix', curr_frame_prefix)
        '''read current frame depth'''

        # color_im_file = data_path + '/frame-' + curr_frame_prefix + '.color.png'
        #print(depth_im_file)
        if npy:
            depth_im_file = data_path +'/frame-' + curr_frame_prefix + '.depth.npy'
            depth_im = np.load(depth_im_file)
            # print(depth_im_file)
        else:
            depth_im_file = data_path +'/frame-' + curr_frame_prefix + '.depth.png'
            depth_im = cv2.imread(depth_im_file,cv2.IMREAD_UNCHANGED)
        # print('max and min depth', np.max(depth_im))
        # print('max and min depth', np.max(depth_im), np.min(depth_im))
        # depth_norm = depth_im/1000.0
        depth_norm = depth_im/1.0
        mask_depth = depth_norm > max_depth
        depth_norm[mask_depth] = max_depth
        depth_norm_flat = depth_norm.flatten().astype(np.float32)


        ''' read base frame camera pose '''
        cam2world_file = data_path+'/frame-' + curr_frame_prefix + '.pose.txt'
        cam2world = LoadMatrixFromFile(cam2world_file)
        cam2world_np = np.array(cam2world, dtype='float32').reshape(4,4)
        # print('cam2world', cam2world_np)
        '''Compute relative camera pose (camera-to-base frame)'''
        cam2base = np.dot(base2world_inv,cam2world_np)
        # print('cam2base', cam2base)
        cam2base_flat = cam2base.flatten()
        cuda.memcpy_htod(gpu_cam2base,cam2base_flat)
        cuda.memcpy_htod(gpu_depth_im,depth_norm_flat)
        #print('fusing')
        integrate_func = mod.get_function('Integrate')
        integrate_func(gpu_cam_K,gpu_cam2base,gpu_depth_im,np.int32(im_height),np.int32(im_width), np.int32(voxel_grid_dim_x), np.int32(voxel_grid_dim_y), np.int32(voxel_grid_dim_z),
            np.float32(voxel_grid_origin_x), np.float32(voxel_grid_origin_y), np.float32(voxel_grid_origin_z), np.float32(voxel_size), np.float32(trunc_margin), gpu_voxel_grid_TSDF, gpu_voxel_grid_weight,
            block=(voxel_grid_dim_y,1,1), grid=(voxel_grid_dim_y,1))


    #print('cam2base')
    #print(cam2base)

    '''Load TSDF voxel grid from GPU to CPU memory'''

    cuda.memcpy_dtoh(voxel_grid_TSDF, gpu_voxel_grid_TSDF)
    cuda.memcpy_dtoh(voxel_grid_weight, gpu_voxel_grid_weight)

    tsdf_thresh = 0.2
    weight_thresh =0.0
    #mask = (abs(voxel_grid_TSDF)<tsdf_thresh) and (voxel_grid_weight>weight_thresh)
    mask = (abs(voxel_grid_TSDF)<tsdf_thresh)
    #print('voxel grid')
    #mask2 =voxel_grid_TSDF)>tsdf_thresh
    #print(voxel_grid_TSDF[0:20])
    #print(np.sum(mask))

    print("Saving surface point cloud (tsdf.ply)...")


    output_pts = data_path + '/house-' + '{:06}'.format(num_frames) + '.ply'
    SaveVoxelGrid2SurfacePointCloud(output_pts, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                                  voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                                  voxel_grid_TSDF, voxel_grid_weight, 0.2, 0.0, save_pcd = save_pcd);
