import os

# def run_opensfm(images = '/home/daryl/OpenDroneMap_v0_3_1/images',out_path =  '/home/daryl/OpenDroneMap_v0_3_1/opensfm_try'):
#     client = docker.from_env()
#     vol1 = images + ':/code/images'
#     vol2 = out_path + ':/code/opensfm'
#     # client.containers.run("my_odm_image", tty = True, remove = True, volumes=['/home/daryl/OpenDroneMap_v0_3_1/images:/code/images','/home/daryl/OpenDroneMap_v0_3_1/opensfm2:/code/opensfm'])
#     ret_val = client.containers.run("my_odm_image", tty = True, remove = True, volumes=[vol1,vol2])
#     return ret_val

def run_opensfm(images = '/home/daryl/OpenDroneMap_v0_3_1/images', out_path =  '/home/daryl/OpenDroneMap_v0_3_1/opensfm_try', view_num=0, bin_path = '/home/daryl/OpenSfM-0.2.0/bin/opensfm_run_unreal '):

    cmd = bin_path + images
    so = os.popen(cmd).read()
    merged_depths = '/%dmerged.ply' % (view_num)
    copy_cmd = 'cp '+ images+'/depthmaps/merged.ply ' + out_path + merged_depths
    print('copy_cmd, ', copy_cmd)
    os.system(copy_cmd)
    # print(so)
    return so
