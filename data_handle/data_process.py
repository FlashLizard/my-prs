import open3d as o3d
import os
import sys
import argparse
import numpy as np
import queue
import regex as re
import subprocess

# 设置 Open3D 的日志级别为 WARNING，这样会忽略 INFO 和 DEBUG 级别的日志
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

def parse_arg():
    parse = argparse.ArgumentParser(description='data-process')
    parse.add_argument('--show', type=bool, help='show after handle', default=False)
    parse.add_argument('--input_path', type=str, help='input path', default='obj-data')
    parse.add_argument('--output_path', type=str, help='output path', default='results')
    parse.add_argument('--sample_points_number', type=int, help='sample points number', default=1000)
    parse.add_argument('--voxel_resolution', type=int, help='voxel resolution', default=32)
    parse.add_argument('--model_per_file', type=int, help='number of model in a output file', default=128)
    parse.add_argument('--max_in_folder', type=int, help='max number in a folder', default=256)
    parse.add_argument('--start_id', type = int, help='start id of output file', default=0)
    parse.add_argument('--auto_rerun', type=bool, help='auto rerun', default=False)
    args = parse.parse_args()
    return args

def get_closet_points(points, mesh):
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene.compute_closest_points(points)['points']
print("??")
if __name__ == '__main__':
    for arg in sys.argv:
        print(arg)
    args = parse_arg()
    input_path = args.input_path
    output_path = args.output_path
    sample_points_number = args.sample_points_number
    should_show = args.show
    voxel_resolution = args.voxel_resolution
    file_size = args.model_per_file
    max_in_folder = args.max_in_folder
    start_id = args.start_id
    auto_rerun = args.auto_rerun
    tmp_file = os.path.join(output_path, 'tmp.txt')
    
    print('****',args)
    if(auto_rerun):
        while True:
            files = os.listdir(output_path)
            for file in files:
                result = re.findall(r'output\d+', file)
                if(len(result)>0):
                    id = int(result[0][6:])+1
                    start_id = max(start_id, id)
            print('start_id is set to', start_id)
            with(open(tmp_file, 'r')) as f:
                strs = f.readlines()
                if(len(strs)>0):
                    fail_file = strs[-1]
                    print(f'fail_file is {fail_file}, remove it')
                    try:
                        os.remove(fail_file)
                    except Exception as e:
                        print(f'fail to remove {fail_file}, because {e}')
            script_path = 'data-process.py'

            new_args = ['--start_id', f'{start_id}', '--input_path',
                        input_path, '--output_path', output_path,
                        '--sample_points_number',
                        f'{sample_points_number}',
                        '--voxel_resolution',
                        f'{voxel_resolution}',
                        '--model_per_file',
                        f'{file_size}',
                        '--max_in_folder',
                        f'{max_in_folder}']
            print('begin to run subprocess')
            result = subprocess.run(['python', script_path] + new_args, capture_output=True, text=True)
            print('result:',result.stdout)
    else:
        with(open(os.path.join(output_path,'log.txt'),'+a')) as f:
            f.write(str(args))
    
    
    regular_points = np.zeros((voxel_resolution,voxel_resolution,voxel_resolution,3), dtype=np.float32)
    for i in range(voxel_resolution):
        for j in range(voxel_resolution):
            for k in range(voxel_resolution):
                regular_points[i][j][k] = [i, j, k]
    regular_points = (regular_points+0.5)/voxel_resolution - 0.5
    if(should_show):
        print('regular_points',regular_points)
    
    models = []
    sps = []
    cps = []
    cnt = start_id
    already = start_id*file_size
    files = os.listdir(input_path)
    que = queue.Queue()
    for file in files:
        que.put(os.path.join(input_path, file))
    
    while not que.empty():
        file_path = que.get()
        if os.path.isdir(file_path):
            dir_files = os.listdir(file_path)
            obj_num = 0
            for dir_file in dir_files:
                obj_num += 1
                if(obj_num>max_in_folder):
                    break
                que.put(os.path.join(file_path, dir_file))
                if should_show:
                    print(f"Add {os.path.join(file_path, dir_file)} to queue.")

        elif file_path.endswith(".obj"):
            if(already>0):
                already -= 1
                continue
            with(open(tmp_file, 'w')) as f:
                f.write(file_path)
            print(f"Processing {file_path}...")
            
            try:
                mesh = o3d.io.read_triangle_mesh(file_path)
                angle = np.random.rand(3)*np.pi
                # 旋转变换
                R = mesh.get_rotation_matrix_from_xyz(angle)  # 45度绕x, y, z轴旋转
                mesh.rotate(R, center=(0, 0, 0))

                # # 平移变换
                # T = np.eye(4)
                # T[:3, 3] = [1, 2, 3]  # 平移向量 (1, 2, 3)
                # mesh.transform(T)

                # # 缩放变换
                # mesh.scale(0.5, center=mesh.get_center())  # 缩放因子 0.5，以网格中心为缩放中心
                
                # Sample points
                sample_points = mesh.sample_points_uniformly(number_of_points=sample_points_number)
                sample_points_arr = np.asarray(sample_points.points)
                # Generate Voxel
                voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=1/voxel_resolution)
                voxel_grid_arr = np.stack(list(vx.grid_index for vx in voxel_grid.get_voxels()))
                model = np.zeros((voxel_resolution,voxel_resolution,voxel_resolution))
                for v in voxel_grid_arr:
                    model[v[0]][v[1]][v[2]] = 1
                # ??
                # voxel_grid = (voxel_grid + 0.5)/voxel_resolution - 0.5
                # Get Closet Points
                closest_points = get_closet_points(regular_points, mesh).numpy()
                
                
                if(should_show):
                    print('sample_points',sample_points_arr)
                    o3d.visualization.draw_geometries([sample_points])
                    
                    print('voxel_gird',voxel_grid_arr)
                    o3d.visualization.draw_geometries([voxel_grid])
                    
                    flat = closest_points.reshape((-1,3))
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(flat)
                    print('closest_points',closest_points)
                    o3d.visualization.draw_geometries([pcd])
                    
                    should_show = (input("Show next? (y/n)") != 'n')
                
                # Save
                models.append(model)
                sps.append(sample_points_arr)
                cps.append(closest_points)
                if(len(models)==file_size):
                    np.savez(os.path.join(output_path, f'output{cnt}'), sample_points=sps, voxel_grid=models, closest_points=cps)
                    models = []
                    sps = []
                    cps = []
                    cnt += 1
                    print(f"Save output{cnt} done.")
            except Exception as e:
                print(f"Processing {file_path} error: {e}")
                continue
            print(f"Processing {file_path} done.")

    if(len(models)>0):
        np.savez(os.path.join(output_path, f'output{cnt}'), sample_points=sps, voxel_grid=models, closest_points=cps)
        print(f"Save output{cnt} done.")
    