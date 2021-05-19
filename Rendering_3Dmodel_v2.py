import os
import bpy
import random
import numpy as np
import transforms3d.euler as txq
import cv2

out_dir = '/Users/komineakina/master/research/dataset/hujukan/culture/hujukan_culture_tubo_tana1_TakenManually_10images/3Dmodel_render_images/'
camera = bpy.data.objects['Camera'] # Camera object
target = bpy.data.objects['Pot_mesh'] # Target object such as empty object
bottle = bpy.data.objects['Bottle_mesh']
karakara = bpy.data.objects['Karakara_mesh']
model = bpy.data.objects['node']

#msh = bpy.data.meshes.new(name="cubemesh")
#msh.from_pydata(verts, [], faces)
#msh.update()

bb_min = [0.2, -0.4, -0.1]    #lower limit of camera position
bb_max = [0.7, 0.3, 0.4] #higher limit of camera position

distance_limit = 0.3  #the camera is not located near than this distance

#for vt in msh.vertices:
#    print("vertex index:{0:2} co:{1} normal:{2}".format(vt.index, vt.co, vt.normal))

#target_3d_location = np.array(target.location)
#M = np.append(target_3d_location, 1.0)
#print('M={}'.format(M))

#target.scale[0]

bpy.data.cameras['Camera'].lens = 29.0

camera_sensor_width = bpy.data.cameras['Camera'].sensor_width
camera_sensor_height = bpy.data.cameras['Camera'].sensor_height
focal_length_mm = bpy.data.cameras['Camera'].lens
print(camera_sensor_width, camera_sensor_height)

fx = (bpy.context.scene.render.resolution_x / camera_sensor_width) * focal_length_mm
fy = (bpy.context.scene.render.resolution_y / camera_sensor_height) * focal_length_mm
print('fx, fy = {}, {}'.format(fx, fy))

resolution_x = bpy.context.scene.render.resolution_x
resolution_y = bpy.context.scene.render.resolution_y

cx = bpy.context.scene.render.resolution_x / 2
cy = bpy.context.scene.render.resolution_y / 2
print('cx, cy = {}, {}'.format(cx, cy))

A = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
print('A={}'.format(A))

camera.constraints.clear()
track_to = camera.constraints.new('TRACK_TO')
track_to.target = target
track_to.up_axis = 'UP_Y'
track_to.track_axis = 'TRACK_NEGATIVE_Z'

target_matrix_world = target.matrix_world
model_matrix_world = model.matrix_world
model_matrix_world = np.array(model_matrix_world)
print('1.model_matrix_world = {}'.format(model_matrix_world))

model_matrix_world_txt_filename = 'model_matrix_world.txt'
model_matrix_world_txt_path = os.path.join(out_dir,model_matrix_world_txt_filename)
np.savetxt(model_matrix_world_txt_path, model_matrix_world, delimiter='    ')

bpy.data.cameras['Camera'].lens = 29.0

annotations_path = os.path.join(out_dir,'annotations.txt')

with open(annotations_path, 'w') as f:
    f.write('')

def random_position():
    while True:
        pos = np.array([random.uniform(bb_min[i], bb_max[i]) for i in range(3) ])
        if np.linalg.norm(pos-target.location) >= distance_limit:
            print(np.linalg.norm(pos-target.location))
            return pos
        
def perspective_projection_change(M):
    '''
    #original:sayuugyaku    
    M_com = np.linalg.inv(camera_pose[:])[:3] @ M
    target_2d_location = A @ M_com
    target_2d_location /= target_2d_location[-1]
    print('target_2d_location = {}'.format(target_2d_location))
    '''
    
    M_com = np.linalg.inv(camera_pose[:])[:3] @ M
    target_2d_location = A @ M_com
    target_2d_location /= target_2d_location[-1]

    target_2d_location[0] = bpy.context.scene.render.resolution_x - target_2d_location[0]
    print('target_2d_location = {}'.format(target_2d_location))    
    return target_2d_location[:2]


def bbox_cal(p1, p2, p3, p4): 
#    np.where(p1[0], p2[0], p3[0], p4[0] > resolution_x, resolution_x, p1[0], p2[0], p3[0], p4[0])
    
    x_min = min(p1[0], p2[0], p3[0], p4[0])
    y_min = min(p1[1], p2[1], p3[1], p4[1])
    x_max = max(p1[0], p2[0], p3[0], p4[0])
    y_max = max(p1[1], p2[1], p3[1], p4[1])
    return [x_min,y_min,x_max,y_max]
    

for i in range(15):
    filename = '%d.jpg' % i
    path = os.path.join(out_dir,filename)
    print("rendering and writing ", path)
        
    camera.location = random_position()
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    bpy.ops.render.render()
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.data.images['Render Result'].save_render(filepath = path )
#    rendering_image = bpy.data.images['Render Result'].pixels[:]
    
#    camera_rotation_mat = txq.euler2mat(camera.rotation_euler[0],camera.rotation_euler[1],camera.rotation_euler[2])
#    print('camera_rotation_mat = {}'.format(camera_rotation_mat))
    
    camera_Tvec = np.array([camera.location]).T
    print('camera_Tvec = {}'.format(camera_Tvec))
    
#    camera_pose = np.hstack((camera_rotation_mat, camera_Tvec))  
#    camera_pose = np.vstack((camera_pose, [0, 0, 0, 1]))

    camera_rotation_mat = camera.matrix_world    
    camera_pose = np.array(camera_rotation_mat)
    print('2.camera_pose = {}'.format(camera_pose[:3]))
    
    external_param = camera_pose
    
#    external_param = np.vstack((external_param, [0, 0, 0, 1]))
    print('external_param = {}'.format(external_param))
    
    pose_txt_filename = '%d.pose.txt' % i
    pose_txt_path = os.path.join(out_dir,pose_txt_filename)
#    np.savetxt(pose_txt_path, external_param, delimiter='    ')

    img = cv2.imread(path,1)

    for class_id, obj in enumerate([target, bottle, karakara]):
        target_3d_location = np.array(obj.location)
        M = np.append(target_3d_location, 1.0)
        print('M={}'.format(M))
        
        # active
#        bpy.context.view_layer.objects.active = obj
        
#        for n in range(len(bpy.context.active_object.data.vertices)):
#            print('{} vertex: {}'.format(obj, bpy.context.active_object.data.vertices[n].co))

       # Perspective projection change
        rad_x, rad_y, rad_z = (obj.empty_display_size*obj.scale)
        print('rad = {},{},{}'.format(rad_x, rad_y, rad_z))
        M1 = np.array([M[0]-rad_x, M[1]-rad_y, M[2]+rad_z, 1.0])
        M2 = np.array([M[0]+rad_x, M[1]+rad_y, M[2]-rad_z, 1.0])
        M3 = np.array([M[0]-rad_x, M[1]+rad_y, M[2]+rad_z, 1.0])
        M4 = np.array([M[0]+rad_x, M[1]-rad_y, M[2]-rad_z, 1.0])
        
        target_2d_location_center = perspective_projection_change(M)
        target_2d_location_p1 = perspective_projection_change(M1)
        target_2d_location_p2 = perspective_projection_change(M2)
        target_2d_location_p3 = perspective_projection_change(M3)
        target_2d_location_p4 = perspective_projection_change(M4)
        
        x_min,y_min,x_max,y_max = bbox_cal(target_2d_location_p1, target_2d_location_p2, target_2d_location_p3, target_2d_location_p4)
        
        print('M = {}'.format(target_2d_location_center))
        print('M1 = {}'.format(target_2d_location_p1))
        print('M2 = {}'.format(target_2d_location_p2))
        print('M3 = {}'.format(target_2d_location_p3))
        print('M4 = {}'.format(target_2d_location_p4))
        
        x_min,y_min,x_max,y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        
        with open(annotations_path, 'a') as f:    
            annotation = '{},{},{},{},{} '.format(x_min, y_min, x_max, y_max, class_id)
            if class_id == 0:
                annotation = path + ' ' + annotation
            elif class_id == 2:
                annotation = annotation.replace(' ', '') + '\n'
            f.write(annotation)
    
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255,255,0), 2)
        
    cv2.imwrite(path,img)