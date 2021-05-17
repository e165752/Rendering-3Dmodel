import os
import bpy
import random
import numpy as np
import transforms3d.euler as txq
import cv2

out_dir = '/Users/komineakina/master/research/taking_pictures_3Dmodel/render_images'
camera = bpy.data.objects['Camera'] # Camera object
target = bpy.data.objects['Target'] # Target object such as empty object
model = bpy.data.objects['node']

bb_min = [-1.4, -1.2, -0.08]    #lower limit of camera position
bb_max = [0.15, 0.3, 0.6] #higher limit of camera position

distance_limit = 0.2  #the camera is not located near than this distance

target_3d_location = np.array(target.location)
M = np.append(target_3d_location, 1.0)
print('M={}'.format(M))

#print('f={}'.format(bpy.types.MovieTrackingCamera['Camera'].focal_length))
f=29.0 # per-pixel
cx = bpy.context.scene.render.resolution_x
cy = bpy.context.scene.render.resolution_y
A = np.array([[f, 0, cx],[0, f, cy],[0, 0, 1]])
print('A={}'.format(A))

camera.constraints.clear()
track_to = camera.constraints.new('TRACK_TO')
track_to.target = target
track_to.up_axis = 'UP_Y'
track_to.track_axis = 'TRACK_NEGATIVE_Z'

# matrix_world of Model  XYZ-euler: (98.4, -21.8, 0)
model_matrix_world = model.matrix_world
model_matrix_world = np.array(model_matrix_world)
print('1.model_matrix_world = {}'.format(model_matrix_world))

model_matrix_world_txt_filename = 'model_matrix_world.txt'
model_matrix_world_txt_path = os.path.join(out_dir,model_matrix_world_txt_filename)
np.savetxt(model_matrix_world_txt_path, model_matrix_world, delimiter='    ')

bpy.data.cameras['Camera'].lens = 29.0

def random_position():
    while True:
        pos = np.array([random.uniform(bb_min[i], bb_max[i]) for i in range(3) ])
        if np.linalg.norm(pos-target.location) >= distance_limit:
            print(np.linalg.norm(pos-target.location))
            return pos

for i in range(170):
    filename = '1_%d.jpg' % i
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

   # Perspective projection change
    target_2d_location = A @ camera_pose[:3] @ M
    target_2d_location /= target_2d_location[-1]
    print('target_2d_location = {}'.format(target_2d_location))
    
    
    #Kasika(check)
    #print(rendering_image)
    #cv2.drawMarker(rendering_image, (337, 50), (255, 0, 255), markerType=cv2.MARKER_SQUARE, markerSize=15)
#    bpy.data.images['Render Result'].save_render(filepath = path )
#    cv2.imwrite(path, rendering_image)