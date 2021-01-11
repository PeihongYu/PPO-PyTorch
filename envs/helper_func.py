import airsim
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync(True).join()

# HUMAN_ID = 'carla_in_place'

# Alternate automatic method

obj_list = client.simListSceneObjects('^(cart|Cart)[\w]*')
assert len(obj_list) == 1 # making sure there is only one human
HUMAN_ID = obj_list[0]

human_pose = client.simGetObjectPose(HUMAN_ID)
human_pose2 = client.simGetObjectPose('carla')
print("hello")

## Get object ID for segmentation


def get_segment_ids(camera_id='0'):
    """
    Function to set the colors for humans and background.
    It would also make the segmentation to show only human (label 1) and rest of the environment would be another segment (label 0).
    This is required because AirSim may change color maps for IDs everytime we launch it

    Parameters
    ----------
        camera_id: Id of teh camera from which we want to capture the scene

    Returns
    -------
        human_seg_id: Color array for humans in segmentation image
        bg_seg_id: Color array for background (everything other than human) in segmentation image
    """

    ############################################################
    ### Background
    # Set everything to ID 0
    client.simSetSegmentationObjectID("[\w]*", 0, True)
    # Get segmentation image
    responses = client.simGetImages([airsim.ImageRequest(camera_id, airsim.ImageType.Segmentation, False, False)])

    # Process reponse into an image
    seg_data = responses[0]
    seg_img = np.frombuffer(seg_data.image_data_uint8, dtype=np.uint8).reshape(seg_data.height, seg_data.width, 3)
    # seg_img_proc = 1000000*seg_img[:,:,0] + 1000*seg_img[:,:,1] + seg_img[:,:,2]

    # Get color for this ID from the image
    bg_seg_id = seg_img[0, 0, :]  # np.unique(seg_img_proc)[0]

    ############################################################
    ### Human
    # Set everything to ID 1
    client.simSetSegmentationObjectID("[\w]*", 1, True)
    # Get segmentation image
    responses = client.simGetImages([airsim.ImageRequest(camera_id, airsim.ImageType.Segmentation, False, False)])

    # Process reponse into an image
    seg_data = responses[0]
    seg_img = np.frombuffer(seg_data.image_data_uint8, dtype=np.uint8).reshape(seg_data.height, seg_data.width, 3)
    # seg_img_proc = 1000000*seg_img[:,:,0] + 1000*seg_img[:,:,1] + seg_img[:,:,2]

    # Get color for this ID from the image
    human_seg_id = seg_img[0, 0, :]  # np.unique(seg_img_proc)[0]

    ## Set everything to ID 0
    client.simSetSegmentationObjectID("[\w]*", 0, True)
    ## Set human to ID 1
    client.simSetSegmentationObjectID(HUMAN_ID, 1, True)

    return human_seg_id, bg_seg_id


## Getting relative location w.r.t vehicle
def get_relloc_vehicle(mode='rot_mat'):
    """
    Function to get position of human relative to the drone

    Parameters
    ----------
        mode:   if 'angle', differece in angles (degrees) is returned
                if 'rot_mat' rotation matrix is returned. C.dot(T) = O

    Returns
    -------
        rel_pos: Relative position (x, y, z). Numpy array
        rel_orient: if mode == angle: Relative orientation (roll, pitch, yaw). Numpy array
                    if mode == rot_mat: Reotation matix is returned
    """

    # Get human's pose
    human_pose = client.simGetObjectPose(HUMAN_ID)
    # Get drone's pose
    drone_pose = client.simGetVehiclePose()

    # Get relative position
    rel_pos = (human_pose.position - drone_pose.position).to_numpy_array()

    if mode == 'angle':
        # Get relative orientation
        human_rpy = np.array(airsim.to_eularian_angles(human_pose.orientation))  # quaternion to Eularian
        drone_rpy = np.array(airsim.to_eularian_angles(drone_pose.orientation))  # quaternion to Eularian
        rel_orient = human_rpy - drone_rpy
    if mode == 'rot_mat':
        # Get rotation matrix
        human_rot = R.from_quat(human_pose.orientation.to_numpy_array()).as_matrix()
        drone_rot = R.from_quat(drone_pose.orientation.to_numpy_array()).as_matrix()

        # Calculate transformation/rotation matrix
        drone_rot_inv = np.linalg.inv(drone_rot)
        rel_orient = drone_rot_inv.dot(human_rot)
        rel_pos = drone_rot_inv.dot(rel_pos)

    return rel_pos, rel_orient


## Getting relative location w.r.t camera
def get_relloc_camera(camera_id='1', mode='rot_mat'):
    """
    Function to get position of human relative to the camera

    Parameters
    ----------
        camera_id: ID of the camera from which we want to observe the environment
        mode:   if 'angle', differece in angles (degrees) is returned
                if 'rot_mat' rotation matrix is returned. C.dot(T) = O

    Returns
    -------
        rel_pos: Relative position (x, y, z). Numpy array
        rel_orient: if mode == angle: Relative orientation (roll, pitch, yaw). Numpy array
                    if mode == rot_mat: Reotation matix is returned
    """
    # Get human's pose
    human_pose = client.simGetObjectPose(HUMAN_ID)
    # Get camera's pose
    camera_pose = client.simGetCameraInfo(camera_id).pose
    drone_pose = client.simGetVehiclePose()

    # Get relative position
    rel_pos = (human_pose.position - camera_pose.position).to_numpy_array()

    if mode == 'angle':
        # Get relative orientation
        human_rpy = np.array(airsim.to_eularian_angles(human_pose.orientation))  # quaternion to Eularian
        camera_rpy = np.array(airsim.to_eularian_angles(camera_pose.orientation))  # quaternion to Eularian
        rel_orient = human_rpy - camera_rpy
    if mode == 'rot_mat':
        # Get rotation matrix
        human_rot = R.from_quat(human_pose.orientation.to_numpy_array()).as_matrix()
        camera_rot = R.from_quat(camera_pose.orientation.to_numpy_array()).as_matrix()

        # Calculate transformation/rotation matrix
        camera_rot_inv = np.linalg.inv(camera_rot)
        rel_orient = camera_rot_inv.dot(human_rot)
        rel_pos = camera_rot_inv.dot(rel_pos)

    com_rot = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    rel_orient = com_rot.dot(rel_orient.dot(com_rot.T))
    rel_pos = com_rot.dot(rel_pos)

    rot = R.from_matrix(rel_orient)
    rel_orient = rot.as_euler('zyx', degrees=True)

    return rel_pos, rel_orient


def get_bounding_box(camera_id, human_seg_id):
    """
    Function to get the bounding box around human in the scene

    Parameters
    ----------
        camera_id: ID of the camera from which we want to observe teh environemnt
        human_seg_id: Color array for human in the segmentation image

    Returns
    -------
        top_left_corner: Top-Left corner of the bounding box. Touple
        width: Width of the bounding box
        height: Height of the bounding box
        scene_img: Scene image

        If human is not in the scene, (None, None), None, None, scene_img is returned

    """

    # Get scene and segmentation images (NOT compressed)
    responses = client.simGetImages([airsim.ImageRequest(camera_id, airsim.ImageType.Scene, False, False),
                                     airsim.ImageRequest(camera_id, airsim.ImageType.Segmentation, False, False)])

    ##############################
    ## Processing scene image
    scene_data = responses[0]
    scene_img = np.frombuffer(scene_data.image_data_uint8, dtype=np.uint8).reshape(scene_data.height, scene_data.width,
                                                                                   3)

    ##############################
    ## Processing segmentation image
    seg_data = responses[1]
    seg_img = np.frombuffer(seg_data.image_data_uint8, dtype=np.uint8).reshape(seg_data.height, seg_data.width, 3)
    # seg_img_proc = 1000000*seg_img[:,:,0] + 1000*seg_img[:,:,1] + seg_img[:,:,2]

    # Find locations where human is located
    rows, cols = np.where((seg_img == human_seg_id).all(axis=2))  # np.where(seg_img_proc == human_seg_id)

    if len(rows) == 0:
        print('Human not found in the scene')
        return None, None, None, None

    else:
        top_row, bottom_row = min(rows), max(rows)
        left_col, right_col = min(cols), max(cols)

        height = bottom_row - top_row
        width = right_col - left_col

        top_left_corner = (left_col, top_row)

        return top_left_corner, width, height, scene_img


######################
## Testing code ######
######################

# client.moveByVelocityAsync(0,0,-1,1).join()
# client.rotateByYawRateAsync(-45,1).join()


camera_id = '0'
# Get sgementation color arrays
human_seg_id, bg_seg_id = get_segment_ids(camera_id)
# Get bounding box paarametrs
top_left_corner, width, height, scene_img = get_bounding_box(camera_id, human_seg_id)

(h, w, c) = scene_img.shape

# Show scene image with bounding box
if width is not None:
    fig, ax = plt.subplots()
    ax.imshow(scene_img)
    ax.add_patch(Rectangle(top_left_corner, width, height, edgecolor='red', facecolor='none', lw=2))
    plt.show()

K = np.array([[w/2, 0, w/2], [0, w/2, h/2], [0, 0, 1]])
pixel = np.array([top_left_corner[0] + width/2, top_left_corner[1] + height, 1])
rel_pos, rel_rot = get_relloc_camera()

print(rel_pos)
print(rel_rot)

img_coord = K.dot(rel_pos)
img_coord = img_coord / img_coord[2]
print(img_coord)
print(pixel)