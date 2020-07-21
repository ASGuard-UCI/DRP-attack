import cv2
import numpy as np

UV_WIDTH = 160
UV_HEIGHT = 80
TILE_SIZE = UV_WIDTH*UV_HEIGHT

# Get perspective transformation matrices
src = np.float32([[0, UV_HEIGHT-1], [319//2, UV_HEIGHT-1], [0, 0], [UV_WIDTH-1, 0]])
dst = np.float32([[150//2, UV_HEIGHT-1], [184//2, UV_HEIGHT-1], [0, 0], [UV_WIDTH-1, 0]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)


def mps_to_mph(mps):
    return 2.23694 * mps


def mph_to_mps(mph):
    return mph / 2.23694


def clip_coord(coord):
    coord[1] = max(0, coord[1])
    coord[1] = min(UV_HEIGHT-1, coord[1])
    coord[0] = max(0, coord[0])
    coord[0] = min(UV_WIDTH-1, coord[0])
    return coord


def clip_coords(coords):
    clipped_coords = np.zeros(coords.shape)
    for i, c in enumerate(coords):
        clipped_coords[i] = clip_coord(c)
    return clipped_coords


def warp_coord(M, coord):
    x = (M[0,0]*coord[0] + M[0,1]*coord[1] + M[0,2])/(M[2,0]*coord[0] + M[2,1]*coord[1] + M[2,2])
    y = (M[1,0]*coord[0] + M[1,1]*coord[1] + M[1,2])/(M[2,0]*coord[0] + M[2,1]*coord[1] + M[2,2])
    warped_coord = np.array([x, y])
    return warped_coord


def warp_coords(M, coords):
    warped_coords = np.zeros((coords.shape[0], 2))
    for i, c in enumerate(coords):
        warped_coords[i] = warp_coord(M, c)
    return warped_coords


def warp_corners(centroid, width, height):
    h_lo = max(0, centroid[1]-height//2)
    h_hi = min(UV_HEIGHT, centroid[1]+height//2)
    w_lo = max(0, centroid[0]-width//2)
    w_hi = min(UV_WIDTH, centroid[0]+width//2)
    warped_corners = np.array([[w_lo, h_lo],
                               [w_hi, h_lo],
                               [w_hi, h_hi],
                               [w_lo, h_hi]])
    corners = warp_coords(Minv, warped_corners)
    return corners


def warp_mask(mask):
    warped_mask = cv2.warpPerspective(mask, Minv, (UV_WIDTH, UV_HEIGHT))
    return warped_mask


def centroids_to_birdeye_masks(centroids, patch_height, patch_width):
    warped_centroids = warp_coords(M, centroids)
    masks = np.zeros((UV_HEIGHT,UV_WIDTH,3))
    for i, c in enumerate(warped_centroids):
        h_lo = max(0, c[1]-patch_height//2)
        h_hi = min(UV_HEIGHT-1, c[1]+patch_height//2)
        w_lo = max(0, c[0]-patch_width//2)
        w_hi = min(UV_WIDTH-1, c[0]+patch_width//2)
        # NOTE: in OpenCV coordinate convention is [column, row]
        warped_corners = np.array([[w_lo, h_lo],
                                   [w_hi, h_lo],
                                   [w_hi, h_hi],
                                   [w_lo, h_hi]])
        masks += cv2.fillPoly(np.zeros((UV_HEIGHT, UV_WIDTH,3), dtype=np.uint8),
                            np.array([warped_corners.astype(np.int32)]),
                            (255,255,255))
    masks = np.clip(masks, 0, 1)
    return masks


def centroids_to_masks(centroids, patch_height, patch_width):
    warped_centroids = warp_coords(M, centroids)
    # Get the mask area in the warped frame
    # Corners are ordered clockwise starting from top-left corner
    masks = np.zeros((UV_HEIGHT,UV_WIDTH,3))
    for i, c in enumerate(warped_centroids):
        h_lo = max(0, c[1]-patch_height//2)
        h_hi = min(UV_HEIGHT-1, c[1]+patch_height//2)
        w_lo = max(0, c[0]-patch_width//2)
        w_hi = min(UV_WIDTH-1, c[0]+patch_width//2)
        # NOTE: in OpenCV coordinate convention: [column, row]
        warped_corners = np.array([[w_lo, h_lo],
                                   [w_hi, h_lo],
                                   [w_hi, h_hi],
                                   [w_lo, h_hi]])
        corners = warp_coords(Minv, warped_corners)
        masks += cv2.fillPoly(np.zeros((UV_HEIGHT, UV_WIDTH,3), dtype=np.uint8),
                            np.array([corners.astype(np.int32)]),
                            (255,255,255))
    # Handle patch overlapping
    masks = np.clip(masks, 0, 1)
    return masks


def lane_rect_masks(num_frames, mph=65, fps=20, start_pix=50, width_scale=1.0):
    DASH_LENGTH = 7  # 7 pixels in BEV => 3 m
    MARK_LENGTH = 6  # 6 pixels in BEV => 2.5 m
    LANE_WIDTH = 27  # 27 pixels in BEV => 3.6 m
    LEFT_LANE = 73
    RIGHT_LANE = 99
    MID = (LEFT_LANE + RIGHT_LANE)//2

    mps = mph_to_mps(mph)
    dist_per_frame = mps * 1.0/fps
    pix_dist_per_frame = int(dist_per_frame * DASH_LENGTH / 3)

    rect_width = int(LANE_WIDTH*width_scale)
    rect_height = DASH_LENGTH
    
    cam_masks = np.zeros((num_frames, UV_HEIGHT, UV_WIDTH, 3))
    cam_corners = np.zeros((num_frames, 4, 2))
    for i in range(num_frames):
        mask = np.zeros((UV_HEIGHT,UV_WIDTH,3), dtype=np.uint8)
        # For reference only
        pt1 = warp_coord(Minv, (LEFT_LANE,0))
        pt2 = warp_coord(Minv, (LEFT_LANE,UV_HEIGHT-1))
        cv2.line(mask, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), (255,255,255), 1)
        pt1 = warp_coord(Minv, (RIGHT_LANE,0))
        pt2 = warp_coord(Minv, (RIGHT_LANE,UV_HEIGHT-1))
        cv2.line(mask, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), (255,255,255), 1)
        
        h = start_pix + pix_dist_per_frame * i
        warped_corners = warp_corners((MID, h), rect_width, rect_height)
        cv2.fillPoly(mask, np.array([warped_corners.astype(np.int32)]), (255,255,255))
        cam_masks[i] = mask
        warped_corners = clip_coords(warped_corners)  # comment this out if you want the non-clipped corners
        cam_corners[i] = warped_corners
    cam_masks = np.clip(cam_masks, 0, 1)
    return cam_masks, cam_corners


def test():
    # Example usage
    centroids = np.array([[20,70],[40,130]])
    patch_height = 5
    patch_width = 5
    # Get the camera frame mask
    cam_masks = centroids_to_masks(centroids, patch_height, patch_width)
    # Get the bird-eye view mask (not required)
    bev_masks = centroids_to_birdeye_masks(centroids, patch_height, patch_width)
    return cam_masks, bev_masks
