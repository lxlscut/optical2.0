import copy

import numpy as np
import cv2

from mesh.bilinear_interpolation import bin_interpolate


def get_line_point(src_warp_gray,mesh_boxes_src,src_warp=None):
    #todo detect lines in source_warp image
    # todo detected the 2 endpoint of the image
    fld = cv2.ximgproc.createFastLineDetector()
    # Get line vectors from the image
    lines = fld.detect(src_warp_gray)
    lines = np.squeeze(lines)
    start = lines[:,:2]
    end = lines[:,2:]
    length = end-start
    length = np.linalg.norm(length,axis=1,ord=2,keepdims=True)
    length  = np.squeeze(length)
    line_index = np.where(length>15)
    line_index = line_index[0]
    qualified_line = lines[line_index,:]
    qualified_line = qualified_line.astype(np.int)
    qualified_length = length[line_index]
    qualified_length_divide_num = (qualified_length/3).astype(int)

    #

    if src_warp is not None:
        src_warp_show_line = np.copy(src_warp)
        for i in range(qualified_line.shape[0]):
            start = qualified_line[i, :2]
            end = qualified_line[i, 2:4]
            cv2.line(src_warp_show_line, (start[0], start[1]), (end[0], end[1]), (0, 255, 0), thickness=1, lineType=8)
        cv2.imshow("src_with_qualified_line", src_warp_show_line)
    # todo sample to all of those lines
    sample_lines = []
    for i in range(qualified_line.shape[0]):
        u = qualified_length_divide_num[i]
        sample_line = []
        line = qualified_line[i]
        start = line[:2]
        end = line[2:]
        li = end-start
        for i in range(0,u+1):
            v = i/u*li+start
            sample_line.append(v)
        sample_line = np.array(sample_line)
        sample_lines.append(sample_line)

    # 绘制出直线上的采样点
    if src_warp is not None:
        src_warp_copy_point = np.copy(src_warp)
        for i_num in range(len(sample_lines)):
            for line_point in range(sample_lines[i_num].shape[0]):
                pts = sample_lines[i_num][line_point, :]
                cv2.circle(src_warp_copy_point, (int(pts[0]), int(pts[1])), 2, (0, 0, 255), thickness=2, lineType=8)
        cv2.imshow("src_warp", src_warp_copy_point)

    weight_lines = []
    location_lines = []
    for i in sample_lines:
        weight_line,location_line = bin_interpolate(i,mesh_boxes_src)
        weight_line = np.squeeze(weight_line)
        location_line = np.squeeze(location_line)
        weight_lines.append(weight_line)
        location_lines.append(location_line)
    return weight_lines,location_lines