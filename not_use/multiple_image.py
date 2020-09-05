import cv2
import numpy as np
from feature.match import Match
from mesh.bilinear_interpolation import bin_interpolate
from mesh.get_triangle import generate_triangle
from mesh.mesh import get_mesh_boxes, get_sample_point
from util.blending_average import blending_average
from util.draw import Draw
from util.gradient import calclulate_gradient, calculate_gradient_graph, gradient_gradient
from util.image_info import Image_info
import optimization
import texture_mapping

from util.triangle_similar import get_triangle_coefficient

if __name__ == '__main__':
    img1 = cv2.imread("image/DSC00319.JPG")
    img2 = cv2.imread("image/DSC00318.JPG")
    img3 = cv2.imread("image/DSC00317.JPG")
    cv2.imshow("img1",img1)
    # 求img1，img2的匹配信息
    match_1_2 = Match(img1,img2)
    match_1_2.getInitialFeaturePairs()
    img1_point = match_1_2.src_match
    img2_point = match_1_2.dst_match

    H_1_2,no = cv2.findHomography(img2_point,img1_point)

    # 求img2,img3的匹配信息
    match_2_3 = Match(img2,img3)
    match_2_3.getInitialFeaturePairs()
    img2_point = match_2_3.src_match
    img3_point = match_2_3.dst_match
    H_2_3,no = cv2.findHomography(img3_point,img2_point)

    H_1_3 = H_1_2.dot(H_2_3)


    # 获取画布的尺寸
    img_info = Image_info()
    img_info.get_final_size(img1,img2,H_1_2)
    #
    img_info1 = Image_info()
    img_info1.get_final_size(img1,img3,H_1_3)

    # 获取最终的画布尺寸
    a = [img_info.left_top,img_info.left_button,img_info.right_top,img_info.right_button,img_info1.left_top,img_info1.left_button,img_info1.right_top,img_info1.right_button]
    a = np.array(a)
    x_min  = np.min(a[:,0])
    y_min = np.min(a[:,1])
    x_max = np.max(a[:,0])
    y_max = np.max(a[:,1])

    offset_x = int(abs(min(x_min,0)))
    offset_y = int(abs(min(y_min,0)))
    height = (y_max-offset_y).astype(int)
    width = (x_max-offset_x).astype(int)

    # 全局矩阵投影结果，全部投影到第一张图上
    img2_transform = cv2.warpPerspective(img2,H_1_2,(width,height))
    img3_transform = cv2.warpPerspective(img3,H_1_3,(width,height))
    dst_warp = np.zeros_like(img3_transform)
    dst_warp[offset_y:img1.shape[0] + offset_y, offset_x:img1.shape[1] + offset_x,:] = img1[:, :, :]
    result0, mask0 = blending_average(img2_transform, dst_warp)
    result1, mask1 = blending_average(img3_transform, result0)
    cv2.imshow("img2_transform",img2_transform)
    cv2.imshow("img3_transform",img3_transform)
    cv2.imshow("result1",result1)
    cv2.imshow("mask0",mask0.astype(np.uint8)*255)
    cv2.imshow("mask1",mask1.astype(np.uint8)*255)

    # 光流优化
    # 方框坐标，横向与竖向的数量
    mesh_boxes_src,src_x_num,src_dst_num = get_mesh_boxes(img2_transform)
    # 获取抽样点
    sample_vertices = get_sample_point(img2_transform).reshape(-1, 2)

    #todo 获取在重叠区域的抽样点
    sample_vertices_or = []
    for sv in range(sample_vertices.shape[0]):
        if mask0[sample_vertices[sv,1],sample_vertices[sv,0]]>0:
            sample_vertices_or.append(sample_vertices[sv])
    sample_vertices_or = np.array(sample_vertices_or,dtype=np.int)
    cv2.waitKey(0)

