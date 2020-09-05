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
    src = cv2.imread("image/DSC00318.JPG")
    dst = cv2.imread("image/DSC00319.JPG")
    # cv2.imshow("src",src)
    # cv2.imshow("dst",dst)
    match = Match(src, dst)
    match.getInitialFeaturePairs()
    src_point = match.src_match
    dst_point = match.dst_match

    draw = Draw()
    H,no = cv2.findHomography(src_point,dst_point)
    img_info = Image_info()
    img_info.get_final_size(src,dst,H)

    # TODO stitching tow image by global homography, the fusion methed is feathering
    src_warp = cv2.warpPerspective(src,H,(img_info.width,img_info.height))
    dst_warp = np.zeros_like(src_warp)
    dst_warp[img_info.offset_y:dst.shape[0]+img_info.offset_y,img_info.offset_x:dst.shape[1]+img_info.offset_x,:] = dst[:,:,:]
    result,mask = blending_average(src_warp,dst_warp)
    # cv2.imshow("mask",mask*255)
    #TODO optimize the result by optical flow


    for i in range(5):
        #todo STEP1 todo first generate rugular mesh box
        mesh_boxes_src,src_x_num,src_dst_num = get_mesh_boxes(src_warp)
        mesh_boxes_dst,dst_x_num,dst_y_num = get_mesh_boxes(dst_warp)
        #todo STEP2 sample pixels from the img, according to the paper we sample the pixel with a interval of 3 pixel on both vertical and herizontal
        sample_vertices = get_sample_point(src_warp).reshape(-1,2)
        #todo get the sample pixel points in overlap ragion
        sample_vertices_or = []
        for sv in range(sample_vertices.shape[0]):
            if mask[sample_vertices[sv,1],sample_vertices[sv,0]]>0:
                sample_vertices_or.append(sample_vertices[sv])
        sample_vertices_or = np.array(sample_vertices_or,dtype=np.int)
        sample_vertices_or_pic = draw.draw(src_warp,sample_vertices_or)
        cv2.imshow("sample_vertices_or_pic",sample_vertices_or_pic)

        #todo Bilinear  interpolate the sample point in overlap ragion by the vertices, here we set dst_warp as the target image, src_warp as reference image
        weight,location = bin_interpolate(sample_vertices_or,mesh_boxes_src)
        weight = np.squeeze(weight)

        #todo we transfrom the image to gray the use the value of pixel as intensity
        src_warp_gray = cv2.cvtColor(src_warp,cv2.COLOR_BGR2GRAY)
        dst_warp_gray = cv2.cvtColor(dst_warp,cv2.COLOR_BGR2GRAY)
        # todo 对dst_warp变换而原图保持不变
        #################################################################
        ##########Ec(τ(q))=||tar(q)+▽Itar(q)τ(q)−Iref(q)||^2############
        ################################################################
        #todo processs the first condition
        #todo for every match sample point
        #first calculate the gradient of every point
        grad = calclulate_gradient(sample_vertices_or,src_warp_gray)
        #second calculate every b
        bs = []
        for i in range(sample_vertices_or.shape[0]):
            # intensity difference
            b1 = int(dst_warp_gray[sample_vertices_or[i,1],sample_vertices_or[i,0]])-int(src_warp_gray[sample_vertices_or[i,1],sample_vertices_or[i,0]])
            b2 = grad[i,0]*sample_vertices_or[i,0]+grad[i,1]*sample_vertices_or[i,1]
            b = b2-b1/2
            bs.append(b)
        bs = np.array(bs,dtype=np.float32)
        # thrid calculate every cofficient
        cofficients = []
        for j in range(sample_vertices_or.shape[0]):
            for k in range(weight[j].shape[0]):
                cofficient = [weight[j,k]*grad[j,0],weight[j,k]*grad[j,1]]
                cofficients.append(cofficient)
        cofficients = np.array(cofficients,dtype=np.float32)
        cofficients = np.squeeze(cofficients)
        cofficients = cofficients.reshape(-1,4,2)
        #TODO process second condition(constrain)
        # to every point not in overlap ragion,we let them have a similarity transform
        triangles = generate_triangle(vertices=mesh_boxes_src)
        triangle_coefficient = get_triangle_coefficient(vertices=mesh_boxes_src,triangle=triangles)
        # TODO process the thrid constrain,the gradient constrain
        src_warp_gradient = calculate_gradient_graph(src_warp_gray)
        dst_warp_gradient = calculate_gradient_graph(dst_warp_gray)
        grad2 = gradient_gradient(sample_vertices_or,src_warp_gradient)
        ###########################################################################
        ############ E = ∑||Gt(pi+τ(pi))−Gs(pi)||^2################################
        ###########################################################################
        cofficients_g = []
        bbss = []
        for i in range(sample_vertices_or.shape[0]):
            bb = grad2[i]*sample_vertices_or[i] + src_warp_gradient[sample_vertices_or[i,1],sample_vertices_or[i,0]] - dst_warp_gradient[sample_vertices_or[i,1],sample_vertices_or[i,0]]
            bbss.append(bb)
            for w in weight[i]:
                cofficient_g = [w*grad2[i,0],w*grad2[i,1]]
                cofficients_g.append(cofficient_g)
        cofficients_g = np.array(cofficients_g)
        cofficients_g = cofficients_g.reshape(-1,4,2)
        bbss = np.array(bbss)

        # todo add strong constrains to the edge of the image,cause we found that the edge often will had serious deformation
        print("mesh_boxes_src.shape"+str(mesh_boxes_src.shape))
        # todo get the vertices on the edge
        c = optimization.optimize(triangles,triangle_coefficient,cofficients,location,bs,mesh_boxes_src,cofficients_g,bbss,0.16)
        c = c.astype(np.int)
        c = c.reshape(dst_y_num,dst_x_num,2)

        #todo dst_warp do transform while src_warp not
        # 怎么做？？？
        # todo 对dst_warp变换而原图保持不变
        #################################################################
        ##########Ec(τ(q))=||tar(q)+▽Itar(q)τ(q)−Iref(q)||^2############
        ################################################################
        #todo processs the first condition
        #todo for every match sample point
        #first calculate the gradient of every point
        grad = calclulate_gradient(sample_vertices_or,dst_warp_gray)
        #second calculate every b
        bs = []
        for i in range(sample_vertices_or.shape[0]):
            # intensity difference
            b1 = int(src_warp_gray[sample_vertices_or[i,1],sample_vertices_or[i,0]])-int(dst_warp_gray[sample_vertices_or[i,1],sample_vertices_or[i,0]])
            b2 = grad[i,0]*sample_vertices_or[i,0]+grad[i,1]*sample_vertices_or[i,1]
            b = b2-b1/2
            bs.append(b)
        bs = np.array(bs,dtype=np.float32)
        # thrid calculate every cofficient
        cofficients = []
        for j in range(sample_vertices_or.shape[0]):
            for k in range(weight[j].shape[0]):
                cofficient = [weight[j,k]*grad[j,0],weight[j,k]*grad[j,1]]
                cofficients.append(cofficient)
        cofficients = np.array(cofficients,dtype=np.float32)
        cofficients = np.squeeze(cofficients)
        cofficients = cofficients.reshape(-1,4,2)
        #TODO process second condition(constrain)
        # to every point not in overlap ragion,we let them have a similarity transform
        triangles = generate_triangle(vertices=mesh_boxes_dst)
        triangle_coefficient = get_triangle_coefficient(vertices=mesh_boxes_dst,triangle=triangles)
        # TODO process the thrid constrain,the gradient constrain
        src_warp_gradient = calculate_gradient_graph(src_warp_gray)
        dst_warp_gradient = calculate_gradient_graph(dst_warp_gray)
        grad2 = gradient_gradient(sample_vertices_or,dst_warp_gradient)
        ###########################################################################
        ############ E = ∑||Gt(pi+τ(pi))−Gs(pi)||^2################################
        ###########################################################################
        cofficients_g = []
        bbss = []
        for i in range(sample_vertices_or.shape[0]):
            bb = grad2[i]*sample_vertices_or[i] + dst_warp_gradient[sample_vertices_or[i,1],sample_vertices_or[i,0]] - src_warp_gradient[sample_vertices_or[i,1],sample_vertices_or[i,0]]
            bbss.append(bb)
            for w in weight[i]:
                cofficient_g = [w*grad2[i,0],w*grad2[i,1]]
                cofficients_g.append(cofficient_g)
        cofficients_g = np.array(cofficients_g)
        cofficients_g = cofficients_g.reshape(-1,4,2)
        bbss = np.array(bbss)

        # todo add strong constrains to the edge of the image,cause we found that the edge often will had serious deformation
        print("mesh_boxes_src.shape"+str(mesh_boxes_dst.shape))
        # todo get the vertices on the edge
        c_dst = optimization.optimize(triangles,triangle_coefficient,cofficients,location,bs,mesh_boxes_dst,cofficients_g,bbss,0.16)
        c_dst = c.astype(np.int)
        c_dst = c.reshape(dst_y_num,dst_x_num,2)
        """the offset of texture_mapping bring"""
        offset_x = abs(min(np.min(c[:,:,0]),0))
        offset_y = abs(min(np.min(c[:,:,1]),0))
        offset_x_dst = abs(min(np.min(c_dst[:,:,0]),0))
        offset_y_dst = abs(min(np.min(c_dst[:,:,1]),0))
        offset_x = max(offset_x,offset_x_dst)
        offset_y = max(offset_y,offset_y_dst)

        """OFFSET_X"""
        # todo get the image after warping
        final_result = texture_mapping.texture_mapping(mesh_boxes_src.astype(np.int), c.astype(np.int),
                                        src_warp)
        final_result = final_result.astype(np.uint8)
        final_result_dst = texture_mapping.texture_mapping(mesh_boxes_dst.astype(np.int), c_dst.astype(np.int),
                                                       dst_warp)
        final_result_dst = final_result_dst.astype(np.uint8)
        fgg = cv2.cvtColor(final_result,cv2.COLOR_BGR2GRAY)
        frdg = cv2.cvtColor(final_result_dst,cv2.COLOR_BGR2GRAY)
        mask = fgg*frdg
        src_warp = final_result
        dst_warp = final_result_dst
    # cv2.imshow("final_result_src", final_result)
    # cv2.imshow("final_result_dst",final_result_dst)
    resulttt, mast = blending_average(final_result, final_result_dst)

    cv2.imshow("resulttt",resulttt)
    cv2.waitKey(0)