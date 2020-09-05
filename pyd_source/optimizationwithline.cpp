// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <iostream>
#include <vl/generic.h>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Sparse>
#include <pybind11/stl.h>


using namespace std;
using namespace Eigen;


pybind11::array_t<uint> optimize(pybind11::array_t<int> &triangle, pybind11::array_t<float> &triangle_cofficient,
	pybind11::array_t<float> &weight, pybind11::array_t<float> &location, pybind11::array_t<float> &b, pybind11::array_t<int> &vertices,
	pybind11::array_t<float> &cofficients_g, pybind11::array_t<float> &bbss, float &lambda,float &beta, float &gama, const vector<pybind11::array_t<float>> &line_weight,
	const vector<pybind11::array_t<int>> &line_location)
{
	auto r1 = triangle.unchecked<3>();
	auto r2 = triangle_cofficient.unchecked<2>();
	auto r3 = weight.unchecked<3>();
	auto r4 = location.unchecked<2>();
	auto r5 = b.unchecked<1>();
	auto r6 = vertices.unchecked<3>();
	auto r8 = cofficients_g.unchecked<3>();
	auto r9 = bbss.unchecked<2>();

	pybind11::array_t<float> result = pybind11::array_t<float>(vertices.shape()[0] * vertices.shape()[1] * 2);
	auto r7 = result.mutable_unchecked<1>();
	//create a sparse matrix
	vector<Eigen::Triplet<float>> m;

	int width = vertices.shape()[1];
	int height = vertices.shape()[0];
	//the total constrains is triangle numbers add weight numbers
	int line_constrain_num = 0;
	for (int i = 0; i < line_location.size(); i++) {
		line_constrain_num += line_location[i].shape()[0];
	}
	int constrain_num = vertices.shape()[0] * vertices.shape()[1] * 2 + weight.shape()[0] * 3 + line_constrain_num*2 + triangle.shape()[0] * 2;

	//reserve memory
	m.reserve(constrain_num);
	//insert the constrain value
	int index = 0;

	//1.Prevent the no overlap area and without lines detected have too much deformation
	for (int i = 0; i < vertices.shape()[0]; i++) {
		for (int j = 0; j < vertices.shape()[1]; j++)
			for (int dim = 0; dim < 2; dim++) {
				m.emplace_back(index, (i*width + j) * 2 + dim, gama);
				index++;
			}
	}

	//2.insert the optical point constrains
	for (int j = 0; j < weight.shape()[0]; j++)
	{
		for (int dim = 0; dim < 2; dim++)
		{
			m.emplace_back(index, (r4(j, 0)*width + r4(j, 1)) * 2 + dim, r3(j, 0, dim));
			m.emplace_back(index, (r4(j, 0)*width + r4(j, 1) + 1) * 2 + dim, r3(j, 1, dim));
			m.emplace_back(index, ((r4(j, 0) + 1)*width + r4(j, 1) + 1) * 2 + dim, r3(j, 2, dim));
			m.emplace_back(index, ((r4(j, 0) + 1)*width + r4(j, 1)) * 2 + dim, r3(j, 3, dim));
		}
		index++;
	}
	cout << "the optical constrains inserted done..." << endl;
	//3.insert the gradient constrains
	for (int j = 0; j < weight.shape()[0]; j++)
	{
		for (int dim = 0; dim < 2; dim++)
		{
			m.emplace_back(index, (r4(j, 0)*width + r4(j, 1)) * 2 + dim, r8(j, 0, dim));
			m.emplace_back(index, (r4(j, 0)*width + r4(j, 1) + 1) * 2 + dim, r8(j, 1, dim));
			m.emplace_back(index, ((r4(j, 0) + 1)*width + r4(j, 1) + 1) * 2 + dim, r8(j, 2, dim));
			m.emplace_back(index, ((r4(j, 0) + 1)*width + r4(j, 1)) * 2 + dim, r8(j, 3, dim));
			index++;
		}
	}
	cout << "insert the gradient constrains done ..." << endl;

	//4.Insert the line constrains,preserve the shape of line
	//对每一条直线，进行遍历
	//[num,lenth,4]
	for (int i = 0; i < line_weight.size(); i++)
	{
		int line_index = 0;
		//【sample_point_num,4】
		auto rw = line_weight[i].unchecked<2>();
		// 【sample_point_num,2】
		auto rl = line_location[i].unchecked<2>();

		for (int j = 0; j < line_weight[i].shape()[0]; j++)
		{

			float u = float(j) / float((line_weight[i].shape()[0] - 1));
			//cout << "u is :" << u << endl;
			for (int dim = 0; dim < 2; dim++)
			{
				//r11 is the left top vertice location of the mesh grid which the sample point located in.
				m.emplace_back(index, (rl(j, 0)*width + rl(j, 1)) * 2 + dim, beta * rw(j, 0));
				m.emplace_back(index, (rl(j, 0)*width + rl(j, 1) + 1) * 2 + dim, beta * rw(j, 1));
				m.emplace_back(index, ((rl(j, 0) + 1)*width + rl(j, 1) + 1) * 2 + dim, beta * rw(j, 2));
				m.emplace_back(index, ((rl(j, 0) + 1)*width + rl(j, 1)) * 2 + dim, beta * rw(j, 3));

				m.emplace_back(index, (rl(0, 0)*width + rl(0, 1)) * 2 + dim, beta * (u - 1) * rw(0, 0));
				m.emplace_back(index, (rl(0, 0)*width + rl(0, 1) + 1) * 2 + dim, beta * (u - 1) * rw(0, 1));
				m.emplace_back(index, ((rl(0, 0) + 1)*width + rl(0, 1) + 1) * 2 + dim, beta * (u - 1) * rw(0, 2));
				m.emplace_back(index, ((rl(0, 0) + 1)*width + rl(0, 1)) * 2 + dim, beta * (u - 1) * rw(0, 3));

				m.emplace_back(index, (rl((line_weight[i].shape()[0] - 1), 0)*width +
					rl((line_weight[i].shape()[0] - 1), 1)) * 2 + dim,
					-u * beta * (rw((line_weight[i].shape()[0] - 1), 0)));
				m.emplace_back(index, (rl((line_weight[i].shape()[0] - 1), 0)*width +
					rl((line_weight[i].shape()[0] - 1), 1) + 1) * 2 + dim,
					-u * beta * (rw((line_weight[i].shape()[0] - 1), 1)));
				m.emplace_back(index, ((rl((line_weight[i].shape()[0] - 1), 0) + 1)*width +
					rl((line_weight[i].shape()[0] - 1), 1) + 1) * 2 + dim,
					-u * beta * (rw((line_weight[i].shape()[0] - 1), 2)));
				m.emplace_back(index, ((rl((line_weight[i].shape()[0] - 1), 0) + 1)*width +
					rl((line_weight[i].shape()[0] - 1), 1)) * 2 + dim,
					-u * beta* (rw((line_weight[i].shape()[0] - 1), 3)));

				index++;
			}
		}
	}

	//5.insert the triangle constrains
	for (int t = 0; t < triangle.shape()[0]; t++) {
		for (int dim = 0; dim < 2; dim++) {
			float u = r2(t, 0);
			float v = r2(t, 1);
			int vertice_a = r1(t, 0, 0) * width + r1(t, 0, 1);
			int vertice_b = r1(t, 1, 0) * width + r1(t, 1, 1);
			int vertice_c = r1(t, 2, 0) * width + r1(t, 2, 1);
			if (dim == 0) {
				m.emplace_back(index, vertice_a * 2, lambda * 1);
				m.emplace_back(index, vertice_b * 2, lambda * (u - 1));
				m.emplace_back(index, vertice_c * 2, lambda * (-u));
				m.emplace_back(index, vertice_c * 2 + 1, lambda * (-v));
				m.emplace_back(index, vertice_b * 2 + 1, lambda * (v));
				index++;
			}
			else
			{
				m.emplace_back(index, vertice_a * 2 + 1, lambda);
				m.emplace_back(index, vertice_b * 2 + 1, lambda * (u - 1));
				m.emplace_back(index, vertice_c * 2 + 1, lambda * (-u));
				m.emplace_back(index, vertice_b * 2, lambda * (-v));
				m.emplace_back(index, vertice_c * 2, lambda * (v));
				index++;
			}
		}
	}

	cout << "insert the triangle constrains done ..." << endl;
	//set the value of b
	VectorXd bb = VectorXd::Zero(constrain_num);
	VectorXd x = VectorXd::Zero(vertices.shape()[0] * vertices.shape()[1] * 2);
	int index1 = 0;
	int index2 = 0;
	index2 = index1;
	for (int i = 0; i < vertices.shape()[0]; i++) {
		for (int j = 0; j < vertices.shape()[1]; j++) {
			for (int dim = 0; dim < 2; dim++) {
				bb[index1] = gama*r6(i, j, dim);
				index1++;
			}
		}
	}
	for (int i = 0; i < b.shape()[0]; i++)
	{
		bb[index1] = r5(i);
		index1++;
	}
	for (int j = 0; j < bbss.shape()[0]; j++)
	{
		for (int dim = 0; dim < 2; dim++)
		{
			bb[index1] = r9(j, dim);
			index1++;
		}
	}

	cout << "set the b value done ..." << endl;
	LeastSquaresConjugateGradient<SparseMatrix<double>> lscg;
	SparseMatrix<double> A(index, vertices.shape()[0] * vertices.shape()[1] * 2);
	A.setFromTriplets(m.begin(), m.end());
	lscg.compute(A);
	x = lscg.solve(bb);

	//for (int i = 0; i < vertices.shape()[0]; i++) {
	//	for (int j = 0; j < vertices.shape()[1]; j++) {
	//		for (int dim = 0; dim < 2; dim++) {
	//			cout << "the vertices difference is :" << r6(i, j, dim) - x[(i*width + j) * 2 + dim] << endl;
	//			index1++;
	//		}
	//	}
	//}
	for (int i = 0; i < vertices.shape()[0] * vertices.shape()[1] * 2; i++)
	{
		r7(i) = x[i];
	}
	return result;
	list<int> a;
	a.resize(5);
	deque<int> bbbbb;
	bbbbb.resize(4);
	vector<int> ppppp;
	ppppp.resize(10);

}

PYBIND11_MODULE(optimization, m)
{
	m.doc() = "the optimization of vertices of the image";
	m.def("optimize", &optimize);
}



