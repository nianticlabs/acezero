/*
Copyright (c) 2018-2020, Heidelberg University

This code fragment contains adapted sources from PyTorch. See license agreement of PyTorch below.

From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.
 
All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.
 
All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.
 
All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.
 
All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.
 
Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   Heidelberg University and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

*/

#pragma once


//#include "opencv2/core/types_c.h"
//#include "opencv2/calib3d/calib3d_c.h"

/*
 * @brief reimplementation of PyTorch svd_backward in C++
 *
 * ref: https://github.com/pytorch/pytorch/blob/1d427fd6f66b0822db62f30e7654cae95abfd207/tools/autograd/templates/Functions.cpp
 * ref: https://j-towns.github.io/papers/svd-derivative.pdf
 *
 * This makes no assumption on the signs of sigma.
 *
 * @param grad: gradients w.r.t U, W, V
 * @param self: matrix decomposed by SVD
 * @param raw_u: U from SVD output
 * @param sigma: W from SVD output
 * @param raw_v: V from SVD output
 *
 */

cv::Mat svd_backward(
	const std::vector<cv::Mat> &grads, 
	const cv::Mat& self, 
	const cv::Mat& raw_u, 
	const cv::Mat& sigma, 
	const cv::Mat& raw_v)
{
	auto m = self.rows;
	auto n = self.cols;
	auto k = sigma.cols;
	auto gsigma = grads[1];

	auto u = raw_u;
	auto v = raw_v;
	auto gu = grads[0];
	auto gv = grads[2];

	auto vt = v.t();

	cv::Mat sigma_term;

	if (!gsigma.empty())
	{
		sigma_term = (u * cv::Mat::diag(gsigma)) * vt;
	}
	else
	{
		sigma_term = cv::Mat::zeros(self.size(), self.type());
	}
	// in case that there are no gu and gv, we can avoid the series of kernel
	// calls below
	if (gv.empty() && gu.empty())
	{
		return sigma_term;
	}

	auto ut = u.t();
	auto im = cv::Mat::eye((int)m, (int)m, self.type());
	auto in = cv::Mat::eye((int)n, (int)n, self.type());
	auto sigma_mat = cv::Mat::diag(sigma);

	cv::Mat sigma_mat_inv;
	cv::pow(sigma, -1, sigma_mat_inv);
	sigma_mat_inv = cv::Mat::diag(sigma_mat_inv);

	cv::Mat sigma_sq, sigma_expanded_sq;
	cv::pow(sigma, 2, sigma_sq);
	sigma_expanded_sq = cv::repeat(sigma_sq, sigma_mat.rows, 1);

	cv::Mat F = sigma_expanded_sq - sigma_expanded_sq.t();
	// The following two lines invert values of F, and fills the diagonal with 0s.
	// Notice that F currently has 0s on diagonal. So we fill diagonal with +inf
	// first to prevent nan from appearing in backward of this function.
	F.diag().setTo(std::numeric_limits<float>::max());
	cv::pow(F, -1, F);

	cv::Mat u_term, v_term;

	if (!gu.empty())
	{
		cv::multiply(F, ut*gu-gu.t()*u, u_term);
		u_term = (u * u_term) * sigma_mat;
		if (m > k)
		{
			u_term = u_term + ((im - u*ut)*gu)*sigma_mat_inv;
		}
		u_term = u_term * vt;
	}
	else
	{
		u_term = cv::Mat::zeros(self.size(), self.type());
	}

	if (!gv.empty())
	{
		auto gvt = gv.t();
		cv::multiply(F, vt*gv - gvt*v, v_term);
		v_term = (sigma_mat*v_term) * vt;
		if (n > k)
		{
			v_term = v_term + sigma_mat_inv*(gvt*(in - v*vt));
		}
		v_term = u * v_term;
	}
	else
	{
		v_term = cv::Mat::zeros(self.size(), self.type());
	}

	return u_term + sigma_term + v_term;
}

/*
 * @brief Compute partial derivatives of the matrix product for each multiplied matrix.
 * 			This wrapper function avoids unnecessary computation
 *
 * @param _Amat: First multiplied matrix.
 * @param _Bmat: Second multiplied matrix.
 * @param _dABdA Output parameter: First output derivative matrix. Pass cv::noArray() if not needed.
 * @param _dABdB Output parameter: Second output derivative matrix. Pass cv::noArray() if not needed.
 *
 */
void matMulDerivWrapper(
	cv::InputArray _Amat, 
	cv::InputArray _Bmat, 
	cv::OutputArray _dABdA, 
	cv::OutputArray _dABdB)
{
	cv::Mat A = _Amat.getMat(), B = _Bmat.getMat();

	if (_dABdA.needed())
	{
		_dABdA.create(A.rows*B.cols, A.rows*A.cols, A.type());
	}

	if (_dABdB.needed())
	{
		_dABdB.create(A.rows*B.cols, B.rows*B.cols, A.type());
	}

	CvMat matA = A, matB = B, c_dABdA=_dABdA.getMat(), c_dABdB=_dABdB.getMat();
	cvCalcMatMulDeriv(&matA, &matB, _dABdA.needed() ? &c_dABdA : 0, _dABdB.needed() ? &c_dABdB : 0);
}

/*
 * @brief Computes extrinsic camera parameters using Kabsch algorithm.
 * 			If jacobean matrix is passed as argument, it further computes the analytical gradients.
 *
 * @param imgdPts: measurements
 * @param objPts: scene points
 * @param extCam Output parameter: extrinsic camera matrix (i.e. rotation vector and translation vector)
 * @param _jacobean Output parameter: 6x3N jacobean matrix of rotation and translation vector
 * 					w.r.t scene point coordinates.
 * 					If gradient computation is not successful, jacobean matrix is set empty.
 *
 */
void kabsch(
	const std::vector<cv::Point3f>& imgdPts, 
	const std::vector<cv::Point3f>& objPts, 
	dsacstar::pose_t& extCam, 
	cv::OutputArray _jacobean=cv::noArray())
{
	unsigned int N = objPts.size();  //number of scene points
	bool calc = _jacobean.needed();  //check if computation of gradient is required
	bool degenerate = false;  //indicate if SVD gives degenerate case, i.e. non-distinct or zero singular values

	cv::Mat P, X, Pc, Xc;  //Nx3
	cv::Mat A, U, W, Vt, V, D, R;  //3x3
	cv::Mat cx, cp, r, t;  //1x3
	cv::Mat invN;  //1xN
	cv::Mat gRodr;  //9x3

	// construct the datasets P and X from input vectors, set false to avoid data copying
	P = cv::Mat(imgdPts, false).reshape(1, N);
	X = cv::Mat(objPts, false).reshape(1, N);

	// compute centroid as average of each coordinate axis
	invN = cv::Mat(1, N, CV_32F, 1.f/N);  //average filter
	cx = invN * X;
	cp = invN * P;

	// move centroid of datasets to origin
	Xc =  X - cv::repeat(cx, N, 1);
	Pc =  P - cv::repeat(cp, N, 1);

	// compute covariance matrix
	A = Pc.t() * Xc;

	// compute SVD of covariance matrix
	cv::SVD::compute(A, W, U, Vt);

	// degenerate if any singular value is zero
	if ((unsigned int)cv::countNonZero(W) != (unsigned int)W.total())
		degenerate = true;

	// degenerate if singular values are not distinct
	if (std::abs(W.at<float>(0,0)-W.at<float>(1,0)) < 1e-6
			|| std::abs(W.at<float>(0,0)-W.at<float>(2,0)) < 1e-6
			|| std::abs(W.at<float>(1,0)-W.at<float>(2,0)) < 1e-6)
		degenerate = true;
	
	// for correcting rotation matrix to ensure a right-handed coordinate system
	float d = cv::determinant(U * Vt);

	D = (cv::Mat_<float>(3,3) <<
				1., 0., 0.,
				0., 1., 0.,
				0., 0., d );
	
	// calculates rotation matrix R
	R = U * (D * Vt);

	// convert rotation matrix to rotation vector,
	// if needed, also compute jacobean matrix of rotation matrix w.r.t rotation vector
	calc ? cv::Rodrigues(R, r, gRodr) : cv::Rodrigues(R, r);

	// calculates translation vector
	t = cp - cx * R.t();  //equiv: cp - (R*cx.t()).t();

	// store results
	extCam.first = cv::Mat_<double>(r.reshape(1, 3));
	extCam.second = cv::Mat_<double>(t.reshape(1, 3));
	
	// end here no gradient is required
	if (!calc)
		return;
	
	// if SVD is degenerate, return empty jacobean matrix
	if (degenerate)
	{
		_jacobean.release();
		return;
	}

	// allocate matrix data
	_jacobean.create(6, N*3, CV_64F);
	cv::Mat jacobean = _jacobean.getMat();

//	cv::Mat dRidU, dRidVt, dRidV, dRidA, dRidXc, dRidX;
	cv::Mat dRdU, dRdVt;  //9x9
	cv::Mat dAdXc;  //9x3N
	cv::Mat dtdR;  //3x9
	cv::Mat dtdcx;  //3x3
	cv::Mat dcxdX, drdX, dtdX;  //3x3N
	cv::Mat dRdX = cv::Mat_<float>::zeros(9, N*3);  //9x3N

	// jacobean matrices of each dot product operation in kabsch algorithm
	matMulDerivWrapper(U, Vt, dRdU, dRdVt);
	matMulDerivWrapper(Pc.t(), Xc, cv::noArray(), dAdXc);
	matMulDerivWrapper(R, cx.t(), dtdR, dtdcx);
	matMulDerivWrapper(invN, X, cv::noArray(), dcxdX);
	
	V = Vt.t();
	W = W.reshape(1, 1);

//	#pragma omp parallel for
	for (int i = 0; i < 9; ++i)
	{
		cv::Mat dRidU, dRidVt, dRidV, dRidA;  //3x3
		cv::Mat dRidXc, dRidX;  //Nx3

		dRidU = dRdU.row(i).reshape(1, 3);
		dRidVt = dRdVt.row(i).reshape(1, 3);

		dRidV = dRidVt.t();

		//W is not used in computation of R, no gradient of W is needed
		std::vector<cv::Mat> grads{dRidU, cv::Mat(), dRidV};

		dRidA = svd_backward(grads, A, U, W, V);
		dRidA = dRidA.reshape(1, 1);
		dRidXc = dRidA * dAdXc;
		dRidXc = dRidXc.reshape(1, N);
		dRidX = cv::Mat::zeros(dRidXc.size(), dRidXc.type());

		int bstep = dRidXc.step/CV_ELEM_SIZE(dRidXc.type());

//		#pragma omp parallel for
		for (int j = 0; j < 3; ++j)
		{
			// compute dRidXj = dRidXcj * dXcjdXj
			float* pdRidXj = (float*)dRidX.data + j;
			const float* pdRidXcj = (const float*)dRidXc.data + j;

			float tmp = 0.f;
			for (unsigned int k = 0; k < N; ++k)
			{
				tmp += pdRidXcj[k*bstep];
			}
			tmp /= N;

			for (unsigned int k = 0; k < N; ++k)
			{
				pdRidXj[k*bstep] = pdRidXcj[k*bstep] - tmp;
			}
		}

		dRidX = dRidX.reshape(1, 1);
		dRidX.copyTo(dRdX.rowRange(i, i+1));
	}

	drdX = gRodr.t() * dRdX;
	drdX.copyTo(jacobean.rowRange(0, 3));
	dtdX = - (dtdR * dRdX + dtdcx * dcxdX);
	dtdX.copyTo(jacobean.rowRange(3, 6));
	
}

/**
* @brief Checks whether the given matrix contains NaN entries.
* @param m Input matrix.
* @return True if m contrains NaN entries.
*/
inline bool containsNaNs(const cv::Mat& m)
{
    return cv::sum(cv::Mat(m != m))[0] > 0;
}

/*
 * @brief Computes gradient of Kabsch algorithm and differentiation
 * 			using central finite differences
 *
 * @param imgdPts: measurement points
 * @param objPts: scene points
 * @param jacobean Output parameter: 6x3N jacobean matrix of rotation and translation vector
 * 										w.r.t scene point coordinates
 * @param eps: step size in finite difference approximation
 *
 */
void dKabschFD(
	std::vector<cv::Point3f>& imgdPts, 
	std::vector<cv::Point3f> objPts, 
	cv::OutputArray _jacobean, 
	float eps = 0.001f)
{
	_jacobean.create(6, objPts.size()*3, CV_64F);
	cv::Mat jacobean = _jacobean.getMat();

	for (unsigned int i = 0; i < objPts.size(); ++i)
	{
		for (unsigned int j = 0; j < 3; ++j)
		{

			if(j == 0) objPts[i].x += eps;
			else if(j == 1) objPts[i].y += eps;
			else if(j == 2) objPts[i].z += eps;

			// forward step

			dsacstar::pose_t fStep;
			kabsch(imgdPts, objPts, fStep);

			if(j == 0) objPts[i].x -= 2 * eps;
			else if(j == 1) objPts[i].y -= 2 * eps;
			else if(j == 2) objPts[i].z -= 2 * eps;

			// backward step
			dsacstar::pose_t bStep;
			kabsch(imgdPts, objPts, bStep);

			if(j == 0) objPts[i].x += eps;
			else if(j == 1) objPts[i].y += eps;
			else if(j == 2) objPts[i].z += eps;

			// gradient calculation
			fStep.first = (fStep.first - bStep.first) / (2 * eps);
			fStep.second = (fStep.second - bStep.second) / (2 * eps);

			fStep.first.copyTo(jacobean.col(i * 3 + j).rowRange(0, 3));
			fStep.second.copyTo(jacobean.col(i * 3 + j).rowRange(3, 6));

			if(containsNaNs(jacobean.col(i * 3 + j)))
				jacobean.setTo(0);

		}
	}

}

void transform(
	const std::vector<cv::Point3f>& objPts, 
	const dsacstar::pose_t& transform, 
	std::vector<cv::Point3f>& transformedPts)
{
	
	cv::Mat rot;
	cv::Rodrigues(transform.first, rot);

	// P = RX + T
	for (unsigned int i = 0; i < objPts.size(); ++i)
	{
		cv::Mat_<double> res= rot * cv::Mat_<double>(objPts[i], false);

		cv::add(res, transform.second, res);

		transformedPts.push_back(cv::Point3f(
			res.at<double>(0,0), 
			res.at<double>(1,0), 
			res.at<double>(2,0)));
	}
}