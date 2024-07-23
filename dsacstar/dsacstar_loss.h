/*
Based on the DSAC++ and ESAC code.
https://github.com/vislearn/LessMore
https://github.com/vislearn/esac

Copyright (c) 2016, TU Dresden
Copyright (c) 2020, Heidelberg University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#define MAXLOSS 10000000.0 // clamp for stability

namespace dsacstar
{
	/**
	 * @brief Calculates the rotational distance in degree between two transformations.
	 * Translation will be ignored.
	 *
	 * @param trans1 Transformation 1.
	 * @param trans2 Transformation 2.
	 * @return Angle in degree.
	 */
	double calcAngularDistance(const dsacstar::trans_t& trans1, const dsacstar::trans_t& trans2)
	{
		cv::Mat rot1 = trans1.colRange(0, 3).rowRange(0, 3);
		cv::Mat rot2 = trans2.colRange(0, 3).rowRange(0, 3);

		cv::Mat rotDiff= rot2 * rot1.t();
		double trace = cv::trace(rotDiff)[0];

		trace = std::min(3.0, std::max(-1.0, trace));
		return 180*acos((trace-1.0)/2.0)/PI;
	}

	/**
	 * @brief Weighted average of  translational error and rotational error between two pose hypothesis.
	 * @param h1 Pose 1.
	 * @param h2 Pose 2.
	 * @param wRot Weight of rotation error.
	 * @param wTrans Weight of translation error.
	 * @param cut Apply soft clamping after this value.
	 * @return Loss.
	 */
	double loss(
		const dsacstar::trans_t& trans1, 
		const dsacstar::trans_t& trans2, 
		double wRot = 1.0, 
		double wTrans = 1.0,
		double cut = 100)
	{
		double rotErr = dsacstar::calcAngularDistance(trans1, trans2);
		double tErr = cv::norm(
			trans1.col(3).rowRange(0, 3) - trans2.col(3).rowRange(0, 3));

		double loss = wRot * rotErr + wTrans * tErr;

		if(loss > cut)
			loss = std::sqrt(cut * loss);

	    return std::min(loss, MAXLOSS);
	}

	/**
	 * @brief Calculate the derivative of the loss w.r.t. the estimated pose.
	 * @param est Estimated pose (6 DoF).
	 * @param gt Ground truth pose (6 DoF).
	 * @param wRot Weight of rotation error.
	 * @param wTrans Weight of translation error.
	 * @param cut Apply soft clamping after this value.	 
	 * @return 1x6 Jacobean.
	 */
	cv::Mat_<double> dLoss(
		const dsacstar::pose_t& est, 
		const dsacstar::pose_t& gt,
		double wRot = 1.0,
		double wTrans = 1.0,
		double cut = 100)
	{
	    cv::Mat rot1, rot2, dRod;
	    cv::Rodrigues(est.first, rot1, dRod);
	    cv::Rodrigues(gt.first, rot2);

	    // measure loss of inverted poses (camera pose instead of scene pose)
	    cv::Mat_<double> invRot1 = rot1.t();
	    cv::Mat_<double> invRot2 = rot2.t();

	    // get the difference rotation
	    cv::Mat diffRot = rot1 * invRot2;

	    // calculate rotational and translational error
	    double trace = cv::trace(diffRot)[0];
	    trace = std::min(3.0, std::max(-1.0, trace));
	    double rotErr = 180*acos((trace-1.0)/2.0)/CV_PI;

	    cv::Mat_<double> invT1 = est.second.clone();
	    invT1 = invRot1 * invT1;

	    cv::Mat_<double> invT2 = gt.second.clone();
	    invT2 = invRot2 * invT2;

	    // zero error, abort
	    double tErr = cv::norm(invT1 - invT2);

	    cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(1, 6);
	    
	    // clamped loss, return zero gradient if loss is bigger than threshold
	    double loss = wRot * rotErr + wTrans * tErr;
	    bool cutLoss = false;


	    if(loss > cut)
	    {
	    	loss = std::sqrt(loss);
	    	cutLoss = true;
	    }

	    if(loss > MAXLOSS)
	        return jacobean;

	    if((tErr + rotErr) < EPS)
	        return jacobean;
		
	  	
	    // return gradient of translational error
	    cv::Mat_<double> dDist_dInvT1(1, 3);
	    for(unsigned i = 0; i < 3; i++)
	        dDist_dInvT1(0, i) = (invT1(i, 0) - invT2(i, 0)) / tErr;

	    cv::Mat_<double> dInvT1_dEstT(3, 3);
	    dInvT1_dEstT = invRot1;

	    cv::Mat_<double> dDist_dEstT = dDist_dInvT1 * dInvT1_dEstT;
	    jacobean.colRange(3, 6) += dDist_dEstT * wTrans;

	    cv::Mat_<double> dInvT1_dInvRot1 = cv::Mat_<double>::zeros(3, 9);

	    dInvT1_dInvRot1(0, 0) = est.second.at<double>(0, 0);
	    dInvT1_dInvRot1(0, 3) = est.second.at<double>(1, 0);
	    dInvT1_dInvRot1(0, 6) = est.second.at<double>(2, 0);

	    dInvT1_dInvRot1(1, 1) = est.second.at<double>(0, 0);
	    dInvT1_dInvRot1(1, 4) = est.second.at<double>(1, 0);
	    dInvT1_dInvRot1(1, 7) = est.second.at<double>(2, 0);

	    dInvT1_dInvRot1(2, 2) = est.second.at<double>(0, 0);
	    dInvT1_dInvRot1(2, 5) = est.second.at<double>(1, 0);
	    dInvT1_dInvRot1(2, 8) = est.second.at<double>(2, 0);

	    dRod = dRod.t();

	    cv::Mat_<double> dDist_dRod = dDist_dInvT1 * dInvT1_dInvRot1 * dRod;
	    jacobean.colRange(0, 3) += dDist_dRod * wTrans;


	    // return gradient of rotational error
	    cv::Mat_<double> dRotDiff = cv::Mat_<double>::zeros(9, 9);
	    invRot2.row(0).copyTo(dRotDiff.row(0).colRange(0, 3));
	    invRot2.row(1).copyTo(dRotDiff.row(1).colRange(0, 3));
	    invRot2.row(2).copyTo(dRotDiff.row(2).colRange(0, 3));

	    invRot2.row(0).copyTo(dRotDiff.row(3).colRange(3, 6));
	    invRot2.row(1).copyTo(dRotDiff.row(4).colRange(3, 6));
	    invRot2.row(2).copyTo(dRotDiff.row(5).colRange(3, 6));

	    invRot2.row(0).copyTo(dRotDiff.row(6).colRange(6, 9));
	    invRot2.row(1).copyTo(dRotDiff.row(7).colRange(6, 9));
	    invRot2.row(2).copyTo(dRotDiff.row(8).colRange(6, 9));

	    dRotDiff = dRotDiff.t();

	    cv::Mat_<double> dTrace = cv::Mat_<double>::zeros(1, 9);
	    dTrace(0, 0) = 1;
	    dTrace(0, 4) = 1;
	    dTrace(0, 8) = 1;

	    cv::Mat_<double> dAngle = (180 / CV_PI * -1 / sqrt(3 - trace * trace + 2 * trace)) * dTrace * dRotDiff * dRod;

	    jacobean.colRange(0, 3) += dAngle * wRot;
		
		if(cutLoss)
			jacobean *= 0.5 / loss;


	    if(cv::sum(cv::Mat(jacobean != jacobean))[0] > 0) //check for NaNs
	        return cv::Mat_<double>::zeros(1, 6);

	    return jacobean;
	}


}
