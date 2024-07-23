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

#include <torch/extension.h>
#include <opencv2/opencv.hpp>

#include <iostream>

#include "thread_rand.h"
#include "stop_watch.h"

#include "dsacstar_types.h"
#include "dsacstar_util.h"
//#include "dsacstar_util_rgbd.h"
#include "dsacstar_loss.h"
#include "dsacstar_derivative.h"

#define MAX_REF_STEPS 100 // max pose refienment iterations
#define MAX_HYPOTHESES_TRIES 16 // repeat sampling x times hypothesis if hypothesis is invalid

/**
 * @brief Estimate a camera pose based on a scene coordinate prediction
 * @param sceneCoordinatesSrc Scene coordinate prediction, (1x3xHxW) with 1=batch dimension (only batch_size=1 supported atm), 3=scene coordainte dimensions, H=height and W=width.
 * @param outPoseSrc Camera pose (output parameter), (4x4) tensor containing the homogeneous camera tranformation matrix.
 * @param ransacHypotheses Number of RANSAC iterations.
 * @param inlierThreshold Inlier threshold for RANSAC in px.
 * @param focalLength Focal length of the camera in px.
 * @param ppointX Coordinate (X) of the prinicpal points.
 * @param ppointY Coordinate (Y) of the prinicpal points.
 * @param inlierAlpha Alpha parameter for soft inlier counting.
 * @param maxReproj Reprojection errors are clamped above this value (px).
 * @param subSampling Sub-sampling  of the scene coordinate prediction wrt the input image.
 * @return The number of inliers for the output pose.
 */
int dsacstar_rgb_forward(
	at::Tensor sceneCoordinatesSrc, 
	at::Tensor outPoseSrc,
	int ransacHypotheses, 
	float inlierThreshold,
	float focalLength,
	float ppointX,
	float ppointY,
	float inlierAlpha,
	float maxReproj,
	int subSampling,
	int randomSeed)
{
	ThreadRand::init(randomSeed);

	// access to tensor objects
	dsacstar::coord_t sceneCoordinates = 
		sceneCoordinatesSrc.accessor<float, 4>();

	// dimensions of scene coordinate predictions
	int imH = sceneCoordinates.size(2);
	int imW = sceneCoordinates.size(3);

	// internal camera calibration matrix
	cv::Mat_<float> camMat = cv::Mat_<float>::eye(3, 3);
	camMat(0, 0) = focalLength;
	camMat(1, 1) = focalLength;
	camMat(0, 2) = ppointX;
	camMat(1, 2) = ppointY;	

	// calculate original image position for each scene coordinate prediction
	cv::Mat_<cv::Point2i> sampling = 
		dsacstar::createSampling(imW, imH, subSampling, 0, 0);

	std::cout << BLUETEXT("Sampling " << ransacHypotheses << " hypotheses.") << std::endl;
	StopWatch stopW;

	// sample RANSAC hypotheses
	std::vector<dsacstar::pose_t> hypotheses;
	std::vector<std::vector<cv::Point2i>> sampledPoints;  
	std::vector<std::vector<cv::Point2f>> imgPts;
	std::vector<std::vector<cv::Point3f>> objPts;

	dsacstar::sampleHypotheses(
		sceneCoordinates,
		sampling,
		camMat,
		ransacHypotheses,
		MAX_HYPOTHESES_TRIES,
		inlierThreshold,
		hypotheses,
		sampledPoints,
		imgPts,
		objPts);

	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;	
	std::cout << BLUETEXT("Calculating scores.") << std::endl;
    
	// compute reprojection error images
	std::vector<cv::Mat_<float>> reproErrs(ransacHypotheses);
	cv::Mat_<double> jacobeanDummy;

	#pragma omp parallel for 
	for(unsigned h = 0; h < hypotheses.size(); h++)
    	reproErrs[h] = dsacstar::getReproErrs(
		sceneCoordinates,
		hypotheses[h], 
		sampling, 
		camMat,
		maxReproj,
		jacobeanDummy);

    // soft inlier counting
	std::vector<double> scores = dsacstar::getHypScores(
    	reproErrs,
    	inlierThreshold,
    	inlierAlpha);

	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
	std::cout << BLUETEXT("Drawing final hypothesis.") << std::endl;	

	// apply soft max to scores to get a distribution
	std::vector<double> hypProbs = dsacstar::softMax(scores);
	double hypEntropy = dsacstar::entropy(hypProbs); // measure distribution entropy
	int hypIdx = dsacstar::draw(hypProbs, false); // select winning hypothesis

	std::cout << "Soft inlier count: " << scores[hypIdx] << " (Selection Probability: " << (int) (hypProbs[hypIdx]*100) << "%)" << std::endl; 
	std::cout << "Entropy of hypothesis distribution: " << hypEntropy << std::endl;


	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
	std::cout << BLUETEXT("Refining winning pose:") << std::endl;

	// refine selected hypothesis
	cv::Mat_<int> inlierMap;

	dsacstar::refineHyp(
		sceneCoordinates,
		reproErrs[hypIdx],
		sampling,
		camMat,
		inlierThreshold,
		MAX_REF_STEPS,
		maxReproj,
		hypotheses[hypIdx],
		inlierMap);

	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

	// write result back to PyTorch
	dsacstar::trans_t estTrans = dsacstar::pose2trans(hypotheses[hypIdx]);

	auto outPose = outPoseSrc.accessor<float, 2>();
	for(unsigned x = 0; x < 4; x++)
	for(unsigned y = 0; y < 4; y++)
		outPose[y][x] = estTrans(y, x);

	// Return the inlier count. cv::sum returns a scalar, so we return its first element.
	return cv::sum(inlierMap)[0];
}
//
///**
// * @brief Performs pose estimation, and calculates the gradients of the pose loss wrt to scene coordinates.
// * @param sceneCoordinatesSrc Scene coordinate prediction, (1x3xHxW) with 1=batch dimension (only batch_size=1 supported atm), 3=scene coordainte dimensions, H=height and W=width.
// * @param outSceneCoordinatesGradSrc Scene coordinate gradients (output parameter). (1x3xHxW) same as scene coordinate input.
// * @param gtPoseSrc Ground truth camera pose, (4x4) tensor.
// * @param ransacHypotheses Number of RANSAC iterations.
// * @param inlierThreshold Inlier threshold for RANSAC in px.
// * @param focalLength Focal length of the camera in px.
// * @param ppointX Coordinate (X) of the prinicpal points.
// * @param ppointY Coordinate (Y) of the prinicpal points.
// * @param wLossRot Weight of the rotation loss term.
// * @param wLossTrans Weight of the translation loss term.
// * @param softClamp Use sqrt of pose loss after this threshold.
// * @param inlierAlpha Alpha parameter for soft inlier counting.
// * @param maxReproj Reprojection errors are clamped above this value (px).
// * @param subSampling Sub-sampling  of the scene coordinate prediction wrt the input image.
// * @param randomSeed External random seed to make sure we draw different samples across calls of this function.
// * @return DSAC expectation of the pose loss.
// */
//
//double dsacstar_rgb_backward(
//	at::Tensor sceneCoordinatesSrc,
//	at::Tensor outSceneCoordinatesGradSrc,
//	at::Tensor gtPoseSrc,
//	int ransacHypotheses,
//	float inlierThreshold,
//	float focalLength,
//	float ppointX,
//	float ppointY,
//	float wLossRot,
//	float wLossTrans,
//	float softClamp,
//	float inlierAlpha,
//	float maxReproj,
//	int subSampling,
//	int randomSeed)
//{
//	ThreadRand::init(randomSeed);
//
//	// access to tensor objects
//	dsacstar::coord_t sceneCoordinates =
//		sceneCoordinatesSrc.accessor<float, 4>();
//
//	dsacstar::coord_t sceneCoordinatesGrads =
//		outSceneCoordinatesGradSrc.accessor<float, 4>();
//
//	// dimensions of scene coordinate predictions
//	int imH = sceneCoordinates.size(2);
//	int imW = sceneCoordinates.size(3);
//
//	// internal camera calibration matrix
//	cv::Mat_<float> camMat = cv::Mat_<float>::eye(3, 3);
//	camMat(0, 0) = focalLength;
//	camMat(1, 1) = focalLength;
//	camMat(0, 2) = ppointX;
//	camMat(1, 2) = ppointY;
//
//	//convert ground truth pose type
//	dsacstar::trans_t gtTrans(4, 4);
//	auto gtPose = gtPoseSrc.accessor<float, 2>();
//
//	for(unsigned x = 0; x < 4; x++)
//	for(unsigned y = 0; y < 4; y++)
//		gtTrans(y, x) = gtPose[y][x];
//
//	// calculate original image position for each scene coordinate prediction
//	cv::Mat_<cv::Point2i> sampling =
//		dsacstar::createSampling(imW, imH, subSampling, 0, 0);
//
//	// sample RANSAC hypotheses
//	std::cout << BLUETEXT("Sampling " << ransacHypotheses << " hypotheses.") << std::endl;
//	StopWatch stopW;
//
//	std::vector<dsacstar::pose_t> initHyps;
//	std::vector<std::vector<cv::Point2i>> sampledPoints;
//	std::vector<std::vector<cv::Point2f>> imgPts;
//	std::vector<std::vector<cv::Point3f>> objPts;
//
//	dsacstar::sampleHypotheses(
//		sceneCoordinates,
//		sampling,
//		camMat,
//		ransacHypotheses,
//		MAX_HYPOTHESES_TRIES,
//		inlierThreshold,
//		initHyps,
//		sampledPoints,
//		imgPts,
//		objPts);
//
//	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//    std::cout << BLUETEXT("Calculating scores.") << std::endl;
//
//	// compute reprojection error images
//	std::vector<cv::Mat_<float>> reproErrs(ransacHypotheses);
//	std::vector<cv::Mat_<double>> jacobeansHyp(ransacHypotheses);
//
//	#pragma omp parallel for
//	for(unsigned h = 0; h < initHyps.size(); h++)
//    	reproErrs[h] = dsacstar::getReproErrs(
//		sceneCoordinates,
//		initHyps[h],
//		sampling,
//		camMat,
//		maxReproj,
//		jacobeansHyp[h],
//		true);
//
//    // soft inlier counting
//	std::vector<double> scores = dsacstar::getHypScores(
//    	reproErrs,
//    	inlierThreshold,
//    	inlierAlpha);
//
//	// apply soft max to scores to get a distribution
//	std::vector<double> hypProbs = dsacstar::softMax(scores);
//	double hypEntropy = dsacstar::entropy(hypProbs); // measure distribution entropy
//	std::cout << "Entropy: " << hypEntropy << std::endl;
//
//	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//	std::cout << BLUETEXT("Refining poses:") << std::endl;
//
//	// collect inliers and refine poses
//	std::vector<dsacstar::pose_t> refHyps(ransacHypotheses);
//	std::vector<cv::Mat_<int>> inlierMaps(refHyps.size());
//
//	#pragma omp parallel for
//	for(unsigned h = 0; h < refHyps.size(); h++)
//	{
//		refHyps[h].first = initHyps[h].first.clone();
//		refHyps[h].second = initHyps[h].second.clone();
//
//		if(hypProbs[h] < PROB_THRESH) continue; // save computation when little influence on expectation
//
//		dsacstar::refineHyp(
//			sceneCoordinates,
//			reproErrs[h],
//			sampling,
//			camMat,
//			inlierThreshold,
//			MAX_REF_STEPS,
//			maxReproj,
//			refHyps[h],
//			inlierMaps[h]);
//	}
//
//	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//
//	// calculate expected pose loss
//	double expectedLoss = 0;
//	std::vector<double> losses(refHyps.size());
//
//	for(unsigned h = 0; h < refHyps.size(); h++)
//	{
//		dsacstar::trans_t estTrans = dsacstar::pose2trans(refHyps[h]);
//		losses[h] = dsacstar::loss(estTrans, gtTrans, wLossRot, wLossTrans, softClamp);
//		expectedLoss += hypProbs[h] * losses[h];
//	}
//
//   	// === doing the backward pass ====================================================================
//
//	// acumulate hypotheses gradients for patches
//	cv::Mat_<double> dLoss_dObj = cv::Mat_<double>::zeros(sampling.rows * sampling.cols, 3);
//
//    // --- path I, hypothesis path --------------------------------------------------------------------
//    std::cout << BLUETEXT("Calculating gradients wrt hypotheses.") << std::endl;
//
//    // precalculate gradients per of hypotheis wrt object coordinates
//    std::vector<cv::Mat_<double>> dHyp_dObjs(refHyps.size());
//
//    #pragma omp parallel for
//    for(unsigned h = 0; h < refHyps.size(); h++)
//    {
//		int batchIdx = 0; // only batch size=1 supported atm
//
//        // differentiate refinement around optimum found in last optimization iteration
//        dHyp_dObjs[h] = cv::Mat_<double>::zeros(6, sampling.rows * sampling.cols * 3);
//
//        if(hypProbs[h] < PROB_THRESH) continue; // skip hypothesis with no impact on expectation
//
//        // collect inlier correspondences of last refinement iteration
//        std::vector<cv::Point2f> imgPts;
//        std::vector<cv::Point2i> srcPts;
//        std::vector<cv::Point3f> objPts;
//
//        for(int x = 0; x < inlierMaps[h].cols; x++)
//        for(int y = 0; y < inlierMaps[h].rows; y++)
//        {
//            if(inlierMaps[h](y, x))
//            {
//                imgPts.push_back(sampling(y, x));
//                srcPts.push_back(cv::Point2i(x, y));
//                objPts.push_back(cv::Point3f(
//					sceneCoordinates[batchIdx][0][y][x],
//					sceneCoordinates[batchIdx][1][y][x],
//					sceneCoordinates[batchIdx][2][y][x]));
//            }
//        }
//
//        if(imgPts.size() < 4)
//            continue;
//
//        // calculate reprojection errors
//        std::vector<cv::Point2f> projections;
//        cv::Mat_<double> projectionsJ;
//        cv::projectPoints(objPts, refHyps[h].first, refHyps[h].second, camMat, cv::Mat(), projections, projectionsJ);
//
//        projectionsJ = projectionsJ.colRange(0, 6);
//
//        //assemble the jacobean of the refinement residuals
//        cv::Mat_<double> jacobeanR = cv::Mat_<double> ::zeros(objPts.size(), 6);
//        cv::Mat_<double> dNdP(1, 2);
//        cv::Mat_<double> dNdH(1, 6);
//
//        for(unsigned ptIdx = 0; ptIdx < objPts.size(); ptIdx++)
//        {
//            double err = std::max(cv::norm(projections[ptIdx] - imgPts[ptIdx]), EPS);
//            if(err > maxReproj)
//                continue;
//
//            // derivative of norm
//            dNdP(0, 0) = 1 / err * (projections[ptIdx].x - imgPts[ptIdx].x);
//            dNdP(0, 1) = 1 / err * (projections[ptIdx].y - imgPts[ptIdx].y);
//
//            dNdH = dNdP * projectionsJ.rowRange(2 * ptIdx, 2 * ptIdx + 2);
//            dNdH.copyTo(jacobeanR.row(ptIdx));
//        }
//
//        //calculate the pseudo inverse
//        jacobeanR = - (jacobeanR.t() * jacobeanR).inv(cv::DECOMP_SVD) * jacobeanR.t();
//
//        double maxJR = dsacstar::getMax(jacobeanR);
//        if(maxJR > 10) jacobeanR = 0; // clamping for stability
//
//        cv::Mat rot;
//        cv::Rodrigues(refHyps[h].first, rot);
//
//        for(unsigned ptIdx = 0; ptIdx < objPts.size(); ptIdx++)
//        {
//            cv::Mat_<double> dNdO = dsacstar::dProjectdObj(imgPts[ptIdx], objPts[ptIdx], rot, refHyps[h].second, camMat, maxReproj);
//            dNdO = jacobeanR.col(ptIdx) * dNdO;
//
//            int dIdx = srcPts[ptIdx].y * sampling.cols * 3 + srcPts[ptIdx].x * 3;
//            dNdO.copyTo(dHyp_dObjs[h].colRange(dIdx, dIdx + 3));
//        }
//    }
//
//    // combine gradients per hypothesis
//    std::vector<cv::Mat_<double>> gradients(refHyps.size());
//    dsacstar::pose_t hypGT = dsacstar::trans2pose(gtTrans);
//
//    #pragma omp parallel for
//    for(unsigned h = 0; h < refHyps.size(); h++)
//    {
//		if(hypProbs[h] < PROB_THRESH) continue;
//
//        cv::Mat_<double> dLoss_dHyp = dsacstar::dLoss(refHyps[h], hypGT, wLossRot, wLossTrans, softClamp);
//        gradients[h] = dLoss_dHyp * dHyp_dObjs[h];
//    }
//
//	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//
//    // --- path II, score path --------------------------------------------------------------------
//
//    std::cout << BLUETEXT("Calculating gradients wrt scores.") << std::endl;
//
//    std::vector<cv::Mat_<double>> dLoss_dScore_dObjs = dsacstar::dSMScore(
//    	sceneCoordinates,
//    	sampling,
//    	sampledPoints,
//    	losses,
//    	hypProbs,
//    	initHyps,
//    	reproErrs,
//    	jacobeansHyp,
//    	camMat,
//    	inlierAlpha,
//    	inlierThreshold,
//    	maxReproj);
//
//    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//
//    // assemble full gradient tensor
//    for(unsigned h = 0; h < refHyps.size(); h++)
//    {
//		if(hypProbs[h] < PROB_THRESH) continue;
//		int batchIdx = 0; // only batch size=1 supported atm
//
//	    for(int idx = 0; idx < sampling.rows * sampling.cols; idx++)
//	    {
//	    	int x = idx % sampling.cols;
//	    	int y = idx / sampling.cols;
//
//	        sceneCoordinatesGrads[batchIdx][0][y][x] +=
//	        	hypProbs[h] * gradients[h](idx * 3 + 0) + dLoss_dScore_dObjs[h](idx, 0);
//	        sceneCoordinatesGrads[batchIdx][1][y][x] +=
//	        	hypProbs[h] * gradients[h](idx * 3 + 1) + dLoss_dScore_dObjs[h](idx, 1);
//	        sceneCoordinatesGrads[batchIdx][2][y][x] +=
//	        	hypProbs[h] * gradients[h](idx * 3 + 2) + dLoss_dScore_dObjs[h](idx, 2);
//	    }
//	}
//
//	return expectedLoss;
//}
//
///**
// * @brief Estimate a camera pose based on a scene coordinate prediction
// * @param sceneCoordinatesSrc Scene coordinate prediction, (1x3xHxW) with 1=batch dimension (only batch_size=1 supported atm), 3=scene coordainte dimensions, H=height and W=width.
// * @param cameraCoordinatesSrc Camera coordinates (from measured depth), same size as scene coordinates.
// * @param outPoseSrc Camera pose (output parameter), (4x4) tensor containing the homogeneous camera tranformation matrix.
// * @param ransacHypotheses Number of RANSAC iterations.
// * @param inlierThreshold Inlier threshold for RANSAC in centimeters.
// * @param inlierAlpha Alpha parameter for soft inlier counting.
// * @param maxDistError Clamp pixel distance error with this value.
// * @return The number of inliers for the output pose.
// */
//int dsacstar_rgbd_forward(
//	at::Tensor sceneCoordinatesSrc,
//	at::Tensor cameraCoordinatesSrc,
//	at::Tensor outPoseSrc,
//	int ransacHypotheses,
//	float inlierThreshold,
//	float inlierAlpha,
//	float maxDistError)
//{
//	ThreadRand::init();
//
//	// access to tensor objects
//	dsacstar::coord_t sceneCoordinates =
//		sceneCoordinatesSrc.accessor<float, 4>();
//
//	dsacstar::coord_t cameraCoordinates =
//		cameraCoordinatesSrc.accessor<float, 4>();
//
//	// dimensions of scene coordinate predictions
//	int imH = sceneCoordinates.size(2);
//	int imW = sceneCoordinates.size(3);
//
//	// collect all points with valid camera coordinate (ie valid depth measurement)
//	std::vector<cv::Point2i> validPts;
//	for(int x = 0; x < imW; x++)
//	for(int y = 0; y < imH; y++)
//	{
//		if(	cameraCoordinates[0][0][y][x] != 0 &&
//			cameraCoordinates[0][0][y][x] != 0 &&
//			cameraCoordinates[0][0][y][x] != 0)
//			validPts.push_back(cv::Point2i(x, y));
//	}
//
//	std::cout << "Valid points: " << validPts.size() << std::endl;
//
//
//	std::cout << BLUETEXT("Sampling " << ransacHypotheses << " hypotheses.") << std::endl;
//	StopWatch stopW;
//
//	// sample RANSAC hypotheses
//	std::vector<dsacstar::pose_t> hypotheses;
//	std::vector<std::vector<cv::Point2i>> sampledPoints;
//	std::vector<std::vector<cv::Point3f>> eyePts;
//	std::vector<std::vector<cv::Point3f>> objPts;
//
//	dsacstar::sampleHypothesesRGBD(
//		sceneCoordinates,
//		cameraCoordinates,
//		validPts,
//		ransacHypotheses,
//		MAX_HYPOTHESES_TRIES,
//		inlierThreshold,
//		hypotheses,
//		sampledPoints,
//		eyePts,
//		objPts);
//
//	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//	std::cout << BLUETEXT("Calculating scores.") << std::endl;
//
//	// compute distance error images
//	std::vector<cv::Mat_<float>> distErrs(ransacHypotheses);
//
//	#pragma omp parallel for
//	for(unsigned h = 0; h < hypotheses.size(); h++)
//    	distErrs[h] = dsacstar::get3DDistErrs(
//		hypotheses[h],
//		sceneCoordinates,
//		cameraCoordinates,
//		validPts,
//		maxDistError);
//
//    // soft inlier counting
//	std::vector<double> scores = dsacstar::getHypScores(
//    	distErrs,
//    	inlierThreshold,
//    	inlierAlpha);
//
//	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//	std::cout << BLUETEXT("Drawing final hypothesis.") << std::endl;
//
//	// apply soft max to scores to get a distribution
//	std::vector<double> hypProbs = dsacstar::softMax(scores);
//	double hypEntropy = dsacstar::entropy(hypProbs); // measure distribution entropy
//	int hypIdx = dsacstar::draw(hypProbs, false); // select winning hypothesis
//
//	std::cout << "Soft inlier count: " << scores[hypIdx] << " (Selection Probability: " << (int) (hypProbs[hypIdx]*100) << "%)" << std::endl;
//	std::cout << "Entropy of hypothesis distribution: " << hypEntropy << std::endl;
//
//
//	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//	std::cout << BLUETEXT("Refining winning pose:") << std::endl;
//
//	// refine selected hypothesis
//	cv::Mat_<int> inlierMap;
//
//	dsacstar::refineHypRGBD(
//		sceneCoordinates,
//		cameraCoordinates,
//		distErrs[hypIdx],
//		validPts,
//		inlierThreshold,
//		MAX_REF_STEPS,
//		maxDistError,
//		hypotheses[hypIdx],
//		inlierMap);
//
//	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//
//	// write result back to PyTorch
//	dsacstar::trans_t estTrans = dsacstar::pose2trans(hypotheses[hypIdx]);
//
//	auto outPose = outPoseSrc.accessor<float, 2>();
//	for(unsigned x = 0; x < 4; x++)
//	for(unsigned y = 0; y < 4; y++)
//		outPose[y][x] = estTrans(y, x);
//
//	// Return the inlier count. cv::sum returns a scalar, so we return its first element.
//	return cv::sum(inlierMap)[0];
//}
//
///**
// * @brief Performs pose estimation from RGB-D, and calculates the gradients of the pose loss wrt to scene coordinates.
// * @param sceneCoordinatesSrc Scene coordinate prediction, (1x3xHxW) with 1=batch dimension (only batch_size=1 supported atm), 3=scene coordainte dimensions, H=height and W=width.
// * @param cameraCoordinatesSrc Camera coordinates (from measured depth), same size as scene coordinates.
// * @param outSceneCoordinatesGradSrc Scene coordinate gradients (output parameter). (1x3xHxW) same as scene coordinate input.
// * @param gtPoseSrc Ground truth camera pose, (4x4) tensor.
// * @param ransacHypotheses Number of RANSAC iterations.
// * @param inlierThreshold Inlier threshold for RANSAC in px.
// * @param wLossRot Weight of the rotation loss term.
// * @param wLossTrans Weight of the translation loss term.
// * @param softClamp Use sqrt of pose loss after this threshold.
// * @param inlierAlpha Alpha parameter for soft inlier counting.
// * @param maxDistError Clamp pixel distance error with this value.
// * @param randomSeed External random seed to make sure we draw different samples across calls of this function.
// * @return DSAC expectation of the pose loss.
// */
//
//double dsacstar_rgbd_backward(
//	at::Tensor sceneCoordinatesSrc,
//	at::Tensor cameraCoordinatesSrc,
//	at::Tensor outSceneCoordinatesGradSrc,
//	at::Tensor gtPoseSrc,
//	int ransacHypotheses,
//	float inlierThreshold,
//	float wLossRot,
//	float wLossTrans,
//	float softClamp,
//	float inlierAlpha,
//	float maxDistError,
//	int randomSeed)
//{
//	ThreadRand::init(randomSeed);
//
//	// access to tensor objects
//	dsacstar::coord_t sceneCoordinates =
//		sceneCoordinatesSrc.accessor<float, 4>();
//
//	dsacstar::coord_t cameraCoordinates =
//		cameraCoordinatesSrc.accessor<float, 4>();
//
//	dsacstar::coord_t sceneCoordinateGrads =
//		outSceneCoordinatesGradSrc.accessor<float, 4>();
//
//	//convert ground truth pose type
//	dsacstar::trans_t gtTrans(4, 4);
//	auto gtPose = gtPoseSrc.accessor<float, 2>();
//
//	for(unsigned x = 0; x < 4; x++)
//	for(unsigned y = 0; y < 4; y++)
//		gtTrans(y, x) = gtPose[y][x];
//
//	// dimensions of scene coordinate predictions
//	int imH = sceneCoordinates.size(2);
//	int imW = sceneCoordinates.size(3);
//
//	// collect all points with valid camera coordinate (ie valid depth measurement)
//	std::vector<cv::Point2i> validPts;
//	for(int x = 0; x < imW; x++)
//	for(int y = 0; y < imH; y++)
//	{
//		if(	cameraCoordinates[0][0][y][x] != 0 &&
//			cameraCoordinates[0][0][y][x] != 0 &&
//			cameraCoordinates[0][0][y][x] != 0)
//			validPts.push_back(cv::Point2i(x, y));
//	}
//
//	std::cout << "Valid points: " << validPts.size() << std::endl;
//
//
//	std::cout << BLUETEXT("Sampling " << ransacHypotheses << " hypotheses.") << std::endl;
//	StopWatch stopW;
//
//	// sample RANSAC hypotheses
//	std::vector<dsacstar::pose_t> initHyps;
//	std::vector<std::vector<cv::Point2i>> sampledPoints;
//	std::vector<std::vector<cv::Point3f>> eyePts;
//	std::vector<std::vector<cv::Point3f>> objPts;
//
//	dsacstar::sampleHypothesesRGBD(
//		sceneCoordinates,
//		cameraCoordinates,
//		validPts,
//		ransacHypotheses,
//		MAX_HYPOTHESES_TRIES,
//		inlierThreshold,
//		initHyps,
//		sampledPoints,
//		eyePts,
//		objPts);
//
//	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//	std::cout << BLUETEXT("Calculating scores.") << std::endl;
//
//	// compute distance error images
//	std::vector<cv::Mat_<float>> distErrs(ransacHypotheses);
//
//	#pragma omp parallel for
//	for(unsigned h = 0; h < initHyps.size(); h++)
//    	distErrs[h] = dsacstar::get3DDistErrs(
//		initHyps[h],
//		sceneCoordinates,
//		cameraCoordinates,
//		validPts,
//		maxDistError);
//
//    // soft inlier counting
//	std::vector<double> scores = dsacstar::getHypScores(
//    	distErrs,
//    	inlierThreshold,
//    	inlierAlpha);
//
//	// apply soft max to scores to get a distribution
//	std::vector<double> hypProbs = dsacstar::softMax(scores);
//	double hypEntropy = dsacstar::entropy(hypProbs); // measure distribution entropy
//	std::cout << "Entropy: " << hypEntropy << std::endl;
//
//	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//	std::cout << BLUETEXT("Refining poses:") << std::endl;
//
//	// collect inliers and refine poses
//	std::vector<dsacstar::pose_t> refHyps(ransacHypotheses);
//	std::vector<cv::Mat_<int>> inlierMaps(refHyps.size());
//
//	#pragma omp parallel for
//	for(unsigned h = 0; h < refHyps.size(); h++)
//	{
//		refHyps[h].first = initHyps[h].first.clone();
//		refHyps[h].second = initHyps[h].second.clone();
//
//		if(hypProbs[h] < PROB_THRESH) continue; // save computation when little influence on expectation
//
//		dsacstar::refineHypRGBD(
//			sceneCoordinates,
//			cameraCoordinates,
//			distErrs[h],
//			validPts,
//			inlierThreshold,
//			MAX_REF_STEPS,
//			maxDistError,
//			refHyps[h],
//			inlierMaps[h]);
//	}
//
//	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//
//	// calculate expected pose loss
//	double expectedLoss = 0;
//	std::vector<double> losses(refHyps.size());
//
//	for(unsigned h = 0; h < refHyps.size(); h++)
//	{
//		dsacstar::trans_t estTrans = dsacstar::pose2trans(refHyps[h]);
//		losses[h] = dsacstar::loss(estTrans, gtTrans, wLossRot, wLossTrans, softClamp);
//		expectedLoss += hypProbs[h] * losses[h];
//	}
//
//   	// === doing the backward pass ====================================================================
//
//	// acumulate hypotheses gradients for patches
//	cv::Mat_<double> dLoss_dObj = cv::Mat_<double>::zeros(imH * imW, 3);
//
//    // --- path I, hypothesis path --------------------------------------------------------------------
//    std::cout << BLUETEXT("Calculating gradients wrt hypotheses.") << std::endl;
//
//    // precalculate gradients per of hypotheis wrt object coordinates
//    std::vector<cv::Mat_<double>> dHyp_dObjs(refHyps.size());
//
//    #pragma omp parallel for
//    for(unsigned h = 0; h < refHyps.size(); h++)
//    {
//		int batchIdx = 0; // only batch size=1 supported atm
//
//        // differentiate refinement around optimum found in last optimization iteration
//        dHyp_dObjs[h] = cv::Mat_<double>::zeros(6, imH * imW * 3);
//
//        if(hypProbs[h] < PROB_THRESH) continue; // skip hypothesis with no impact on expectation
//
//		// collect inlier correspondences of last refinement iteration
//		std::vector<cv::Point3f> eyePts;
//		std::vector<cv::Point2i> srcPts;
//		std::vector<cv::Point3f> objPts;
//
//		for(int x = 0; x < inlierMaps[h].cols; x++)
//		for(int y = 0; y < inlierMaps[h].rows; y++)
//		{
//			if(inlierMaps[h](y, x))
//			{
//				srcPts.push_back(cv::Point2i(x, y));
//				objPts.push_back(cv::Point3f(
//					sceneCoordinates[batchIdx][0][y][x],
//					sceneCoordinates[batchIdx][1][y][x],
//					sceneCoordinates[batchIdx][2][y][x]));
//				eyePts.push_back(cv::Point3f(
//					cameraCoordinates[batchIdx][0][y][x],
//					cameraCoordinates[batchIdx][1][y][x],
//					cameraCoordinates[batchIdx][2][y][x]));
//			}
//		}
//
//        if(eyePts.size() < 3)
//            continue;
//
//        // calculate gradients of final inlier set
//        dsacstar::pose_t cvHypDummy;
//        cv::Mat_<double> dHdO;
//        kabsch(eyePts, objPts, cvHypDummy, dHdO);
//        if (dHdO.empty())
//        	dKabschFD(eyePts, objPts, dHdO);
//
//        for(unsigned ptIdx = 0; ptIdx < objPts.size(); ptIdx++)
//        {
//            int dIdx = srcPts[ptIdx].y * imW * 3 + srcPts[ptIdx].x * 3;
//            dHdO.colRange(ptIdx*3, ptIdx*3+3).copyTo(dHyp_dObjs[h].colRange(dIdx, dIdx + 3));
//        }
//    }
//
//    // combine gradients per hypothesis
//    std::vector<cv::Mat_<double>> gradients(refHyps.size());
//    dsacstar::pose_t hypGT = dsacstar::trans2pose(gtTrans);
//
//    #pragma omp parallel for
//    for(unsigned h = 0; h < refHyps.size(); h++)
//    {
//		if(hypProbs[h] < PROB_THRESH) continue;
//
//        cv::Mat_<double> dLoss_dHyp = dsacstar::dLoss(refHyps[h], hypGT, wLossRot, wLossTrans, softClamp);
//        gradients[h] = dLoss_dHyp * dHyp_dObjs[h];
//    }
//
//	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//
//    // --- path II, score path --------------------------------------------------------------------
//
//    std::cout << BLUETEXT("Calculating gradients wrt scores.") << std::endl;
//
//    std::vector<cv::Mat_<double>> dLoss_dScore_dObjs = dsacstar::dSMScoreRGBD(
//    	sceneCoordinates,
//    	cameraCoordinates,
//    	validPts,
//    	sampledPoints,
//    	losses,
//    	hypProbs,
//    	initHyps,
//    	distErrs,
//    	inlierAlpha,
//    	inlierThreshold,
//    	maxDistError);
//
//    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//
//    // assemble full gradient tensor
//    for(unsigned h = 0; h < refHyps.size(); h++)
//    {
//		if(hypProbs[h] < PROB_THRESH) continue;
//		int batchIdx = 0; // only batch size=1 supported atm
//
//	    for(int idx = 0; idx < imH * imW; idx++)
//	    {
//	    	int x = idx % imW;
//	    	int y = idx / imW;
//
//	        sceneCoordinateGrads[batchIdx][0][y][x] +=
//	        	hypProbs[h] * gradients[h](idx * 3 + 0) + dLoss_dScore_dObjs[h](idx, 0);
//	        sceneCoordinateGrads[batchIdx][1][y][x] +=
//	        	hypProbs[h] * gradients[h](idx * 3 + 1) + dLoss_dScore_dObjs[h](idx, 1);
//	        sceneCoordinateGrads[batchIdx][2][y][x] +=
//	        	hypProbs[h] * gradients[h](idx * 3 + 2) + dLoss_dScore_dObjs[h](idx, 2);
//	    }
//	}
//
//	return expectedLoss;
//}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward_rgb", &dsacstar_rgb_forward, "DSAC* forward (RGB)");
//	m.def("backward_rgb", &dsacstar_rgb_backward, "DSAC* backward (RGB)");
//	m.def("forward_rgbd", &dsacstar_rgbd_forward, "DSAC* forward (RGB-D)");
//	m.def("backward_rgbd", &dsacstar_rgbd_backward, "DSAC* backward (RGB-D)");
}
