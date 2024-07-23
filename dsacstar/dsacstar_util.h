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

#include <omp.h>
#include "thread_rand.h"
//#include "dsacstar_util_rgbd.h"

// makros for coloring console output
#define GREENTEXT(output) "\x1b[32;1m" << output << "\x1b[0m"
#define REDTEXT(output) "\x1b[31;1m" << output << "\x1b[0m"
#define BLUETEXT(output) "\x1b[34;1m" << output << "\x1b[0m"
#define YELLOWTEXT(output) "\x1b[33;1m" << output << "\x1b[0m"

#define EPS 0.00000001
#define PI 3.1415926

namespace dsacstar
{
	/**
	* @brief Calculate original image positions of a scene coordinate prediction.
	* @param outW Width of the scene coordinate prediction.
	* @param outH Height of the scene coordinate prediction.
	* @param subSampling Sub-sampling of the scene coordinate prediction wrt. to the input image.
	* @param shiftX Horizontal offset in case the input image has been shifted before scene coordinare prediction.
	* @param shiftY Vertical offset in case the input image has been shifted before scene coordinare prediction.
	* @return Matrix where each entry contains the original 2D image position.
	*/
	cv::Mat_<cv::Point2i> createSampling(
		unsigned outW, unsigned outH, 
		int subSampling, 
		int shiftX, int shiftY)
	{
		cv::Mat_<cv::Point2i> sampling(outH, outW);

		#pragma omp parallel for
		for(unsigned x = 0; x < outW; x++)
		for(unsigned y = 0; y < outH; y++)
		{
			sampling(y, x) = cv::Point2i(
				x * subSampling + subSampling / 2 - shiftX,
				y * subSampling + subSampling / 2 - shiftY);
		}

		return sampling;
	}

	/**
	* @brief Wrapper for OpenCV solvePnP.
	* Properly handles empty pose inputs.
	* @param objPts List of 3D scene points.
	* @param imgPts List of corresponding 2D image points.
	* @param camMat Internal calibration matrix of the camera.
	* @param distCoeffs Distortion coefficients.
	* @param rot Camera rotation (input/output), axis-angle representation.
	* @param trans Camera translation.
	* @param extrinsicGuess Whether rot and trans already contain an pose estimate.
	* @param methodFlag OpenCV PnP method flag.
	* @return True if pose estimation succeeded.
	*/
	inline bool safeSolvePnP(
		const std::vector<cv::Point3f>& objPts,
		const std::vector<cv::Point2f>& imgPts,
		const cv::Mat& camMat,
		const cv::Mat& distCoeffs,
		cv::Mat& rot,
		cv::Mat& trans,
		bool extrinsicGuess,
		int methodFlag)
	{
		if(rot.type() == 0) rot = cv::Mat_<double>::zeros(1, 3);
		if(trans.type() == 0) trans= cv::Mat_<double>::zeros(1, 3);

		if(!cv::solvePnP(
			objPts, 
			imgPts, 
			camMat, 
			distCoeffs, 
			rot, 
			trans, 
			extrinsicGuess,
			methodFlag))
		{
			rot = cv::Mat_<double>::zeros(3, 1);
			trans = cv::Mat_<double>::zeros(3, 1);
			return false;
		}

		return true;
	}

	/**
	* @brief Samples a set of RANSAC camera pose hypotheses using PnP
	* @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
	* @param sampling Contains original image coordinate for each scene coordinate predicted.
	* @param camMat Camera calibration matrix.
	* @param ransacHypotheses RANSAC iterations.
	* @param maxTries Repeat sampling an hypothesis if it is invalid
	* @param inlierThreshold RANSAC inlier threshold in px.
	* @param hypotheses (output parameter) List of sampled pose hypotheses.
	* @param sampledPoints (output parameter) Corresponding minimal set for each hypotheses, scene coordinate indices.
	* @param imgPts (output parameter) Corresponding minimal set for each hypotheses, 2D image coordinates.
	* @param objPts (output parameter) Corresponding minimal set for each hypotheses, 3D scene coordinates.
	*/
	inline void sampleHypotheses(
		dsacstar::coord_t& sceneCoordinates,
		const cv::Mat_<cv::Point2i>& sampling,
		const cv::Mat_<float>& camMat,
		int ransacHypotheses,
		unsigned maxTries,
		float inlierThreshold,
		std::vector<dsacstar::pose_t>& hypotheses,
		std::vector<std::vector<cv::Point2i>>& sampledPoints,     
		std::vector<std::vector<cv::Point2f>>& imgPts,
		std::vector<std::vector<cv::Point3f>>& objPts)
	{
		int imH = sceneCoordinates.size(2);
		int imW = sceneCoordinates.size(3);

		// keep track of the points each hypothesis is sampled from
		sampledPoints.resize(ransacHypotheses);     
		imgPts.resize(ransacHypotheses);
		objPts.resize(ransacHypotheses);
		hypotheses.resize(ransacHypotheses);

		// sample hypotheses
		#pragma omp parallel for
		for(unsigned h = 0; h < hypotheses.size(); h++)
		for(unsigned t = 0; t < maxTries; t++)
		{
			int batchIdx = 0; // only batch size=1 supported atm

			std::vector<cv::Point2f> projections;
			imgPts[h].clear();
			objPts[h].clear();
			sampledPoints[h].clear();

			for(int j = 0; j < 4; j++)
			{
				// 2D location in the subsampled image
				int x = irand(0, imW);
				int y = irand(0, imH);

				// 2D location in the original RGB image
				imgPts[h].push_back(sampling(y, x)); 
				// 3D object coordinate
				objPts[h].push_back(cv::Point3f(
					sceneCoordinates[batchIdx][0][y][x],
					sceneCoordinates[batchIdx][1][y][x],
					sceneCoordinates[batchIdx][2][y][x])); 
				// 2D pixel location in the subsampled image
				sampledPoints[h].push_back(cv::Point2i(x, y)); 
			}

			if(!dsacstar::safeSolvePnP(
				objPts[h], 
				imgPts[h], 
				camMat, 
				cv::Mat(), 
				hypotheses[h].first, 
				hypotheses[h].second, 
				false, 
				cv::SOLVEPNP_P3P))
			{
				continue;
			}

			// check reconstruction, 4 sampled points should be reconstructed perfectly
			cv::projectPoints(
				objPts[h], 
				hypotheses[h].first, 
				hypotheses[h].second, 
				camMat, 
				cv::Mat(), 
				projections);

			bool foundOutlier = false;
			for(unsigned j = 0; j < imgPts[h].size(); j++)
			{
				if(cv::norm(imgPts[h][j] - projections[j]) < inlierThreshold)
					continue;
				foundOutlier = true;
				break;
			}

			if(foundOutlier)
				continue;
			else
				break;			
		}		
	}

//	/**
//	* @brief Samples a set of RANSAC camera pose hypotheses using Kabsch
//	* @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
//	* @param camera coordinates Camera coordinates calculated from measured depth, same format and size as scene coordinates.
//	* @param validPts A list of valid 2D image positions where camera coordinates / measured depth exists.
//	* @param ransacHypotheses RANSAC iterations.
//	* @param maxTries Repeat sampling an hypothesis if it is invalid
//	* @param inlierThreshold RANSAC inlier threshold in px.
//	* @param hypotheses (output parameter) List of sampled pose hypotheses.
//	* @param sampledPoints (output parameter) Corresponding minimal set for each hypotheses, scene coordinate indices.
//	* @param eyePts (output parameter) Corresponding minimal set for each hypotheses, 3D camera coordinates.
//	* @param objPts (output parameter) Corresponding minimal set for each hypotheses, 3D scene coordinates.
//	*/
//	inline void sampleHypothesesRGBD(
//		dsacstar::coord_t& sceneCoordinates,
//		dsacstar::coord_t& cameraCoordinates,
//		const std::vector<cv::Point2i>& validPts,
//		int ransacHypotheses,
//		unsigned maxTries,
//		float inlierThreshold,
//		std::vector<dsacstar::pose_t>& hypotheses,
//		std::vector<std::vector<cv::Point2i>>& sampledPoints,
//		std::vector<std::vector<cv::Point3f>>& eyePts,
//		std::vector<std::vector<cv::Point3f>>& objPts)
//	{
//		// keep track of the points each hypothesis is sampled from
//		sampledPoints.resize(ransacHypotheses);
//		eyePts.resize(ransacHypotheses);
//		objPts.resize(ransacHypotheses);
//		hypotheses.resize(ransacHypotheses);
//
//		// sample hypotheses
//		#pragma omp parallel for
//		for(unsigned h = 0; h < hypotheses.size(); h++)
//		for(unsigned t = 0; t < maxTries; t++)
//		{
//			int batchIdx = 0; // only batch size=1 supported atm
//
//			std::vector<cv::Point3f> dists;
//			eyePts[h].clear();
//			objPts[h].clear();
//			sampledPoints[h].clear();
//
//			for(int j = 0; j < 3; j++)
//			{
//				// 2D location in the subsampled image
//				int ptIdx = irand(0, validPts.size());
//				int x = validPts[ptIdx].x;
//				int y = validPts[ptIdx].y;
//
//				// 3D camera coordinate
//				eyePts[h].push_back(cv::Point3f(
//					cameraCoordinates[batchIdx][0][y][x],
//					cameraCoordinates[batchIdx][1][y][x],
//					cameraCoordinates[batchIdx][2][y][x]));
//				// 3D object (=scene) coordinate
//				objPts[h].push_back(cv::Point3f(
//					sceneCoordinates[batchIdx][0][y][x],
//					sceneCoordinates[batchIdx][1][y][x],
//					sceneCoordinates[batchIdx][2][y][x]));
//				// 2D pixel location in the subsampled image
//				sampledPoints[h].push_back(cv::Point2i(x, y));
//			}
//
//
//			kabsch(eyePts[h], objPts[h], hypotheses[h]);
//			transform(objPts[h], hypotheses[h], dists);
//
//
//			// check reconstruction, 3 sampled points should be reconstructed perfectly
//			bool foundOutlier = false;
//			for(unsigned j = 0; j < eyePts[h].size(); j++)
//			{
//				if(cv::norm(eyePts[h][j] - dists[j])*100 < inlierThreshold) //measure distance in centimeters
//					continue;
//				foundOutlier = true;
//				break;
//			}
//
//			if(foundOutlier)
//				continue;
//			else
//				break;
//		}
//	}

	/**
	* @brief Calculate soft inlier counts.
	* @param reproErrs Image of reprojection error for each pose hypothesis.
	* @param inlierThreshold RANSAC inlier threshold.
	* @param inlierAlpha Alpha parameter for soft inlier counting.
	* @return List of soft inlier counts for each hypothesis.
	*/
	inline std::vector<double> getHypScores(
		const std::vector<cv::Mat_<float>>& reproErrs,
		float inlierThreshold,
		float inlierAlpha)
	{
		std::vector<double> scores(reproErrs.size(), 0);

		// beta parameter for soft inlier counting
		float inlierBeta = 5 / inlierThreshold;

		#pragma omp parallel for
		for(unsigned h = 0; h < reproErrs.size(); h++)
		for(int x = 0; x < reproErrs[h].cols; x++)
		for(int y = 0; y < reproErrs[h].rows; y++)
		{
			double softThreshold = inlierBeta * (reproErrs[h](y, x) - inlierThreshold);
			softThreshold = 1 / (1+std::exp(-softThreshold));
			scores[h] += 1 - softThreshold;
		}

		#pragma omp parallel for
		for(unsigned h = 0; h < reproErrs.size(); h++)
		{
			scores[h] *= inlierAlpha / reproErrs[h].cols / reproErrs[h].rows;
		}

		return scores;
	}

	/**
	* @brief Calculate image of reprojection errors.
	* @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
	* @param hyp Pose hypothesis to calculate the errors for.
	* @param sampling Contains original image coordinate for each scene coordinate predicted.
	* @param camMat Camera calibration matrix.
	* @param maxReproj Reprojection errors are clamped to this maximum value.
	* @param jacobeanHyp Jacobean matrix with derivatives of the 6D pose wrt. the reprojection error (num pts x 6).
	* @param calcJ Whether to calculate the jacobean matrix or not.
	* @return Image of reprojection errors.
	*/
	cv::Mat_<float> getReproErrs(
		dsacstar::coord_t& sceneCoordinates,
		const dsacstar::pose_t& hyp,
		const cv::Mat_<cv::Point2i>& sampling,
		const cv::Mat& camMat,
		float maxReproj,
	  	cv::Mat_<double>& jacobeanHyp,
  		bool calcJ = false)
	{
		int batchIdx = 0; // only batch size=1 supported atm

		cv::Mat_<float> reproErrs = cv::Mat_<float>::zeros(sampling.size());

		std::vector<cv::Point3f> points3D;
		std::vector<cv::Point2f> projections;	
		std::vector<cv::Point2f> points2D;
		std::vector<cv::Point2f> sources2D;

		// collect 2D-3D correspondences
		for(int x = 0; x < sampling.cols; x++)
		for(int y = 0; y < sampling.rows; y++)
		{		
			// get 2D location of the original RGB frame
			cv::Point2f pt2D(sampling(y, x).x, sampling(y, x).y);

			// get associated 3D object coordinate prediction
			points3D.push_back(cv::Point3f(
				sceneCoordinates[batchIdx][0][y][x],
				sceneCoordinates[batchIdx][1][y][x],
				sceneCoordinates[batchIdx][2][y][x]));
			points2D.push_back(pt2D);
			sources2D.push_back(cv::Point2f(x, y));
		}

		if(points3D.empty()) return reproErrs;
	    
	    if(!calcJ)
	    {
			// project object coordinate into the image using the given pose
			cv::projectPoints(
				points3D, 
				hyp.first, 
				hyp.second, 
				camMat, 
				cv::Mat(), 
				projections);
		}
	    else
	    {
	        cv::Mat_<double> projectionsJ;
	        cv::projectPoints(
	        	points3D, 
	        	hyp.first, 
	        	hyp.second, 
	        	camMat, 
	        	cv::Mat(), 
	        	projections, 
	        	projectionsJ);

	        projectionsJ = projectionsJ.colRange(0, 6);

	        //assemble the jacobean of the refinement residuals
	        jacobeanHyp = cv::Mat_<double>::zeros(points2D.size(), 6);
	        cv::Mat_<double> dNdP(1, 2);
	        cv::Mat_<double> dNdH(1, 6);

	        for(unsigned ptIdx = 0; ptIdx < points2D.size(); ptIdx++)
	        {
	            double err = std::max(cv::norm(projections[ptIdx] - points2D[ptIdx]), EPS);
	            if(err > maxReproj)
	                continue;

	            // derivative of norm
	            dNdP(0, 0) = 1 / err * (projections[ptIdx].x - points2D[ptIdx].x);
	            dNdP(0, 1) = 1 / err * (projections[ptIdx].y - points2D[ptIdx].y);

	            dNdH = dNdP * projectionsJ.rowRange(2 * ptIdx, 2 * ptIdx + 2);
	            dNdH.copyTo(jacobeanHyp.row(ptIdx));
	        }
	    }		

		// measure reprojection errors
		for(unsigned p = 0; p < projections.size(); p++)
		{
			cv::Point2f curPt = points2D[p] - projections[p];
			float l = std::min((float) cv::norm(curPt), maxReproj);
			reproErrs(sources2D[p].y, sources2D[p].x) = l;
		}

		return reproErrs;    
	}

//	/**
//	 * @brief Calculate an image of 3D distance errors for between scene coordinates and camera coordinates, given a pose.
//	 * @param hyp Pose estimate.
//	 * @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
//	 * @param camera coordinates Camera coordinates calculated from measured depth, same format and size as scene coordinates.
//	 * @param validPts A list of valid 2D image positions where camera coordinates / measured depth exists.
//	 * @param maxDist Clamp distance error with this value.
//	 * @return Image of reprojectiob errors.
//	 */
//	cv::Mat_<float> get3DDistErrs(
//	  const dsacstar::pose_t& hyp,
//	  const dsacstar::coord_t& sceneCoordinates,
//	  const dsacstar::coord_t& cameraCoordinates,
//	  const std::vector<cv::Point2i>& validPts,
//	  float maxDist)
//	{
//		int imH = sceneCoordinates.size(2);
//		int imW = sceneCoordinates.size(3);
//		int batchIdx = 0;  // only batch size=1 supported atm
//
//	    cv::Mat_<float> distMap = cv::Mat_<float>::ones(imH, imW) * maxDist;
//
//	    std::vector<cv::Point3f> points3D;
//	    std::vector<cv::Point3f> transformed3D;
//	    std::vector<cv::Point3f> pointsCam3D;
//	    std::vector<cv::Point2f> sources2D;
//
//	    // collect 2D-3D correspondences
//	    for(unsigned i = 0; i < validPts.size(); i++)
//	    {
//	    	int x = validPts[i].x;
//	    	int y = validPts[i].y;
//
//			pointsCam3D.push_back(cv::Point3f(
//				cameraCoordinates[batchIdx][0][y][x],
//				cameraCoordinates[batchIdx][1][y][x],
//				cameraCoordinates[batchIdx][2][y][x]));
//
//	 		points3D.push_back(cv::Point3f(
//				sceneCoordinates[batchIdx][0][y][x],
//				sceneCoordinates[batchIdx][1][y][x],
//				sceneCoordinates[batchIdx][2][y][x]));
//	    }
//
//	    if(points3D.empty()) return distMap;
//
//	    // transform scene coordinates to camera coordinates
//	    transform(points3D, hyp, transformed3D);
//
//	    // measure 3D distance
//	    for(unsigned p = 0; p < transformed3D.size(); p++)
//	    {
//			cv::Point3f curPt = pointsCam3D[p] - transformed3D[p];
//			//measure distance in centimeters
//			float l = std::min((float) cv::norm(curPt)*100, maxDist);
//			distMap(validPts[p].y, validPts[p].x) = l;
//	    }
//
//	    return distMap;
//	}


	/**
	* @brief Refine a pose hypothesis by iteratively re-fitting it to all inliers.
	* @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
	* @param reproErrs Original reprojection errors of the pose hypothesis, used to collect the first set of inliers.
	* @param sampling Contains original image coordinate for each scene coordinate predicted.
	* @param camMat Camera calibration matrix.
	* @param inlierThreshold RANSAC inlier threshold.
	* @param maxRefSteps Maximum refinement iterations (re-calculating inlier and refitting).
	* @param maxReproj Reprojection errors are clamped to this maximum value.
	* @param hypothesis (output parameter) Refined pose.
	* @param inlierMap (output parameter) 2D image indicating which scene coordinate are (final) inliers.
	*/
	inline void refineHyp(
		dsacstar::coord_t& sceneCoordinates,
		const cv::Mat_<float>& reproErrs,
		const cv::Mat_<cv::Point2i>& sampling,
		const cv::Mat_<float>& camMat,
		float inlierThreshold,
		unsigned maxRefSteps,
		float maxReproj,
		dsacstar::pose_t& hypothesis,
		cv::Mat_<int>& inlierMap)
	{
		cv::Mat_<float> localReproErrs = reproErrs.clone();
		int batchIdx = 0; // only batch size=1 supported atm

		// refine as long as inlier count increases 
		unsigned bestInliers = 4; 

		// refine current hypothesis
		for(unsigned rStep = 0; rStep < maxRefSteps; rStep++)
		{
			// collect inliers
			std::vector<cv::Point2f> localImgPts;
			std::vector<cv::Point3f> localObjPts; 
			cv::Mat_<int> localInlierMap = cv::Mat_<int>::zeros(localReproErrs.size());

			for(int x = 0; x < sampling.cols; x++)
			for(int y = 0; y < sampling.rows; y++)
			{
				if(localReproErrs(y, x) < inlierThreshold)
				{
					localImgPts.push_back(sampling(y, x));
					localObjPts.push_back(cv::Point3f(
						sceneCoordinates[batchIdx][0][y][x],
						sceneCoordinates[batchIdx][1][y][x],
						sceneCoordinates[batchIdx][2][y][x]));
					localInlierMap(y, x) = 1;
				}
			}

			if(localImgPts.size() <= bestInliers)
				break; // converged
			bestInliers = localImgPts.size();

			// recalculate pose
			dsacstar::pose_t hypUpdate;
			hypUpdate.first = hypothesis.first.clone();
			hypUpdate.second = hypothesis.second.clone();

    		if(!dsacstar::safeSolvePnP(
				localObjPts, 
				localImgPts, 
				camMat, 
				cv::Mat(), 
				hypUpdate.first, 
				hypUpdate.second, 
				true, 
				(localImgPts.size() > 4) ? 
					cv::SOLVEPNP_ITERATIVE : 
					cv::SOLVEPNP_P3P))
                		break; //abort if PnP fails

			hypothesis = hypUpdate;
			inlierMap = localInlierMap;

			// recalculate pose errors
			cv::Mat_<double> jacobeanDummy;

			localReproErrs = dsacstar::getReproErrs(
				sceneCoordinates,
				hypothesis, 
				sampling, 
				camMat,
				maxReproj,
				jacobeanDummy);
		}			
	}

//	/**
//	* @brief Refine a pose hypothesis by iteratively re-fitting it to all inliers (RGB-D version).
//	* @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
//	* @param camera coordinates Camera coordinates calculated from measured depth, same format and size as scene coordinates.
//	* @param distErrs Original 3D distance errors of the pose hypothesis, used to collect the first set of inliers.
// 	* @param validPts A list of valid 2D image positions where camera coordinates / measured depth exists.
//	* @param inlierThreshold RANSAC inlier threshold in centimeters.
//	* @param maxRefSteps Maximum refinement iterations (re-calculating inlier and refitting).
//	* @param maxDist Clamp distance error with this value.
//	* @param hypothesis (output parameter) Refined pose.
//	* @param inlierMap (output parameter) 2D image indicating which scene coordinate are (final) inliers.
//	*/
//	inline void refineHypRGBD(
//		dsacstar::coord_t& sceneCoordinates,
//		dsacstar::coord_t& cameraCoordinates,
//		const cv::Mat_<float>& distErrs,
//		const std::vector<cv::Point2i>& validPts,
//		float inlierThreshold,
//		unsigned maxRefSteps,
//		float maxDist,
//		dsacstar::pose_t& hypothesis,
//		cv::Mat_<int>& inlierMap)
//	{
//		cv::Mat_<float> localDistErrs = distErrs.clone();
//		int batchIdx = 0; // only batch size=1 supported atm
//
//		// refine as long as inlier count increases
//		unsigned bestInliers = 3;
//
//		// refine current hypothesis
//		for(unsigned rStep = 0; rStep < maxRefSteps; rStep++)
//		{
//			// collect inliers
//			std::vector<cv::Point3f> localEyePts;
//			std::vector<cv::Point3f> localObjPts;
//			cv::Mat_<int> localInlierMap = cv::Mat_<int>::zeros(localDistErrs.size());
//
//			for(unsigned ptIdx = 0; ptIdx < validPts.size(); ptIdx++)
//			{
//				int x = validPts[ptIdx].x;
//				int y = validPts[ptIdx].y;
//
//				if(localDistErrs(y, x) < inlierThreshold)
//				{
//					localObjPts.push_back(cv::Point3f(
//						sceneCoordinates[batchIdx][0][y][x],
//						sceneCoordinates[batchIdx][1][y][x],
//						sceneCoordinates[batchIdx][2][y][x]));
//					localEyePts.push_back(cv::Point3f(
//						cameraCoordinates[batchIdx][0][y][x],
//						cameraCoordinates[batchIdx][1][y][x],
//						cameraCoordinates[batchIdx][2][y][x]));
//					localInlierMap(y, x) = 1;
//				}
//			}
//
//			if(localEyePts.size() <= bestInliers)
//				break; // converged
//			bestInliers = localEyePts.size();
//
//			// recalculate pose
//			dsacstar::pose_t hypUpdate;
//			hypUpdate.first = hypothesis.first.clone();
//			hypUpdate.second = hypothesis.second.clone();
//
//			kabsch(localEyePts, localObjPts, hypUpdate);
//
//			hypothesis = hypUpdate;
//			inlierMap = localInlierMap;
//
//			// recalculate pose errors
//			localDistErrs = dsacstar::get3DDistErrs(
//				hypothesis,
//				sceneCoordinates,
//				cameraCoordinates,
//				validPts,
//				maxDist);
//		}
//	}

	/**
	* @brief Applies soft max to the given list of scores.
	* @param scores List of scores.
	* @return Soft max distribution (sums to 1)
	*/
	std::vector<double> softMax(const std::vector<double>& scores)
	{
		double maxScore = 0;
		for(unsigned i = 0; i < scores.size(); i++)
			if(i == 0 || scores[i] > maxScore) maxScore = scores[i];

		std::vector<double> sf(scores.size());
		double sum = 0.0;

		for(unsigned i = 0; i < scores.size(); i++)
		{
			sf[i] = std::exp(scores[i] - maxScore);
			sum += sf[i];
		}
		for(unsigned i = 0; i < scores.size(); i++)
		{
			sf[i] /= sum;
		}

		return sf;
	}

	/**
	* @brief Calculate the Shannon entropy of a discrete distribution.
	* @param dist Discrete distribution. Probability per entry, should sum to 1.
	* @return  Shannon entropy.
	*/
	double entropy(const std::vector<double>& dist)
	{
		double e = 0;
		for(unsigned i = 0; i < dist.size(); i++)
			if(dist[i] > 0)
				e -= dist[i] * std::log2(dist[i]);

		return e;
	}

	/**
	* @brief Sample a hypothesis index.
	* @param probs Selection probabilities.
	* @param training If false, do not sample, but take argmax.
	* @return Hypothesis index.
	*/
	int draw(const std::vector<double>& probs, bool training)
	{
		std::map<double, int> cumProb;
		double probSum = 0;
		double maxProb = -1;
		double maxIdx = 0; 

		for(unsigned idx = 0; idx < probs.size(); idx++)
		{
			if(probs[idx] < EPS) continue;

			probSum += probs[idx];
			cumProb[probSum] = idx;

			if(maxProb < 0 || probs[idx] > maxProb)
			{
				maxProb = probs[idx];
				maxIdx = idx;
			}
		}

		if(training)
			return cumProb.upper_bound(drand(0, probSum))->second;
		else
			return maxIdx;
	}

	/**
	* @brief Transform scene pose (OpenCV format) to camera transformation, related by inversion.
	* @param pose Scene pose in OpenCV format (i.e. axis-angle and translation).
	* @return Camera transformation matrix (4x4).
	*/
	dsacstar::trans_t pose2trans(const dsacstar::pose_t& pose)
	{
		dsacstar::trans_t rot, trans = dsacstar::trans_t::eye(4, 4);
		cv::Rodrigues(pose.first, rot);

		rot.copyTo(trans.rowRange(0,3).colRange(0,3));
		trans(0, 3) = pose.second.at<double>(0, 0);
		trans(1, 3) = pose.second.at<double>(1, 0);
		trans(2, 3) = pose.second.at<double>(2, 0);

		return trans.inv(); // camera transformation is inverted scene pose
	}

	/**
	* @brief Transform camera transformation to scene pose (OpenCV format), related by inversion.
	* @param trans Camera transformation matrix (4x4)
	* @return Scene pose in OpenCV format (i.e. axis-angle and translation).
	*/
	dsacstar::pose_t trans2pose(const dsacstar::trans_t& trans)
	{
		dsacstar::trans_t invTrans = trans.inv();

		dsacstar::pose_t pose;
		cv::Rodrigues(invTrans.colRange(0,3).rowRange(0,3), pose.first);

		pose.second = cv::Mat_<double>(3, 1);
		pose.second.at<double>(0, 0) = invTrans(0, 3);
		pose.second.at<double>(1, 0) = invTrans(1, 3);
		pose.second.at<double>(2, 0) = invTrans(2, 3);

		return pose; // camera transformation is inverted scene pose
	}

	/**
	 * @brief Calculate the average of all matrix entries.
	 * @param mat Input matrix.
	 * @return Average of entries.
	 */
	double getAvg(const cv::Mat_<double>& mat)
	{
	    double avg = 0;
	    int count = 0;
	    
	    for(int x = 0; x < mat.cols; x++)
	    for(int y = 0; y < mat.rows; y++)
	    {
	    	double entry = std::abs(mat(y, x));
			if(entry > EPS)
			{
				avg += entry;
				count++;
			}
	    }
	    
	    return avg / (EPS + count);
	}

	/**
	 * @brief Return the maximum entry of the given matrix.
	 * @param mat Input matrix.
	 * @return Maximum entry.
	 */
	double getMax(const cv::Mat_<double>& mat)
	{
	    double m = -1;
	    
	    for(int x = 0; x < mat.cols; x++)
	    for(int y = 0; y < mat.rows; y++)
	    {
			double val = std::abs(mat(y, x));
			if(m < 0 || val > m)
			  m = val;
	    }
	    
	    return m;
	}

	/**
	 * @brief Return the median of all entries of the given matrix.
	 * @param mat Input matrix.
	 * @return Median entry.
	 */
	double getMed(const cv::Mat_<double>& mat)
	{
	    std::vector<double> vals;
	    
	    for(int x = 0; x < mat.cols; x++)
	    for(int y = 0; y < mat.rows; y++)
	    {
	    	double entry = std::abs(mat(y, x));
	    	if(entry > EPS) vals.push_back(entry);
	    }

	    if(vals.empty()) 
	    	return 0;

	    std::sort(vals.begin(), vals.end());
	    
	    return vals[vals.size() / 2];
	}	
}
