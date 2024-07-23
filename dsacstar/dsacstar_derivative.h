/*
Based on the DSAC++ and ESAC code.
https://github.com/vislearn/LessMore
https://github.com/vislearn/esac


Copyright (c) 2016, TU Dresden
Copyright (c) 2010, Heidelberg University
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

#define PROB_THRESH 0.001 // ignore hypotheses with low probability for expectations

namespace dsacstar
{
	/**
	* @brief Calculates the Jacobean of the projection function w.r.t the given 3D point, ie. the function has the form 3 -> 1
	* @param pt Ground truth 2D location.
	* @param obj 3D point.
	* @param rot Rotation in axis-angle format (OpenCV convention)
	* @param trans Translation vector (OpenCV convention).
	* @param camMat Calibration matrix of the camera.
	* @param maxReproj Reprojection errors are clamped to this maximum value.
	* @return 1x3 Jacobean matrix of partial derivatives.
	*/
	cv::Mat_<double> dProjectdObj(
		const cv::Point2f& pt, 
		const cv::Point3f& obj, 
		const cv::Mat& rot, 
		const cv::Mat& trans, 
		const cv::Mat& camMat, 
		float maxReproErr)
	{
	    double f = camMat.at<float>(0, 0);
	    double ppx = camMat.at<float>(0, 2);
	    double ppy = camMat.at<float>(1, 2);

	    //transform point
	    cv::Mat objMat = cv::Mat(obj);
	    objMat.convertTo(objMat, CV_64F);

	    objMat = rot * objMat + trans;

	    if(std::abs(objMat.at<double>(2, 0)) < EPS) // prevent division by zero
	        return cv::Mat_<double>::zeros(1, 3);

	    // project
	    double px = f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) + ppx;
	    double py = f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) + ppy;

	    // calculate error
	    double err = std::sqrt((pt.x - px) * (pt.x - px) + (pt.y - py) * (pt.y - py));

	    // early out if projection error is above threshold
	    if(err > maxReproErr)
	        return cv::Mat_<double>::zeros(1, 3);

	    err += EPS; // avoid dividing by zero

	    // derivative in x direction of obj coordinate
	    double pxdx = f * rot.at<double>(0, 0) / objMat.at<double>(2, 0) - f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 0);
	    double pydx = f * rot.at<double>(1, 0) / objMat.at<double>(2, 0) - f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 0);
	    double dx = 0.5 / err * (2 * (pt.x - px) * -pxdx + 2 * (pt.y - py) * -pydx);

	    // derivative in y direction of obj coordinate
	    double pxdy = f * rot.at<double>(0, 1) / objMat.at<double>(2, 0) - f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 1);
	    double pydy = f * rot.at<double>(1, 1) / objMat.at<double>(2, 0) - f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 1);
	    double dy = 0.5 / err * (2 * (pt.x - px) * -pxdy + 2 * (pt.y - py) * -pydy);

	    // derivative in z direction of obj coordinate
	    double pxdz = f * rot.at<double>(0, 2) / objMat.at<double>(2, 0) - f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 2);
	    double pydz = f * rot.at<double>(1, 2) / objMat.at<double>(2, 0) - f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 2);
	    double dz = 0.5 / err * (2 * (pt.x - px) * -pxdz + 2 * (pt.y - py) * -pydz);

	    cv::Mat_<double> jacobean(1, 3);
	    jacobean(0, 0) = dx;
	    jacobean(0, 1) = dy;
	    jacobean(0, 2) = dz;

	    return jacobean;
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

	/**
	 * @brief Calculates the Jacobean of the PNP function w.r.t. the scene coordinate inputs.
	 *
	 * PNP is treated as a n x 3 -> 6 fnuction, i.e. it takes n 3D coordinates and maps them to a 6D pose.
	 * The Jacobean is therefore 6x3n. 
	 * The Jacobean is calculated using central differences, and hence only suitable for small point sets.
	 * For gradients of large points sets, we use an analytical approximaten, see the backward function in dsacstar.cpp.
	 *
	 * @param imgPts List of 2D points.
	 * @param objPts List of corresponding 3D points.
	 * @param camMat Camera calibration matrix.
	 * @param eps Step size for central differences.
	 * @return 6x3n Jacobean matrix of partial derivatives.
	 */
	cv::Mat_<double> dPNP(    
	    const std::vector<cv::Point2f>& imgPts,
	    std::vector<cv::Point3f> objPts,
	    const cv::Mat& camMat,
	    float eps = 0.001f)
	{

	    int pnpMethod = (imgPts.size() == 4) ? cv::SOLVEPNP_P3P : cv::SOLVEPNP_ITERATIVE;

	    //in case of P3P the 4th point is needed to resolve ambiguities, its derivative is zero
	    int effectiveObjPoints = (pnpMethod == cv::SOLVEPNP_P3P) ? 3 : objPts.size();

	    cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(6, objPts.size() * 3);
	    bool success;
	    
	    // central differences
	    for(int i = 0; i < effectiveObjPoints; i++)
	    for(unsigned j = 0; j < 3; j++)
	    {
	        if(j == 0) objPts[i].x += eps;
	        else if(j == 1) objPts[i].y += eps;
	        else if(j == 2) objPts[i].z += eps;

	        // forward step
	        dsacstar::pose_t fStep;
	        success = safeSolvePnP(objPts, imgPts, camMat, cv::Mat(), fStep.first, fStep.second, false, pnpMethod);

	        if(!success)
	            return cv::Mat_<double>::zeros(6, objPts.size() * 3);

	        if(j == 0) objPts[i].x -= 2 * eps;
	        else if(j == 1) objPts[i].y -= 2 * eps;
	        else if(j == 2) objPts[i].z -= 2 * eps;

	        // backward step
	        dsacstar::pose_t bStep;
	        success = safeSolvePnP(objPts, imgPts, camMat, cv::Mat(), bStep.first, bStep.second, false, pnpMethod);

	        if(!success)
	            return cv::Mat_<double>::zeros(6, objPts.size() * 3);

	        if(j == 0) objPts[i].x += eps;
	        else if(j == 1) objPts[i].y += eps;
	        else if(j == 2) objPts[i].z += eps;

	        // gradient calculation
	        fStep.first = (fStep.first - bStep.first) / (2 * eps);
	        fStep.second = (fStep.second - bStep.second) / (2 * eps);

	        fStep.first.copyTo(jacobean.col(i * 3 + j).rowRange(0, 3));
	        fStep.second.copyTo(jacobean.col(i * 3 + j).rowRange(3, 6));

	        if(containsNaNs(jacobean.col(i * 3 + j)))
	            return cv::Mat_<double>::zeros(6, objPts.size() * 3);
	    }

	    return jacobean;
	}

	/**
	 * @brief Calculates the Jacobean matrix of the function that maps n estimated scene coordinates to a score, ie. the function has the form n x 3 -> 1. Returns one Jacobean matrix per hypothesis.
	 * @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
	 * @param sampling Contains original image coordinate for each scene coordinate predicted.
	 * @param sampledPoints Corresponding minimal set for each hypotheses as scene coordinate indices.
	 * @param jacobeansScore (output paramter) List of Jacobean matrices. One 1 x 3n matrix per pose hypothesis.
	 * @param scoreOutputGradients Gradients w.r.t the score i.e. the gradients of the loss up to the soft inlier count.
	 * @param hyps List of RANSAC hypotheses.
	 * @param reproErrs Image of reprojection error for each pose hypothesis.
	 * @param jacobeanHyps List of jacobean matrices with derivatives of the 6D pose wrt. the reprojection errors.
	 * @param hypProbs Selection probabilities over all hypotheses.
	 * @param camMat Camera calibration matrix.
	 * @param inlierAlpha Alpha parameter for soft inlier counting.
	 * @param inlierThreshold RANSAC inlier threshold.
	 * @param maxReproj Reprojection errors are clamped to this maximum value.
	 */
	void dScore(
	    dsacstar::coord_t& sceneCoordinates,
	    const cv::Mat_<cv::Point2i>& sampling,
	    const std::vector<std::vector<cv::Point2i>>& sampledPoints,
	    std::vector<cv::Mat_<double>>& jacobeansScore,
	    const std::vector<double>& scoreOutputGradients,
	    const std::vector<dsacstar::pose_t>& hyps,
	    const std::vector<cv::Mat_<float>>& reproErrs,
	    const std::vector<cv::Mat_<double>>& jacobeansHyps,
	    const std::vector<double>& hypProbs,
	    const cv::Mat& camMat,
	    float inlierAlpha,
	    float inlierThreshold,
	    float maxReproErr)
	{
	    int hypCount = sampledPoints.size();
	    // beta parameter for soft inlier counting.	
	    float inlierBeta = 5 / inlierThreshold; 
	    
	    // collect 2d-3D correspondences
	    std::vector<std::vector<cv::Point2f>> imgPts(hypCount);
	    std::vector<std::vector<cv::Point3f>> objPts(hypCount);
	    
	    #pragma omp parallel for
	    for(int h = 0; h < hypCount; h++)
	    {
	    	if(hypProbs[h] < PROB_THRESH) continue;

			int batchIdx = 0; // ony batch size = 1 supported atm

	        for(unsigned i = 0; i < sampledPoints[h].size(); i++)
	        {
	            int x = sampledPoints[h][i].x;
	            int y = sampledPoints[h][i].y;
		  
	            imgPts[h].push_back(sampling(y, x));
	            objPts[h].push_back(cv::Point3f(
					sceneCoordinates[batchIdx][0][y][x],
					sceneCoordinates[batchIdx][1][y][x],
					sceneCoordinates[batchIdx][2][y][x]));
	        }
	    }
	    
	    // derivatives of the soft inlier scores
	    std::vector<cv::Mat_<double>> dReproErrs(reproErrs.size());

	    #pragma omp parallel for
	    for(int h = 0; h < hypCount; h++)
	    {
	        if(hypProbs[h] < PROB_THRESH) continue;
			
	        dReproErrs[h] = cv::Mat_<double>::zeros(reproErrs[h].size());

			for(int x = 0; x < sampling.cols; x++)
			for(int y = 0; y < sampling.rows; y++)
			{
	            double softThreshold = inlierBeta * (reproErrs[h](y, x) - inlierThreshold);
	            softThreshold = 1 / (1+std::exp(-softThreshold));
	            dReproErrs[h](y, x) = -softThreshold * (1 - softThreshold) * inlierBeta * scoreOutputGradients[h];
	        }

	        dReproErrs[h] *= inlierAlpha  / dReproErrs[h].cols / dReproErrs[h].rows;
	    }

	    jacobeansScore.resize(hypCount);

	    // derivative of the loss wrt the score
	    #pragma omp parallel for
	    for(int h = 0; h < hypCount; h++)
	    {  
	        cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(1, sampling.cols * sampling.rows * 3);
	        jacobeansScore[h] = jacobean;

			if(hypProbs[h] < PROB_THRESH) continue;

			int batchIdx = 0; // ony batch size = 1 supported atm

	        // accumulate derivate of score wrt the scene coordinates that are used to calculate the pose
	        cv::Mat_<double> supportPointGradients = cv::Mat_<double>::zeros(1, 12);

	        cv::Mat_<double> dHdO = dPNP(imgPts[h], objPts[h], camMat); // 6x12

	        if(dsacstar::getMax(dHdO) > 10) dHdO = 0; // clamping for stability

	        cv::Mat rot;
	        cv::Rodrigues(hyps[h].first, rot);

			for(int x = 0; x < sampling.cols; x++)
			for(int y = 0; y < sampling.rows; y++)
			{
				int ptIdx = x * dReproErrs[h].rows + y;

	            cv::Point2f pt(sampling(y, x).x, sampling(y, x).y);
	            cv::Point3f obj = cv::Point3f(
					sceneCoordinates[batchIdx][0][y][x],
					sceneCoordinates[batchIdx][1][y][x],
					sceneCoordinates[batchIdx][2][y][x]);
		  
	            // account for the direct influence of all scene coordinates in the score
	            cv::Mat_<double> dPdO = dProjectdObj(pt, obj, rot, hyps[h].second, camMat, maxReproErr);
	            dPdO *= dReproErrs[h](y, x);
	            dPdO.copyTo(jacobean.colRange(x * dReproErrs[h].rows * 3 + y * 3, x * dReproErrs[h].rows * 3 + y * 3 + 3));

	            // account for the indirect influence of the scene coorindates that are used to calculate the pose
	            cv::Mat_<double> dPdH = jacobeansHyps[h].row(ptIdx);

	            supportPointGradients += dReproErrs[h](y, x) * dPdH * dHdO;
	        }

	        // add the accumulated derivatives for the scene coordinates that are used to calculate the pose
	        for(unsigned i = 0; i < sampledPoints[h].size(); i++)
	        {
	            unsigned x = sampledPoints[h][i].x;
	            unsigned y = sampledPoints[h][i].y;
		    
	            jacobean.colRange(x * dReproErrs[h].rows * 3 + y * 3, x * dReproErrs[h].rows * 3 + y * 3 + 3) += supportPointGradients.colRange(i * 3, i * 3 + 3);
	        }
	    }
	    
	}

	/**
	 * @brief Calculates the Jacobean matrix of the function that maps n estimated scene coordinates to a soft max score, ie. the function has the form n x 3 -> 1. Returns one Jacobean matrix per hypothesis.
	 *
	 * This is the Soft maxed version of dScore (see above).
	 *
	 * @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
	 * @param sampling Contains original image coordinate for each scene coordinate predicted.
	 * @param sampledPoints Corresponding minimal set for each hypotheses as scene coordinate indices.
	 * @param losses Loss value for each hypothesis.
	 * @param hypProbs Selection probabilities over all hypotheses.
	 * @paran initHyps List of unrefined hypotheses.
	 * @paran initReproErrs List of reprojection error images of unrefined hypotheses.
	 * @param jacobeanHyps List of jacobean matrices with derivatives of the 6D pose wrt. the reprojection errors.
	 * @param camMat Camera calibration matrix.
	 * @param inlierAlpha Alpha parameter for soft inlier counting.
	 * @param inlierThreshold RANSAC inlier threshold.
	 * @param maxReproj Reprojection errors are clamped to this maximum value.	 
	 * @return List of Jacobean matrices. One 1 x 3n matrix per pose hypothesis.
	 */
	std::vector<cv::Mat_<double>> dSMScore(
	    dsacstar::coord_t& sceneCoordinates,
	    const cv::Mat_<cv::Point2i>& sampling,
	    const std::vector<std::vector<cv::Point2i>>& sampledPoints,
	    const std::vector<double>& losses,
	    const std::vector<double>& hypProbs,
	    const std::vector<dsacstar::pose_t>& initHyps,
	    const std::vector<cv::Mat_<float>>& initReproErrs,
	    const std::vector<cv::Mat_<double>>& jacobeansHyps,
	    const cv::Mat& camMat,
	    float inlierAlpha,
	    float inlierThreshold,
	    float maxReproErr)
	{

	    // assemble the gradients wrt the scores, ie the gradients of soft max function
	    std::vector<double> scoreOutputGradients(sampledPoints.size());
	        
	    #pragma omp parallel for
	    for(unsigned i = 0; i < sampledPoints.size(); i++)
	    {
			if(hypProbs[i] < PROB_THRESH) continue;

	        scoreOutputGradients[i] = hypProbs[i] * losses[i];
	        for(unsigned j = 0; j < sampledPoints.size(); j++)
	            scoreOutputGradients[i] -= hypProbs[i] * hypProbs[j] * losses[j];
	    }
	 
	    // calculate gradients of the score function
	    std::vector<cv::Mat_<double>> jacobeansScore;
	    dScore(
	    	sceneCoordinates, 
	    	sampling, 
	    	sampledPoints, 
	    	jacobeansScore, 
	    	scoreOutputGradients, 
	    	initHyps, 
	    	initReproErrs, 
	    	jacobeansHyps, 
	    	hypProbs,
	    	camMat,
	    	inlierAlpha,
	    	inlierThreshold,
	    	maxReproErr);

	    // data conversion
	    #pragma omp parallel for
	    for(unsigned i = 0; i < jacobeansScore.size(); i++)
	    {
	        // reorder to points row first into rows
	        cv::Mat_<double> reformat = cv::Mat_<double>::zeros(sampling.cols * sampling.rows, 3);
		
			if(hypProbs[i] >= PROB_THRESH)
			{
				for(int x = 0; x < sampling.cols; x++)
				for(int y = 0; y < sampling.rows; y++)
				{
		            cv::Mat_<double> patchGrad = jacobeansScore[i].colRange(
		              x * sampling.rows * 3 + y * 3,
		              x * sampling.rows * 3 + y * 3 + 3);
			    
		            patchGrad.copyTo(reformat.row(y * sampling.cols + x));
		        }
			}

	        jacobeansScore[i] = reformat;
	    }
	    
	    return jacobeansScore;
	}

	/**
	 * @brief Calculates the Jacobean of the transform function w.r.t the given 3D point, ie. the function has the form 3 -> 1
	 * @param pt Ground truth 3D location in camera coordinates.
	 * @param obj 3D point.
	 * @param hyp Pose estimate.
	 * @param maxDist Distance errors are clamped to this maximum value.
	 * @return 1x3 Jacobean matrix of partial derivatives.
	 */
	cv::Mat_<double> dTransformdObj(
		const cv::Point3f& pt, 
		const cv::Point3f& obj, 
		const dsacstar::pose_t& hyp,
		float maxDist)
	{
	    //transform point
	    cv::Mat objMat = cv::Mat(obj);
	    objMat.convertTo(objMat, CV_64F);

	    cv::Mat rot;
	    cv::Rodrigues(hyp.first, rot);

	    objMat = rot * objMat + hyp.second;

	    cv::Point3d objPt(objMat.at<double>(0, 0), objMat.at<double>(1, 0), objMat.at<double>(2, 0));

	    // calculate error
	    double err = std::sqrt((pt.x - objPt.x) * (pt.x - objPt.x) + (pt.y - objPt.y) * (pt.y - objPt.y) + (pt.z - objPt.z) * (pt.z - objPt.z));

	    // early out if projection error is above threshold
	    if(err*100 > maxDist)
			return cv::Mat_<double>::zeros(1, 3);

	    err += EPS; // avoid dividing by zero

	    // derivative in x direction of obj coordinate
	    double dx = 0.5 / err * (2 * (pt.x - objPt.x) * -rot.at<double>(0, 0) + 2 * (pt.y - objPt.y) * -rot.at<double>(1, 0) + 2 * (pt.z - objPt.z) * -rot.at<double>(2, 0));

	    // derivative in x direction of obj coordinate
	    double dy = 0.5 / err * (2 * (pt.x - objPt.x) * -rot.at<double>(0, 1) + 2 * (pt.y - objPt.y) * -rot.at<double>(1, 1) + 2 * (pt.z - objPt.z) * -rot.at<double>(2, 1));

	    // derivative in x direction of obj coordinate
	    double dz = 0.5 / err * (2 * (pt.x - objPt.x) * -rot.at<double>(0, 2) + 2 * (pt.y - objPt.y) * -rot.at<double>(1, 2) + 2 * (pt.z - objPt.z) * -rot.at<double>(2, 2));

	    cv::Mat_<double> jacobean(1, 3);
	    jacobean(0, 0) = dx;
	    jacobean(0, 1) = dy;
	    jacobean(0, 2) = dz;

	    return jacobean;
	}

/**
 * @brief Calculates the Jacobean of the transform function w.r.t the given 6D pose, ie. the function has the form 6 -> 1
 * @param pt Ground truth 3D location in camera coordinate.
 * @param obj 3D point.
 * @param hyp Pose estimate.
 * @param maxDist Distance errors are clamped to this maximum value.
 * @return 1x6 Jacobean matrix of partial derivatives.
 */
cv::Mat_<double> dTransformdHyp(
	const cv::Point3f& pt, 
	const cv::Point3f& obj, 
	const dsacstar::pose_t& hyp,
	float maxDist)
{
    //transform point
    cv::Mat objMat = cv::Mat(obj);
    objMat.convertTo(objMat, CV_64F);

    cv::Mat rot, dRdH;
    cv::Rodrigues(hyp.first, rot, dRdH);
    dRdH = dRdH.t();

    cv::Mat eyeMat = rot * objMat + hyp.second;

    cv::Point3d eyePt(eyeMat.at<double>(0, 0), eyeMat.at<double>(1, 0), eyeMat.at<double>(2, 0));

	// calculate error
	double err = std::sqrt((pt.x - eyePt.x) * (pt.x - eyePt.x) + (pt.y - eyePt.y) * (pt.y - eyePt.y) + (pt.z - eyePt.z) * (pt.z - eyePt.z));

    // early out if projection error is above threshold
    if(err * 100 > maxDist)
    	return cv::Mat_<double>::zeros(1, 6);

    err += EPS; // avoid dividing by zero

    // derivative of the error wrt to transformation
    cv::Mat_<double> dNdTf = cv::Mat_<double>::zeros(1, 3);
    dNdTf(0, 0) = -1 / err * (pt.x - eyePt.x);
    dNdTf(0, 1) = -1 / err * (pt.y - eyePt.y);
    dNdTf(0, 2) = -1 / err * (pt.z - eyePt.z);

    // derivative of transformation function wrt rotation matrix
    cv::Mat_<double> dTfdR = cv::Mat_<double>::zeros(3, 9);
    dTfdR.row(0).colRange(0, 3) = objMat.t();
    dTfdR.row(1).colRange(3, 6) = objMat.t();
    dTfdR.row(2).colRange(6, 9) = objMat.t();

    // combined derivative of the error wrt the rodriguez vector
    cv::Mat_<double> dNdH = dNdTf * dTfdR * dRdH;

    // derivative of transformation wrt the translation vector
    cv::Mat_<double> dTfdT = cv::Mat_<double>::eye(3, 3);

    // combined derivative of error wrt the translation vector
    cv::Mat_<double> dNdT = dNdTf * dTfdT;

    cv::Mat_<double> jacobean(1, 6);
    dNdH.copyTo(jacobean.colRange(0, 3));
    dNdT.copyTo(jacobean.colRange(3, 6));
    return jacobean;
}

//	/**
//	 * @brief Calculates the Jacobean matrix of the function that maps n estimated scene coordinates to a score, ie. the function has the form n x 3 -> 1. Returns one Jacobean matrix per hypothesis. RGBD version.
//	 * @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
//	 * @param cameraCoordinates Camera coordinates calculated from measured depth, same format and size as scene coordinates.
//	 * @param validPts A list of valid 2D image positions where camera coordinates / measured depth exists.
//	 * @param sampledPoints Corresponding minimal set for each hypotheses as scene coordinate indices.
//	 * @param jacobeansScore (output paramter) List of Jacobean matrices. One 1 x 3n matrix per pose hypothesis.
//	 * @param scoreOutputGradients Gradients w.r.t the score i.e. the gradients of the loss up to the soft inlier count.
//	 * @param hyps List of RANSAC hypotheses.
//	 * @param distErrs Image of 3D distance error for each pose hypothesis.
//	 * @param hypProbs Selection probabilities over all hypotheses.
//	 * @param inlierAlpha Alpha parameter for soft inlier counting.
//	 * @param inlierThreshold RANSAC inlier threshold.
//	 * @param maxDistErr Distance errors are clamped to this maximum value.
//	 */
//	void dScoreRGBD(
//	    dsacstar::coord_t& sceneCoordinates,
//	    dsacstar::coord_t& cameraCoordinates,
//	    const std::vector<cv::Point2i>& validPts,
//	    const std::vector<std::vector<cv::Point2i>>& sampledPoints,
//	    std::vector<cv::Mat_<double>>& jacobeansScore,
//	    const std::vector<double>& scoreOutputGradients,
//	    const std::vector<dsacstar::pose_t>& hyps,
//	    const std::vector<cv::Mat_<float>>& distErrs,
//	    const std::vector<double>& hypProbs,
//	    float inlierAlpha,
//	    float inlierThreshold,
//	    float maxDistErr)
//	{
//		int imH = sceneCoordinates.size(2);
//		int imW = sceneCoordinates.size(3);
//
//	    int hypCount = sampledPoints.size();
//	    // beta parameter for soft inlier counting.
//	    float inlierBeta = 5 / inlierThreshold;
//
//		int batchIdx = 0; // ony batch size = 1 supported atm
//
//	    // collect 2d-3D correspondences
//		std::vector<std::vector<cv::Point3f>> eyePts(hypCount);
//		std::vector<std::vector<cv::Point3f>> objPts(hypCount);
//
//	    #pragma omp parallel for
//	    for(int h = 0; h < hypCount; h++)
//	    {
//	    	if(hypProbs[h] < PROB_THRESH) continue;
//
//	        for(unsigned i = 0; i < sampledPoints[h].size(); i++)
//	        {
//	            int x = sampledPoints[h][i].x;
//	            int y = sampledPoints[h][i].y;
//
//	            eyePts[h].push_back(cv::Point3f(
//					cameraCoordinates[batchIdx][0][y][x],
//					cameraCoordinates[batchIdx][1][y][x],
//					cameraCoordinates[batchIdx][2][y][x]));
//	            objPts[h].push_back(cv::Point3f(
//					sceneCoordinates[batchIdx][0][y][x],
//					sceneCoordinates[batchIdx][1][y][x],
//					sceneCoordinates[batchIdx][2][y][x]));
//	        }
//	    }
//
//	    // derivatives of the soft inlier scores
//	    std::vector<cv::Mat_<double>> dDistErrs(distErrs.size());
//
//	    #pragma omp parallel for
//	    for(int h = 0; h < hypCount; h++)
//	    {
//	        if(hypProbs[h] < PROB_THRESH) continue;
//
//	        dDistErrs[h] = cv::Mat_<double>::zeros(distErrs[h].size());
//
//			for(unsigned ptIdx = 0; ptIdx < validPts.size(); ptIdx++)
//			{
//				int x = validPts[ptIdx].x;
//				int y = validPts[ptIdx].y;
//
//	            double softThreshold = inlierBeta * (distErrs[h](y, x) - inlierThreshold);
//	            softThreshold = 1 / (1+std::exp(-softThreshold));
//	            dDistErrs[h](y, x) = -softThreshold * (1 - softThreshold) * inlierBeta * scoreOutputGradients[h];
//	        }
//
//	        dDistErrs[h] *= inlierAlpha  / dDistErrs[h].cols / dDistErrs[h].rows;
//	    }
//
//	    jacobeansScore.resize(hypCount);
//
//	    // derivative of the loss wrt the score
//	    #pragma omp parallel for
//	    for(int h = 0; h < hypCount; h++)
//	    {
//			if(hypProbs[h] < PROB_THRESH) continue;
//
//	        cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(1, imW * imH * 3);
//	        jacobeansScore[h] = jacobean;
//
//
//
//	        // accumulate derivate of score wrt the scene coordinates that are used to calculate the pose
//	        cv::Mat_<double> supportPointGradients = cv::Mat_<double>::zeros(1, 9);
//
//	        cv::Mat_<double> dHdO;
//	        dsacstar::pose_t cvHyp;
//	        kabsch(eyePts[h], objPts[h], cvHyp, dHdO);
//	        if (dHdO.empty())
//	        	dKabschFD(eyePts[h], objPts[h], dHdO);
//
//	        if(dsacstar::getMax(dHdO) > 10) dHdO = 0; // clamping for stability
//
//			for(unsigned ptIdx = 0; ptIdx < validPts.size(); ptIdx++)
//			{
//				int x = validPts[ptIdx].x;
//				int y = validPts[ptIdx].y;
//
//	            cv::Point3f eye = cv::Point3f(
//					cameraCoordinates[batchIdx][0][y][x],
//					cameraCoordinates[batchIdx][1][y][x],
//					cameraCoordinates[batchIdx][2][y][x]);
//	            cv::Point3f obj = cv::Point3f(
//					sceneCoordinates[batchIdx][0][y][x],
//					sceneCoordinates[batchIdx][1][y][x],
//					sceneCoordinates[batchIdx][2][y][x]);
//
//	            // account for the direct influence of all scene coordinates in the score
//	            cv::Mat_<double> dPdO = dTransformdObj(eye, obj, hyps[h], maxDistErr);
//	            dPdO *= dDistErrs[h](y, x);
//	            dPdO.copyTo(jacobean.colRange(x * dDistErrs[h].rows * 3 + y * 3, x * dDistErrs[h].rows * 3 + y * 3 + 3));
//
//          		// account for the indirect influence of the scene coorindates that are used to calculate the pose
//            	cv::Mat_<double> dPdH = dTransformdHyp(eye, obj, hyps[h], maxDistErr);
//
//	            supportPointGradients += dDistErrs[h](y, x) * dPdH * dHdO;
//	        }
//
//	        // add the accumulated derivatives for the scene coordinates that are used to calculate the pose
//	        for(unsigned i = 0; i < sampledPoints[h].size(); i++)
//	        {
//	            unsigned x = sampledPoints[h][i].x;
//	            unsigned y = sampledPoints[h][i].y;
//
//	            jacobean.colRange(x * dDistErrs[h].rows * 3 + y * 3, x * dDistErrs[h].rows * 3 + y * 3 + 3) += supportPointGradients.colRange(i * 3, i * 3 + 3);
//	        }
//	    }
//
//	}


//	/**
//	 * @brief Calculates the Jacobean matrix of the function that maps n estimated scene coordinates to a soft max score, ie. the function has the form n x 3 -> 1. Returns one Jacobean matrix per hypothesis. RGB-D version.
//	 *
//	 * This is the Soft maxed version of dScoreRGBD (see above).
//	 *
//	 * @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
//	 * @param cameraCoordinates Camera coordinates calculated from measured depth, same format and size as scene coordinates.
//	 * @param validPts A list of valid 2D image positions where camera coordinates / measured depth exists.
//	 * @param sampledPoints Corresponding minimal set for each hypotheses as scene coordinate indices.
//	 * @param losses Loss value for each hypothesis.
//	 * @param hypProbs Selection probabilities over all hypotheses.
//	 * @paran initHyps List of unrefined hypotheses.
//	 * @paran initDistErrs List of 3D distance error images of unrefined hypotheses.
//	 * @param inlierAlpha Alpha parameter for soft inlier counting.
//	 * @param inlierThreshold RANSAC inlier threshold.
//	 * @param maxDistErr Clamp distance error with this value.
//	 * @return List of Jacobean matrices. One 1 x 3n matrix per pose hypothesis.
//	 */
//	std::vector<cv::Mat_<double>> dSMScoreRGBD(
//	    dsacstar::coord_t& sceneCoordinates,
//	    dsacstar::coord_t& cameraCoordinates,
//	    const std::vector<cv::Point2i>& validPts,
//	    const std::vector<std::vector<cv::Point2i>>& sampledPoints,
//	    const std::vector<double>& losses,
//	    const std::vector<double>& hypProbs,
//	    const std::vector<dsacstar::pose_t>& initHyps,
//	    const std::vector<cv::Mat_<float>>& initDistErrs,
//	    float inlierAlpha,
//	    float inlierThreshold,
//	    float maxDistErr)
//	{
//		int imH = sceneCoordinates.size(2);
//		int imW = sceneCoordinates.size(3);
//
//	    // assemble the gradients wrt the scores, ie the gradients of soft max function
//	    std::vector<double> scoreOutputGradients(sampledPoints.size());
//
//	    #pragma omp parallel for
//	    for(unsigned i = 0; i < sampledPoints.size(); i++)
//	    {
//			if(hypProbs[i] < PROB_THRESH) continue;
//
//	        scoreOutputGradients[i] = hypProbs[i] * losses[i];
//	        for(unsigned j = 0; j < sampledPoints.size(); j++)
//	            scoreOutputGradients[i] -= hypProbs[i] * hypProbs[j] * losses[j];
//	    }
//
//	    // calculate gradients of the score function
//	    std::vector<cv::Mat_<double>> jacobeansScore;
//
//	    dScoreRGBD(
//	    	sceneCoordinates,
//	    	cameraCoordinates,
//	    	validPts,
//	    	sampledPoints,
//	    	jacobeansScore,
//	    	scoreOutputGradients,
//	    	initHyps,
//	    	initDistErrs,
//	    	hypProbs,
//	    	inlierAlpha,
//	    	inlierThreshold,
//	    	maxDistErr);
//
//	    // data conversion
//	    #pragma omp parallel for
//	    for(unsigned i = 0; i < jacobeansScore.size(); i++)
//	    {
//	        // reorder to points row first into rows
//	        cv::Mat_<double> reformat = cv::Mat_<double>::zeros(imW * imH, 3);
//
//			if(hypProbs[i] >= PROB_THRESH)
//			{
//				for(unsigned ptIdx = 0; ptIdx < validPts.size(); ptIdx++)
//				{
//					int x = validPts[ptIdx].x;
//					int y = validPts[ptIdx].y;
//
//		            cv::Mat_<double> patchGrad = jacobeansScore[i].colRange(
//		              x * imH * 3 + y * 3,
//		              x * imH * 3 + y * 3 + 3);
//
//		            patchGrad.copyTo(reformat.row(y * imW + x));
//		        }
//			}
//
//	        jacobeansScore[i] = reformat;
//	    }
//
//	    return jacobeansScore;
//
//	}


}
