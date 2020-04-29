#ifndef _ME_VSLAM_HPP_
#define _ME_VSLAM_HPP_

/**
 * @cpp_header geometry and calculus tools
 */ 
#include <vo/camera.hpp>
#include <vo/jaccobian.hpp>

/**
 * @cpp_header basic-elements types
 */ 
#include <vo/frame.hpp>
#include <vo/feature.hpp>
#include <vo/map_point.hpp>
#include <vo/map.hpp>

/**
 * @cpp_header main components
 */ 
#include <vo/depth_filter.hpp>
#include <vo/pose_estimator.hpp>
#include <vo/reprojector.hpp>
#include <vo/initializer.hpp>
#include <vo/matcher.hpp>

/**
 * @cpp_header pipline implementation
 */ 
#include <vo/core.hpp>

#endif