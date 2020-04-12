#ifndef _ME_VSLAM_CONFIG_HPP_
#define _ME_VSLAM_CONFIG_HPP_

#include <memory>

#include <opencv2/opencv.hpp>

namespace utils {

    struct config {

        static const std::string default_config_file;

		template <typename _Parameter>
		static _Parameter get(const std::string& name) {
			return _Parameter(config::_global_conf->_storage[name]);
		}

		static bool load_configuration();
		static bool load_configuration(const std::string& path);

	private:
		using config_ptr = std::shared_ptr<config>;

		static config_ptr _global_conf;
		cv::FileStorage   _storage;

		explicit config(const std::string& path);
	};
}

#endif