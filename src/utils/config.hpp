#ifndef _ME_VSLAM_CONFIG_HPP_
#define _ME_VSLAM_CONFIG_HPP_

#include <common.hpp>

namespace vslam {

    struct config {

        static const std::string default_config_file;

		template <typename _Parameter>
		static _Parameter get(const std::string& name) {
			return _Parameter(config::_global_conf->_storage[name]);
		}

		static void load_configuration(const std::string& path) {
			if (nullptr == _global_conf) {
				_global_conf = std::shared_ptr<config>(new config(path));
			}
			if (!_global_conf->_storage.isOpened()) {
				_global_conf.reset();
			}
		}

	private:
		static config_ptr _global_conf;
		cv::FileStorage   _storage;

		explicit config(const std::string& path) :
			_storage(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML)
		{
			if (!_storage.isOpened()) {
				std::cerr << "Failed to open file: " << path << std::endl;
				_storage.release();
			}
		}
	};

	config_ptr config::_global_conf(nullptr);
    const std::string config::default_config_file = "conf/default.yaml";
}

#endif