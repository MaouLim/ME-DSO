#include <utils/config.hpp>

#include <exception>

namespace utils {

    config::config_ptr config::_global_conf       (nullptr);
    const std::string  config::default_config_file("conf/default.yaml");

    config::config(const std::string& path) {
		try {
			_storage.open(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
		}
		catch (const std::exception& ex) {
			if (!_storage.isOpened()) {
				_storage.release();
			}
			std::cerr << "Failed to open file: " << path << std::endl;
			std::cerr << "Error info: " << ex.what() << std::endl;
		}
	}

	// template <typename _Parameter>
	// static _Parameter get(
	// 	const std::string& name, 
	// 	const _Parameter&  default_val
	// ) {
	// 	try {
	// 		config::_global_conf->_storage.
	// 		cv::FileNode node = config::_global_conf->_storage[name];
	// 		if (node.isNone()) { return default_val; }
	// 		return _Parameter(node);
	// 	}
	// 	catch (const std::exception& ex) {  return default_val; }
	// }

	bool config::load_configuration() { 
		return load_configuration(config::default_config_file);
	}

	bool config::load_configuration(const std::string& path) {
		if (nullptr == config::_global_conf) {
			_global_conf = std::shared_ptr<config>(new config(path));
		}
		if (!_global_conf->_storage.isOpened()) {
			_global_conf.reset();
			return false;
		}
		return true;
	}
    
} // namespace utils
