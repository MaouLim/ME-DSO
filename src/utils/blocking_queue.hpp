#ifndef _ME_VSLAM_BLOCKING_QUEUE_HPP_
#define _ME_VSLAM_BLOCKING_QUEUE_HPP_

#include <deque>
#include <mutex>
#include <condition_variable>

namespace utils {

    /**
	 * @brief A thread-safe queue(bounded) for message passing.
	 */
	template <
		typename _Tp, 
		typename _Container = std::deque<_Tp>
	>
	class bounded_blocking_queue {

		static_assert(std::is_same<_Tp, typename _Container::value_type>::value);

		typedef _Container                              container_type;
		typedef bounded_blocking_queue<_Tp, _Container> self_type;
		typedef typename container_type::size_type      size_type;

	public:
		typedef _Tp                        value_type;
		typedef _Tp&                       reference;
		typedef const _Tp&                 const_reference;
		typedef std::shared_ptr<_Tp>       pointer;
		typedef const std::shared_ptr<_Tp> const_pointer;

	public:
		bounded_blocking_queue() : m_capacity(default_capacity) { }
		explicit bounded_blocking_queue(size_type capacity) : m_capacity(capacity) { }

		/* uncopyable */
		bounded_blocking_queue(const self_type&) = delete;
		self_type& operator=(const self_type&) = delete;

		/**
		 * @brief force to push a item immediately even if the queue is full,
		 *        if the queue is full, the front element of the queue will be
		 *        pop out.
		 */
		void force_to_push(const value_type& item, value_type& out) {
			std::unique_lock<std::mutex> locker(m_mutex);
			if (this->_full()) { out = m_container.front(); m_container.pop_front();  }
			m_container.push_back(item);
			m_empty.notify_one();
		}

		pointer force_to_push(const value_type& item) {
			std::unique_lock<std::mutex> locker(m_mutex);
			std::shared_ptr<value_type> ptr = nullptr;
			if (this->_full()) { 
				ptr = std::make_shared<value_type>(m_container.front()); 
				m_container.pop_front();
			}
			m_container.push_back(item);
			m_empty.notify_one();
			return ptr;
		}

		void wait_and_push(const value_type& item) {
			std::unique_lock<std::mutex> locker(m_mutex);
			m_full.wait(locker, [this]() { return !this->_full(); });
			m_container.push_back(item);
			m_empty.notify_one();
		}

		void wait_and_pop(value_type& out) {
			std::unique_lock<std::mutex> locker(m_mutex);
			m_empty.wait(locker, [this]() { return !this->_empty(); });
			out = m_container.front();
			m_container.pop_front();
			m_full.notify_one();
		}

		pointer wait_and_pop() {
			std::unique_lock<std::mutex> locker(m_mutex);
			m_empty.wait(locker, [this]() { return !this->_empty(); });
			auto ptr = std::make_shared<value_type>(m_container.front());
			m_container.pop_front();
			m_full.notify_one();
			return ptr;
		}

		bool try_push(const value_type& item) {
			std::lock_guard<std::mutex> locker(m_mutex);
			if (this->_full()) { return false; }

			m_container.push_back(item);
			m_empty.notify_one();

			return true;
		}

		bool try_pop(value_type& out) {
			std::lock_guard<std::mutex> locker(m_mutex);
			if (this->_empty()) { return false; }

			out = m_container.front();
			m_container.pop_front();
			m_full.notify_one();

			return true;
		}

		pointer try_pop() {
			std::lock_guard<std::mutex> locker(m_mutex);
			if (this->_empty()) { return pointer(nullptr); }

			auto ptr = std::make_shared<value_type>(m_container.front());
			m_container.pop_front();
			m_full.notify_one();

			return ptr;
		}

		pointer front() const {
			std::lock_guard<std::mutex> locker(m_mutex);
			if (this->_empty()) { return pointer(nullptr); }
			auto ptr = std::make_shared<value_type>(m_container.front());
			return ptr;
		}

		pointer back() const {
			std::lock_guard<std::mutex> locker(m_mutex);
			if (this->_empty()) { return pointer(nullptr); }
			auto ptr = std::make_shared<value_type>(m_container.back());
			return ptr;
		}

		void clear() {
			std::lock_guard<std::mutex> locker(m_mutex);
			m_container.clear();
			m_full.notify_one();
		}

	private:
		bool _empty() const { return m_container.empty(); }
		bool _full() const { return m_capacity <= m_container.size(); }

	private:
		static const size_type default_capacity = 1024u;

		/* mutex and conditions */
		mutable std::mutex      m_mutex;
		std::condition_variable m_full;
		std::condition_variable m_empty;

		const size_type         m_capacity;
		container_type          m_container;
	};

	/** 
	 * @brief Optimized version for smart_ptr type
	 */
	template <typename _Tp, typename _Container>
	class bounded_blocking_queue<std::shared_ptr<_Tp>, _Container> {

		static_assert(std::is_same<
			std::shared_ptr<_Tp>, 
			typename _Container::value_type
		>::value);

		typedef _Container                                    container_type;
		typedef std::shared_ptr<_Tp>                          smart_ptr;
		typedef bounded_blocking_queue<smart_ptr, _Container> self_type;
		typedef typename container_type::size_type            size_type;

		typedef _Tp* raw_ptr;

	public:
		typedef _Tp                        value_type;
		typedef _Tp&                       reference;
		typedef const _Tp&                 const_reference;
		typedef std::shared_ptr<_Tp>       pointer;
		typedef const std::shared_ptr<_Tp> const_pointer;

	public:
		bounded_blocking_queue() : m_capacity(default_capacity) { }
		explicit bounded_blocking_queue(size_type capacity) : m_capacity(capacity) { }

		~bounded_blocking_queue() { clear(); }

		/* uncopyable */
		bounded_blocking_queue(const self_type&) = delete;
		self_type& operator=(const self_type&) = delete;

		void wait_and_push(pointer&& item) {
			std::unique_lock<std::mutex> locker(m_mutex);
			m_full.wait(locker, [this]() { return !this->_full(); });
			m_container.push_back(std::move(item));
			m_empty.notify_one();
		}

		void wait_and_push(raw_ptr p) {
			std::unique_lock<std::mutex> locker(m_mutex);
			m_full.wait(locker, [this]() { return !this->_full(); });
			m_container.push_back(smart_ptr(p));
			m_empty.notify_one();
		}

		template <typename _Rep, typename _Period>
		bool wait_and_push_for(raw_ptr p, const std::chrono::duration<_Rep, _Period>& time) {
			std::unique_lock<std::mutex> locker(m_mutex);
			bool not_full =
				m_empty.wait_for(locker, time, [this]() { return !this->_full(); });

			if (not_full) {
				m_container.push_back(smart_ptr(p));
				m_empty.notify_one();
			}

			return not_full;
		}

		template <typename _Rep, typename _Period>
		bool wait_and_push_for(pointer&& item, const std::chrono::duration<_Rep, _Period>& time) {
			std::unique_lock<std::mutex> locker(m_mutex);
			bool not_full =
				m_empty.wait_for(locker, time, [this]() { return !this->_full(); });

			if (not_full) {
				m_container.push_back(std::move(item));
				m_empty.notify_one();
			}

			return not_full;
		}

		void wait_and_pop(value_type& out) {
			std::unique_lock<std::mutex> locker(m_mutex);
			m_empty.wait(locker, [this]() { return !this->_empty(); });
			out = *m_container.front();
			m_container.pop_front();
			m_full.notify_one();
		}

		pointer wait_and_pop() {
			std::unique_lock<std::mutex> locker(m_mutex);
			m_empty.wait(locker, [this]() { return !this->_empty(); });
			auto ptr = std::move(m_container.front());
			m_container.pop_front();
			m_full.notify_one();
			return ptr;
		}

		template <typename _Rep, typename _Period>
		pointer wait_and_pop_for(const std::chrono::duration<_Rep, _Period>& time) {
			std::unique_lock<std::mutex> locker(m_mutex);
			bool not_empty =
				m_empty.wait_for(locker, time, [this]() { return !this->_empty(); });
			if (not_empty) {
				auto ptr = std::move(m_container.front());
				m_container.pop_front();
				m_full.notify_one();
				return ptr;
			}
			return pointer();
		}

		bool try_push(pointer&& item) {
			std::lock_guard<std::mutex> locker(m_mutex);
			if (this->_full()) { return false; }

			m_container.push_back(std::move(item));
			m_empty.notify_one();

			return true;
		}

		bool try_push(raw_ptr p) {
			std::lock_guard<std::mutex> locker(m_mutex);
			if (this->_full()) { return false; }

			m_container.push_back(smart_ptr(p));
			m_empty.notify_one();

			return true;
		}

		bool try_pop(value_type& out) {
			std::lock_guard<std::mutex> locker(m_mutex);
			if (this->_empty()) { return false; }

			out = *m_container.front();
			m_container.pop_front();
			m_full.notify_one();

			return true;
		}

		pointer try_pop() {
			std::lock_guard<std::mutex> locker(m_mutex);
			if (this->_empty()) { return pointer(nullptr); }

			auto ptr = std::move(m_container.front());
			m_container.pop_front();
			m_full.notify_one();

			return ptr;
		}

		pointer front() const {
			std::lock_guard<std::mutex> locker(m_mutex);
			if (this->_empty()) { return pointer(nullptr); }
			return std::make_shared<value_type>(*m_container.front());
		}

		pointer back() const {
			std::lock_guard<std::mutex> locker(m_mutex);
			if (this->_empty()) { return pointer(nullptr); }
			return std::make_shared<value_type>(*m_container.back());
		}

		void clear() {
			std::lock_guard<std::mutex> locker(m_mutex);
			m_container.clear();
			m_full.notify_one();
		}

	private:
		bool _empty() const { return m_container.empty(); }
		bool _full() const { return m_capacity <= m_container.size(); }

	private:
		static const size_type default_capacity = 1024u;

		/* mutex and conditions */
		mutable std::mutex      m_mutex;
		std::condition_variable m_full;
		std::condition_variable m_empty;

		const size_type         m_capacity;
		container_type          m_container;
	};
    
} // namespace utils

#endif