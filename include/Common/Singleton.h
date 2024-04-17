#pragma once

#include <memory>


template <typename T> 
class SingletonHolder {
private:
  static std::unique_ptr<T> m_instance;

  SingletonHolder() = default;

public:
  ~SingletonHolder() = default;

  static std::unique_ptr<T>& instance() { return m_instance; }

  static void instance(std::unique_ptr<T> ptr) { m_instance = std::move(ptr); }

  static void clear() { m_instance.reset(); }
};

template<typename T>
class Singleton
{
public:
    Singleton() = default;
    ~Singleton() = default;


    template<typename... Args>
    std::unique_ptr<T> Instance(Args&&... args) {
        if(m_instance == nullptr)
            m_instance = CreateInstance(std::forward<Args>(args)...);

        return m_instance;
    }

    void Clear() { m_instance.reset(); };

private:

    template<typename... Args>
    static std::unique_ptr<T> CreateInstance(Args&&... args) {
        return std::make_unique<T>(std::forward<Args>(args)...);
    }


private:

    std::unique_ptr<T> m_instance;
};


