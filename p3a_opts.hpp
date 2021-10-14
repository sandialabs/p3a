#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

namespace p3a {

class opt_error : public std::runtime_error {
 public:
  opt_error(std::string const& arg)
    :std::runtime_error(arg)
  {}
};

class opt {
  std::string m_name;
  std::vector<std::string> m_arguments;
  bool m_is_set{false};
  int m_expected_argument_count{-1};
 public:
  opt(std::string const& name_arg);
  std::string const& name() const;
  opt& expect_arguments(int n = -1);
  int expected_argument_count() const;
  void set();
  bool is_set() const;
  int argument_count() const;
  void add_argument(std::string const& arg);
  std::string const& argument(int i) const;
  template <class T>
  T argument_as(int i) const
  {
    std::stringstream stream(argument(i));
    T result;
    stream >> result;
    if (stream.fail() || !stream.eof()) {
      throw opt_error(
          "option " +
          m_name +
          " argument " +
          argument(i) +
          " is not convertible to the desired type");
    }
    return result;
  }
};

class opts {
  std::vector<opt> m_options;
  std::vector<opt> m_positional_options;
 public:
  opt& add(std::string const& name);
  opt& add_positional(std::string const& name);
  void parse(int& argc, char** argv, bool allow_unrecognized = false);
  bool has(std::string const& name) const;
  int argument_count(std::string const& name) const;
  std::string const& argument(std::string const& name, int i = 0) const;
  template <class T>
  T argument_as(std::string const& name, int i = 0) const
  {
    try {
      return get_option(name).argument_as<T>(i);
    } catch (opt_error const& e) {
      throw opt_error(std::string(e.what()) + "\n" + help_text());
    }
  }
 private:
  opt const& get_option(std::string const& name) const;
  std::string help_text() const;
};

}
