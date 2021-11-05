#include "p3a_opts.hpp"

#include <queue>

namespace p3a {

opt::opt(std::string const& name_arg)
  :m_name(name_arg)
{
  for (char c : m_name) {
    if (!std::isalnum(c) && !(c == '-')) {
      throw opt_error(
          "option name " +
          m_name +
          " contains more than " +
          "letters, numbers, and dashes");
    }
  }
}

std::string const& opt::name() const
{
  return m_name;
}

opt& opt::expect_arguments(int n)
{
  m_expected_argument_count = n;
  return *this;
}

int opt::expected_argument_count() const
{
  return m_expected_argument_count;
}

void opt::set()
{
  if (m_is_set) {
    throw opt_error(
        "option " +
        m_name +
        " was set twice");
  }
  m_is_set = true;
}

bool opt::is_set() const
{
  return m_is_set;
}

int opt::argument_count() const
{
  return int(m_arguments.size());
}

void opt::add_argument(std::string const& arg)
{
  if (m_expected_argument_count != -1 &&
      m_expected_argument_count == argument_count()) {
    throw opt_error(
        "option " +
        m_name +
        " given too many arguments, expects " +
        std::to_string(m_expected_argument_count));
  }
  m_arguments.push_back(arg);
}

std::string const& opt::argument(int i) const
{
  if (!m_is_set) {
    throw opt_error(
        "argument requested from option " +
        m_name +
        " but that option was not given");
  }
  if (i < 0) {
    throw opt_error(
        "option " +
        m_name +
        " negative argument index requested");
  }
  if (i >= argument_count()) {
    throw opt_error(
        "option " +
        m_name +
        " argument index out of range " +
        std::to_string(i) +
        " >= " +
        std::to_string(argument_count()));
  }
  return m_arguments[std::size_t(i)];
}

opt& opts::add(std::string const& name)
{
  for (opt const& o : m_options) {
    if (o.name() == name) {
      throw opt_error(
          "can't add option " +
          name +
          " because it already exists");
    }
  }
  for (opt const& o : m_positional_options) {
    if (o.name() == name) {
      throw opt_error(
          "can't add option " +
          name +
          " because it already exists");
    }
  }
  m_options.push_back(opt(name));
  return m_options.back();
}

opt& opts::add_positional(std::string const& name)
{
  for (opt const& o : m_options) {
    if (o.name() == name) {
      throw opt_error(
          "can't add positional option " +
          name +
          " because it already exists");
    }
  }
  for (opt const& o : m_positional_options) {
    if (o.name() == name) {
      throw opt_error(
          "can't add positional option " +
          name +
          " because it already exists");
    }
  }
  m_positional_options.push_back(opt(name));
  m_positional_options.back().expect_arguments(1);
  return m_positional_options.back();
}

void opts::parse(int& argc, char** argv, bool allow_unrecognized)
{
  try {
    std::queue<char*> input_arguments;
    for (int i = 1; i < argc; ++i) {
      input_arguments.push(argv[i]);
    }
    std::queue<char*> output_arguments;
    opt* option_receiving_arguments = nullptr;
    while (!input_arguments.empty()) {
      char* c_argument = input_arguments.front();
      input_arguments.pop();
      std::string const argument(c_argument);
      if (option_receiving_arguments &&
          (option_receiving_arguments->argument_count() ==
           option_receiving_arguments->expected_argument_count())) {
        option_receiving_arguments = nullptr;
      }
      if (option_receiving_arguments) {
        if (option_receiving_arguments->expected_argument_count() == -1) {
          bool was_new_option = false;
          for (opt& o : m_options) {
            if (argument == "--" + o.name()) {
              o.set();
              was_new_option = true;
              option_receiving_arguments = &o;
            }
          }
          if (!was_new_option) {
            option_receiving_arguments->add_argument(argument);
          }
        } else {
          option_receiving_arguments->add_argument(argument);
        }
      } else {
        bool was_new_option = false;
        for (opt& o : m_options) {
          if (argument == "--" + o.name()) {
            o.set();
            option_receiving_arguments = &o;
            was_new_option = true;
          }
        }
        if (!was_new_option) {
          bool was_given_to_positional = false;
          for (opt& o : m_positional_options) {
            if (o.expected_argument_count() == 0) {
              throw opt_error("positional option " + o.name() + " expects no arguments");
            }
            if (o.argument_count() != o.expected_argument_count()) {
              o.set();
              option_receiving_arguments = &o;
              o.add_argument(argument);
              was_given_to_positional = true;
            }
          }
          if (!was_given_to_positional) {
            if (allow_unrecognized) {
              output_arguments.push(c_argument);
            } else {
              throw opt_error(
                  "unrecognized command line argument " +
                  argument);
            }
          }
        }
      }
    }
    for (opt const& o : m_options) {
      if (o.is_set() &&
          o.expected_argument_count() != -1 &&
          o.expected_argument_count() != o.argument_count()) {
        throw opt_error(
            "option " +
            o.name() +
            " expected " +
            std::to_string(o.expected_argument_count()) +
            " arguments but received " +
            std::to_string(o.argument_count()));
      }
    }
    for (opt const& o : m_positional_options) {
      if (o.is_set() &&
          o.expected_argument_count() != -1 &&
          o.expected_argument_count() != o.argument_count()) {
        throw opt_error(
            "option " +
            o.name() +
            " expected " +
            std::to_string(o.expected_argument_count()) +
            " arguments but received " +
            std::to_string(o.argument_count()));
      }
    }
    // rewrite argc and argv to remove the things this opts object has parsed
    fprintf(stderr, "before argc/argv rewrite, output_arguments.size()=%d and old argc=%d\n",
        int(output_arguments.size()),
        argc);
    argc = 1;
    while (!output_arguments.empty()) {
      fprintf(stderr, "setting argv[%d] to \"%s\"\n",
          argc, output_arguments.front());
      argv[argc] = output_arguments.front();
      output_arguments.pop();
      ++argc;
    }
  } catch (opt_error const& e) {
    throw opt_error(std::string(e.what()) + "\n" + help_text());
  }
}

bool opts::has(std::string const& name) const
{
  return get_option(name).is_set();
}

int opts::argument_count(std::string const& name) const
{
  return get_option(name).argument_count();
}

std::string const& opts::argument(std::string const& name, int i) const
{
  try {
    return get_option(name).argument(i);
  } catch (opt_error const& e) {
    throw opt_error(
        std::string(e.what()) + "\n" + help_text());
  }
}

opt const& opts::get_option(std::string const& name) const
{
  for (opt const& o : m_options) {
    if (o.name() == name) {
      return o;
    }
  }
  for (opt const& o : m_positional_options) {
    if (o.name() == name) {
      return o;
    }
  }
  throw opt_error(
      "option " +
      name +
      " requested but was never defined");
}

std::string opts::help_text() const
{
  std::string result;
  result += "  command line options are:\n ";
  for (opt const& o : m_positional_options) {
    result += " [";
    if (o.expected_argument_count() == -1) {
      result += o.name();
      result += "...";
    } else {
      for (int i = 0; i < o.expected_argument_count(); ++i) {
        if (i > 0) result += " ";
        result += o.name();
        result += std::to_string(i);
      }
    }
    result += "]";
  }
  for (opt const& o : m_options) {
    result += " [--";
    result += o.name();
    if (o.expected_argument_count() == -1) {
      result += " [args...]";
    } else {
      for (int i = 0; i < o.expected_argument_count(); ++i) {
        result += " arg";
        result += std::to_string(i);
      }
    }
    result += "]";
  }
  result += "\n";
  return result;
}

}
