#pragma once

#include <mpi.h>

namespace p3a {

namespace mpi {

class exception : public std::exception {
  std::string error_string;
 public:
  exception(int errorcode) {
    char c_error_string[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errorcode, c_error_string, &resultlen);
    c_error_string[resultlen] = '\0';
    error_string = c_error_string;
  }
  const char* what() const noexcept override {
    return error_string.c_str();
  }
};

namespace details {

inline void handle_error_code(int errorcode) {
  if (errorcode == MPI_SUCCESS) return;
  throw exception(errorcode);
}

}

class status {
  MPI_Status implementation;
 public:
  status(MPI_Status implementation_arg)
    :implementation(implementation_arg)
  {}
  status() = default;
  int source() const { return implementation.MPI_SOURCE; }
  int tag() const { return implementation.MPI_TAG; }
  int error() const { return implementation.MPI_ERROR; }
};

class request {
  MPI_Request implementation;
 public:
  constexpr request()
    :implementation(MPI_REQUEST_NULL)
  {}
  constexpr request(MPI_Request implementation_arg)
    :implementation(implementation_arg)
  {}
  request(request const&) = delete;
  request& operator=(request const&) = delete;
  constexpr request(request&& other)
    :implementation(other.implementation)
  {
    other.implementation = MPI_REQUEST_NULL;
  }
  request& operator=(request&& other)
  {
    wait();
    implementation = other.implementation;
    other.implementation = MPI_REQUEST_NULL;
    return *this;
  }
  void wait()
  {
    if (implementation != MPI_REQUEST_NULL) {
      details::handle_error_code(MPI_Wait(&implementation, MPI_STATUS_IGNORE));
    }
  }
  bool test()
  {
    int flag = 1;
    if (implementation != MPI_REQUEST_NULL) {
      details::handle_error_code(MPI_Test(&implementation, &flag, MPI_STATUS_IGNORE));
    }
    return bool(flag);
  }
  void wait(status& status_arg)
  {
    if (implementation != MPI_REQUEST_NULL) {
      MPI_Status status_implementation;
      details::handle_error_code(MPI_Wait(&implementation, &status_implementation));
      status_arg = status(status_implementation);
    }
  }
  bool test(status& status_arg)
  {
    int flag = 1;
    if (implementation != MPI_REQUEST_NULL) {
      MPI_Status status_implementation;
      details::handle_error_code(MPI_Test(&implementation, &flag, &status_implementation));
      status_arg = status(status_implementation);
    }
    return bool(flag);
  }
  ~request()
  {
    wait();
  }
  MPI_Request& get_implementation() { return implementation; }
};

void waitall(int count, request* array_of_requests)
{
  MPI_Request* array_of_implementations = &(array_of_requests->get_implementation());
  details::handle_error_code(
      MPI_Waitall(
        count,
        array_of_implementations,
        MPI_STATUSES_IGNORE));
}

class op {
  MPI_Op implementation;
  bool owned;
 public:
  constexpr op(MPI_Op implementation_in,
     bool owned_in = true)
    :implementation(implementation_in)
    ,owned(owned_in)
  {}
  constexpr op()
    :implementation(MPI_OP_NULL)
    ,owned(false)
  {}
  op(op const&) = delete;
  op& operator=(op const&) = delete;
  constexpr op(op&& other)
    :implementation(other.implementation)
    ,owned(other.owned)
  {
    other.implementation = MPI_OP_NULL;
    other.owned = false;
  }
  op& operator=(op&& other)
  {
    if (owned) {
      details::handle_error_code(MPI_Op_free(&implementation));
    }
    implementation = other.implementation;
    owned = other.owned;
    other.implementation = MPI_OP_NULL;
    other.owned = false;
    return *this;
  }
  ~op()
  {
    if (owned) {
      details::handle_error_code(MPI_Op_free(&implementation));
    }
  }
  MPI_Op get_implementation() const { return implementation; }
  static op sum()
  {
    return op(MPI_SUM, false);
  }
  static op min()
  {
    return op(MPI_MIN, false);
  }
  static op max()
  {
    return op(MPI_MAX, false);
  }
  static op bor()
  {
    return op(MPI_BOR, false);
  }
};

class datatype {
  MPI_Datatype implementation;
  bool owned;
 public:
  constexpr datatype(
      MPI_Datatype implementation_in,
      bool owned_in = true)
    :implementation(implementation_in)
    ,owned(owned_in)
  {}
  constexpr datatype()
    :implementation(MPI_DATATYPE_NULL)
    ,owned(false)
  {}
  datatype(datatype const&) = delete;
  datatype& operator=(datatype const&) = delete;
  constexpr datatype(datatype&& other)
    :implementation(other.implementation)
    ,owned(other.owned)
  {
    other.implementation = MPI_DATATYPE_NULL;
    other.owned = false;
  }
  datatype& operator=(datatype&& other)
  {
    if (owned) {
      details::handle_error_code(MPI_Type_free(&implementation));
    }
    implementation = other.implementation;
    owned = other.owned;
    other.implementation = MPI_DATATYPE_NULL;
    other.owned = false;
    return *this;
  }
  ~datatype()
  {
    if (owned) {
      details::handle_error_code(MPI_Type_free(&implementation));
    }
  }
  constexpr MPI_Datatype get_implementation() const { return implementation; }
  static datatype predefined_byte()
  {
    return datatype(MPI_BYTE, false);
  }
  static datatype predefined_char()
  {
    return datatype(MPI_CHAR, false);
  }
  static datatype predefined_unsigned()
  {
    return datatype(MPI_UNSIGNED, false);
  }
  static datatype predefined_unsigned_long()
  {
    return datatype(MPI_UNSIGNED_LONG, false);
  }
  static datatype predefined_unsigned_long_long()
  {
    return datatype(MPI_UNSIGNED_LONG_LONG, false);
  }
  static datatype predefined_int()
  {
    return datatype(MPI_INT, false);
  }
  static datatype predefined_long_long_int()
  {
    return datatype(MPI_LONG_LONG_INT, false);
  }
  static datatype predefined_float()
  {
    return datatype(MPI_FLOAT, false);
  }
  static datatype predefined_double()
  {
    return datatype(MPI_DOUBLE, false);
  }
};

namespace details {

template <class T>
class predefined_datatype_helper;

template <>
class predefined_datatype_helper<std::byte>
{
 public:
  static datatype value()
  {
    return datatype::predefined_byte();
  }
};

template <>
class predefined_datatype_helper<char>
{
 public:
  static datatype value()
  {
    return datatype::predefined_char();
  }
};

template <>
class predefined_datatype_helper<unsigned>
{
 public:
  static datatype value()
  {
    return datatype::predefined_unsigned();
  }
};

template <>
class predefined_datatype_helper<unsigned long>
{
 public:
  static datatype value()
  {
    return datatype::predefined_unsigned_long();
  }
};

template <>
class predefined_datatype_helper<unsigned long long>
{
 public:
  static datatype value()
  {
    return datatype::predefined_unsigned_long_long();
  }
};

template <>
class predefined_datatype_helper<int>
{
 public:
  static datatype value()
  {
    return datatype::predefined_int();
  }
};

template <>
class predefined_datatype_helper<long long int>
{
 public:
  static datatype value()
  {
    return datatype::predefined_long_long_int();
  }
};

template <>
class predefined_datatype_helper<float>
{
 public:
  static datatype value()
  {
    return datatype::predefined_float();
  }
};

template <>
class predefined_datatype_helper<double>
{
 public:
  static datatype value()
  {
    return datatype::predefined_double();
  }
};

}

template <class T>
datatype predefined_datatype()
{
  return details::predefined_datatype_helper<T>::value();
}

class comm {
  MPI_Comm implementation;
  bool owned;
 public:
  constexpr comm(
      MPI_Comm implementation_arg,
      bool owned_arg = true)
    :implementation(implementation)
    ,owned(owned_arg)
  {}
  constexpr comm()
    :implementation(MPI_COMM_NULL)
    ,owned(false)
  {}
  comm(comm const&) = delete;
  comm& operator=(comm const&) = delete;
  constexpr comm(comm&& other)
    :implementation(other.implementation)
    ,owned(other.owned)
  {
    other.implementation = MPI_COMM_NULL;
    other.owned = false;
  }
  comm& operator=(comm&& other) {
    if (owned) {
      details::handle_error_code(MPI_Comm_free(&implementation));
    }
    implementation = other.implementation;
    owned = other.owned;
    other.implementation = MPI_COMM_NULL;
    other.owned = false;
    return *this;
  }
  ~comm()
  {
    if (owned) {
      details::handle_error_code(MPI_Comm_free(&implementation));
    }
  }
  request iallreduce(
      void const* sendbuf,
      void* recvbuf,
      int count,
      datatype datatype_arg,
      op op_arg)
  {
    MPI_Request request_implementation;
    details::handle_error_code(
        MPI_Iallreduce(
          sendbuf,
          recvbuf,
          count,
          datatype_arg.get_implementation(),
          op_arg.get_implementation(),
          implementation,
          &request_implementation));
    return request(request_implementation);
  }
  template <class T>
  request iallreduce(
      T const* sendbuf,
      T* recvbuf,
      int count,
      op op_arg)
  {
    datatype datatype_arg = predefined_datatype<T>();
    MPI_Request request_implementation;
    details::handle_error_code(
        MPI_Iallreduce(
          sendbuf,
          recvbuf,
          count,
          datatype_arg.get_implementation(),
          op_arg.get_implementation(),
          implementation,
          &request_implementation));
    return request(request_implementation);
  }
  template <class T>
  request iallreduce(
      T* buf,
      int count,
      op op_arg)
  {
    datatype datatype_arg = predefined_datatype<T>();
    MPI_Request request_implementation;
    details::handle_error_code(
        MPI_Iallreduce(
          MPI_IN_PLACE,
          buf,
          count,
          datatype_arg.get_implementation(),
          op_arg.get_implementation(),
          implementation,
          &request_implementation));
    return request(request_implementation);
  }
  request isend(
      void const* buf,
      int count,
      datatype datatype_arg,
      int dest,
      int tag)
  {
    MPI_Request request_implementation;
    details::handle_error_code(
        MPI_Isend(
          buf,
          count,
          datatype_arg.get_implementation(),
          dest,
          tag,
          implementation,
          &request_implementation));
    return request(request_implementation);
  }
  template <class T>
  request isend(
      T const* buf,
      int count,
      int dest,
      int tag)
  {
    datatype datatype_arg = predefined_datatype<T>();
    MPI_Request request_implementation;
    details::handle_error_code(
        MPI_Isend(
          buf,
          count,
          datatype_arg.get_implementation(),
          dest,
          tag,
          implementation,
          &request_implementation));
    return request(request_implementation);
  }
  request irecv(
      void* buf,
      int count,
      datatype datatype_arg,
      int dest,
      int tag)
  {
    MPI_Request request_implementation;
    details::handle_error_code(
        MPI_Irecv(
          buf,
          count,
          datatype_arg.get_implementation(),
          dest,
          tag,
          implementation,
          &request_implementation));
    return request(request_implementation);
  }
  template <class T>
  request irecv(
      T* buf,
      int count,
      int dest,
      int tag)
  {
    datatype datatype_arg = predefined_datatype<T>();
    MPI_Request request_implementation;
    details::handle_error_code(
        MPI_Irecv(
          buf,
          count,
          datatype_arg.get_implementation(),
          dest,
          tag,
          implementation,
          &request_implementation));
    return request(request_implementation);
  }
  static comm world()
  {
    return comm(MPI_COMM_WORLD, false);
  }
  static comm self()
  {
    return comm(MPI_COMM_SELF, false);
  }
};

class library {
 public:
  library(int* argc, char*** argv) {
    int flag;
    details::handle_error_code(MPI_Initialized(&flag));
    if (!flag) {
      details::handle_error_code(MPI_Init(argc, argv));
    }
  }
  ~library() {
    int flag;
    details::handle_error_code(MPI_Finalized(&flag));
    if (!flag) {
      details::handle_error_code(MPI_Finalize());
    }
  }
  library(library const&) = delete;
  library& operator=(library const&) = delete;
  library(library&&) = delete;
  library& operator=(library&&) = delete;
};

}

}
