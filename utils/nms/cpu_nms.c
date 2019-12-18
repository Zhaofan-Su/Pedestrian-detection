
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#ifndef Py_PYTHON_H
    #error Python headers needed to compile C extensions, please install development version of Python.
#elif PY_VERSION_HEX < 0x02060000 || (0x03000000 <= PY_VERSION_HEX && PY_VERSION_HEX < 0x03020000)
    #error Cython requires Python 2.6+ or Python 3.2+.
#else
#define CYTHON_ABI "0_25_2"
#include <stddef.h>
#ifndef offsetof
  #define offsetof(type, member) ( (size_t) & ((type*)0) -> member )
#endif
#if !defined(WIN32) && !defined(MS_WINDOWS)
  #ifndef __stdcall
    #define __stdcall
  #endif
  #ifndef __cdecl
    #define __cdecl
  #endif
  #ifndef __fastcall
    #define __fastcall
  #endif
#endif
#ifndef DL_IMPORT
  #define DL_IMPORT(t) t
#endif
#ifndef DL_EXPORT
  #define DL_EXPORT(t) t
#endif
#ifndef HAVE_LONG_LONG
  #if PY_VERSION_HEX >= 0x03030000 || (PY_MAJOR_VERSION == 2 && PY_VERSION_HEX >= 0x02070000)
    #define HAVE_LONG_LONG
  #endif
#endif
#ifndef PY_LONG_LONG
  #define PY_LONG_LONG LONG_LONG
#endif
#ifndef Py_HUGE_VAL
  #define Py_HUGE_VAL HUGE_VAL
#endif
#ifdef PYPY_VERSION
  #define CYTHON_COMPILING_IN_PYPY 1
  #define CYTHON_COMPILING_IN_PYSTON 0
  #define CYTHON_COMPILING_IN_CPYTHON 0
  #undef CYTHON_USE_TYPE_SLOTS
  #define CYTHON_USE_TYPE_SLOTS 0
  #undef CYTHON_USE_ASYNC_SLOTS
  #define CYTHON_USE_ASYNC_SLOTS 0
  #undef CYTHON_USE_PYLIST_INTERNALS
  #define CYTHON_USE_PYLIST_INTERNALS 0
  #undef CYTHON_USE_UNICODE_INTERNALS
  #define CYTHON_USE_UNICODE_INTERNALS 0
  #undef CYTHON_USE_UNICODE_WRITER
  #define CYTHON_USE_UNICODE_WRITER 0
  #undef CYTHON_USE_PYLONG_INTERNALS
  #define CYTHON_USE_PYLONG_INTERNALS 0
  #undef CYTHON_AVOID_BORROWED_REFS
  #define CYTHON_AVOID_BORROWED_REFS 1
  #undef CYTHON_ASSUME_SAFE_MACROS
  #define CYTHON_ASSUME_SAFE_MACROS 0
  #undef CYTHON_UNPACK_METHODS
  #define CYTHON_UNPACK_METHODS 0
  #undef CYTHON_FAST_THREAD_STATE
  #define CYTHON_FAST_THREAD_STATE 0
  #undef CYTHON_FAST_PYCALL
  #define CYTHON_FAST_PYCALL 0
#elif defined(PYSTON_VERSION)
  #define CYTHON_COMPILING_IN_PYPY 0
  #define CYTHON_COMPILING_IN_PYSTON 1
  #define CYTHON_COMPILING_IN_CPYTHON 0
  #ifndef CYTHON_USE_TYPE_SLOTS
    #define CYTHON_USE_TYPE_SLOTS 1
  #endif
  #undef CYTHON_USE_ASYNC_SLOTS
  #define CYTHON_USE_ASYNC_SLOTS 0
  #undef CYTHON_USE_PYLIST_INTERNALS
  #define CYTHON_USE_PYLIST_INTERNALS 0
  #ifndef CYTHON_USE_UNICODE_INTERNALS
    #define CYTHON_USE_UNICODE_INTERNALS 1
  #endif
  #undef CYTHON_USE_UNICODE_WRITER
  #define CYTHON_USE_UNICODE_WRITER 0
  #undef CYTHON_USE_PYLONG_INTERNALS
  #define CYTHON_USE_PYLONG_INTERNALS 0
  #ifndef CYTHON_AVOID_BORROWED_REFS
    #define CYTHON_AVOID_BORROWED_REFS 0
  #endif
  #ifndef CYTHON_ASSUME_SAFE_MACROS
    #define CYTHON_ASSUME_SAFE_MACROS 1
  #endif
  #ifndef CYTHON_UNPACK_METHODS
    #define CYTHON_UNPACK_METHODS 1
  #endif
  #undef CYTHON_FAST_THREAD_STATE
  #define CYTHON_FAST_THREAD_STATE 0
  #undef CYTHON_FAST_PYCALL
  #define CYTHON_FAST_PYCALL 0
#else
  #define CYTHON_COMPILING_IN_PYPY 0
  #define CYTHON_COMPILING_IN_PYSTON 0
  #define CYTHON_COMPILING_IN_CPYTHON 1
  #ifndef CYTHON_USE_TYPE_SLOTS
    #define CYTHON_USE_TYPE_SLOTS 1
  #endif
  #if PY_MAJOR_VERSION < 3
    #undef CYTHON_USE_ASYNC_SLOTS
    #define CYTHON_USE_ASYNC_SLOTS 0
  #elif !defined(CYTHON_USE_ASYNC_SLOTS)
    #define CYTHON_USE_ASYNC_SLOTS 1
  #endif
  #if PY_VERSION_HEX < 0x02070000
    #undef CYTHON_USE_PYLONG_INTERNALS
    #define CYTHON_USE_PYLONG_INTERNALS 0
  #elif !defined(CYTHON_USE_PYLONG_INTERNALS)
    #define CYTHON_USE_PYLONG_INTERNALS 1
  #endif
  #ifndef CYTHON_USE_PYLIST_INTERNALS
    #define CYTHON_USE_PYLIST_INTERNALS 1
  #endif
  #ifndef CYTHON_USE_UNICODE_INTERNALS
    #define CYTHON_USE_UNICODE_INTERNALS 1
  #endif
  #if PY_VERSION_HEX < 0x030300F0
    #undef CYTHON_USE_UNICODE_WRITER
    #define CYTHON_USE_UNICODE_WRITER 0
  #elif !defined(CYTHON_USE_UNICODE_WRITER)
    #define CYTHON_USE_UNICODE_WRITER 1
  #endif
  #ifndef CYTHON_AVOID_BORROWED_REFS
    #define CYTHON_AVOID_BORROWED_REFS 0
  #endif
  #ifndef CYTHON_ASSUME_SAFE_MACROS
    #define CYTHON_ASSUME_SAFE_MACROS 1
  #endif
  #ifndef CYTHON_UNPACK_METHODS
    #define CYTHON_UNPACK_METHODS 1
  #endif
  #ifndef CYTHON_FAST_THREAD_STATE
    #define CYTHON_FAST_THREAD_STATE 1
  #endif
  #ifndef CYTHON_FAST_PYCALL
    #define CYTHON_FAST_PYCALL 1
  #endif
#endif
#if !defined(CYTHON_FAST_PYCCALL)
#define CYTHON_FAST_PYCCALL  (CYTHON_FAST_PYCALL && PY_VERSION_HEX >= 0x030600B1)
#endif
#if CYTHON_USE_PYLONG_INTERNALS
  #include "longintrepr.h"
  #undef SHIFT
  #undef BASE
  #undef MASK
#endif
#if CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX < 0x02070600 && !defined(Py_OptimizeFlag)
  #define Py_OptimizeFlag 0
#endif
#define __PYX_BUILD_PY_SSIZE_T "n"
#define CYTHON_FORMAT_SSIZE_T "z"
#if PY_MAJOR_VERSION < 3
  #define __Pyx_BUILTIN_MODULE_NAME "__builtin__"
  #define __Pyx_PyCode_New(a, k, l, s, f, code, c, n, v, fv, cell, fn, name, fline, lnos)\
          PyCode_New(a+k, l, s, f, code, c, n, v, fv, cell, fn, name, fline, lnos)
  #define __Pyx_DefaultClassType PyClass_Type
#else
  #define __Pyx_BUILTIN_MODULE_NAME "builtins"
  #define __Pyx_PyCode_New(a, k, l, s, f, code, c, n, v, fv, cell, fn, name, fline, lnos)\
          PyCode_New(a, k, l, s, f, code, c, n, v, fv, cell, fn, name, fline, lnos)
  #define __Pyx_DefaultClassType PyType_Type
#endif
#ifndef Py_TPFLAGS_CHECKTYPES
  #define Py_TPFLAGS_CHECKTYPES 0
#endif
#ifndef Py_TPFLAGS_HAVE_INDEX
  #define Py_TPFLAGS_HAVE_INDEX 0
#endif
#ifndef Py_TPFLAGS_HAVE_NEWBUFFER
  #define Py_TPFLAGS_HAVE_NEWBUFFER 0
#endif
#ifndef Py_TPFLAGS_HAVE_FINALIZE
  #define Py_TPFLAGS_HAVE_FINALIZE 0
#endif
#ifndef METH_FASTCALL
  #define METH_FASTCALL 0x80
  typedef PyObject *(*__Pyx_PyCFunctionFast) (PyObject *self, PyObject **args,
                                              Py_ssize_t nargs, PyObject *kwnames);
#else
  #define __Pyx_PyCFunctionFast _PyCFunctionFast
#endif
#if CYTHON_FAST_PYCCALL
#define __Pyx_PyFastCFunction_Check(func)\
    ((PyCFunction_Check(func) && (METH_FASTCALL == (PyCFunction_GET_FLAGS(func) & ~(METH_CLASS | METH_STATIC | METH_COEXIST)))))
#else
#define __Pyx_PyFastCFunction_Check(func) 0
#endif
#if PY_VERSION_HEX > 0x03030000 && defined(PyUnicode_KIND)
  #define CYTHON_PEP393_ENABLED 1
  #define __Pyx_PyUnicode_READY(op)       (likely(PyUnicode_IS_READY(op)) ?\
                                              0 : _PyUnicode_Ready((PyObject *)(op)))
  #define __Pyx_PyUnicode_GET_LENGTH(u)   PyUnicode_GET_LENGTH(u)
  #define __Pyx_PyUnicode_READ_CHAR(u, i) PyUnicode_READ_CHAR(u, i)
  #define __Pyx_PyUnicode_MAX_CHAR_VALUE(u)   PyUnicode_MAX_CHAR_VALUE(u)
  #define __Pyx_PyUnicode_KIND(u)         PyUnicode_KIND(u)
  #define __Pyx_PyUnicode_DATA(u)         PyUnicode_DATA(u)
  #define __Pyx_PyUnicode_READ(k, d, i)   PyUnicode_READ(k, d, i)
  #define __Pyx_PyUnicode_WRITE(k, d, i, ch)  PyUnicode_WRITE(k, d, i, ch)
  #define __Pyx_PyUnicode_IS_TRUE(u)      (0 != (likely(PyUnicode_IS_READY(u)) ? PyUnicode_GET_LENGTH(u) : PyUnicode_GET_SIZE(u)))
#else
  #define CYTHON_PEP393_ENABLED 0
  #define PyUnicode_1BYTE_KIND  1
  #define PyUnicode_2BYTE_KIND  2
  #define PyUnicode_4BYTE_KIND  4
  #define __Pyx_PyUnicode_READY(op)       (0)
  #define __Pyx_PyUnicode_GET_LENGTH(u)   PyUnicode_GET_SIZE(u)
  #define __Pyx_PyUnicode_READ_CHAR(u, i) ((Py_UCS4)(PyUnicode_AS_UNICODE(u)[i]))
  #define __Pyx_PyUnicode_MAX_CHAR_VALUE(u)   ((sizeof(Py_UNICODE) == 2) ? 65535 : 1114111)
  #define __Pyx_PyUnicode_KIND(u)         (sizeof(Py_UNICODE))
  #define __Pyx_PyUnicode_DATA(u)         ((void*)PyUnicode_AS_UNICODE(u))
  #define __Pyx_PyUnicode_READ(k, d, i)   ((void)(k), (Py_UCS4)(((Py_UNICODE*)d)[i]))
  #define __Pyx_PyUnicode_WRITE(k, d, i, ch)  (((void)(k)), ((Py_UNICODE*)d)[i] = ch)
  #define __Pyx_PyUnicode_IS_TRUE(u)      (0 != PyUnicode_GET_SIZE(u))
#endif
#if CYTHON_COMPILING_IN_PYPY
  #define __Pyx_PyUnicode_Concat(a, b)      PyNumber_Add(a, b)
  #define __Pyx_PyUnicode_ConcatSafe(a, b)  PyNumber_Add(a, b)
#else
  #define __Pyx_PyUnicode_Concat(a, b)      PyUnicode_Concat(a, b)
  #define __Pyx_PyUnicode_ConcatSafe(a, b)  ((unlikely((a) == Py_None) || unlikely((b) == Py_None)) ?\
      PyNumber_Add(a, b) : __Pyx_PyUnicode_Concat(a, b))
#endif
#if CYTHON_COMPILING_IN_PYPY && !defined(PyUnicode_Contains)
  #define PyUnicode_Contains(u, s)  PySequence_Contains(u, s)
#endif
#if CYTHON_COMPILING_IN_PYPY && !defined(PyByteArray_Check)
  #define PyByteArray_Check(obj)  PyObject_TypeCheck(obj, &PyByteArray_Type)
#endif
#if CYTHON_COMPILING_IN_PYPY && !defined(PyObject_Format)
  #define PyObject_Format(obj, fmt)  PyObject_CallMethod(obj, "__format__", "O", fmt)
#endif
#if CYTHON_COMPILING_IN_PYPY && !defined(PyObject_Malloc)
  #define PyObject_Malloc(s)   PyMem_Malloc(s)
  #define PyObject_Free(p)     PyMem_Free(p)
  #define PyObject_Realloc(p)  PyMem_Realloc(p)
#endif
#if CYTHON_COMPILING_IN_PYSTON
  #define __Pyx_PyCode_HasFreeVars(co)  PyCode_HasFreeVars(co)
  #define __Pyx_PyFrame_SetLineNumber(frame, lineno) PyFrame_SetLineNumber(frame, lineno)
#else
  #define __Pyx_PyCode_HasFreeVars(co)  (PyCode_GetNumFree(co) > 0)
  #define __Pyx_PyFrame_SetLineNumber(frame, lineno)  (frame)->f_lineno = (lineno)
#endif
#define __Pyx_PyString_FormatSafe(a, b)   ((unlikely((a) == Py_None)) ? PyNumber_Remainder(a, b) : __Pyx_PyString_Format(a, b))
#define __Pyx_PyUnicode_FormatSafe(a, b)  ((unlikely((a) == Py_None)) ? PyNumber_Remainder(a, b) : PyUnicode_Format(a, b))
#if PY_MAJOR_VERSION >= 3
  #define __Pyx_PyString_Format(a, b)  PyUnicode_Format(a, b)
#else
  #define __Pyx_PyString_Format(a, b)  PyString_Format(a, b)
#endif
#if PY_MAJOR_VERSION < 3 && !defined(PyObject_ASCII)
  #define PyObject_ASCII(o)            PyObject_Repr(o)
#endif
#if PY_MAJOR_VERSION >= 3
  #define PyBaseString_Type            PyUnicode_Type
  #define PyStringObject               PyUnicodeObject
  #define PyString_Type                PyUnicode_Type
  #define PyString_Check               PyUnicode_Check
  #define PyString_CheckExact          PyUnicode_CheckExact
#endif
#if PY_MAJOR_VERSION >= 3
  #define __Pyx_PyBaseString_Check(obj) PyUnicode_Check(obj)
  #define __Pyx_PyBaseString_CheckExact(obj) PyUnicode_CheckExact(obj)
#else
  #define __Pyx_PyBaseString_Check(obj) (PyString_Check(obj) || PyUnicode_Check(obj))
  #define __Pyx_PyBaseString_CheckExact(obj) (PyString_CheckExact(obj) || PyUnicode_CheckExact(obj))
#endif
#ifndef PySet_CheckExact
  #define PySet_CheckExact(obj)        (Py_TYPE(obj) == &PySet_Type)
#endif
#define __Pyx_TypeCheck(obj, type) PyObject_TypeCheck(obj, (PyTypeObject *)type)
#define __Pyx_PyException_Check(obj) __Pyx_TypeCheck(obj, PyExc_Exception)
#if PY_MAJOR_VERSION >= 3
  #define PyIntObject                  PyLongObject
  #define PyInt_Type                   PyLong_Type
  #define PyInt_Check(op)              PyLong_Check(op)
  #define PyInt_CheckExact(op)         PyLong_CheckExact(op)
  #define PyInt_FromString             PyLong_FromString
  #define PyInt_FromUnicode            PyLong_FromUnicode
  #define PyInt_FromLong               PyLong_FromLong
  #define PyInt_FromSize_t             PyLong_FromSize_t
  #define PyInt_FromSsize_t            PyLong_FromSsize_t
  #define PyInt_AsLong                 PyLong_AsLong
  #define PyInt_AS_LONG                PyLong_AS_LONG
  #define PyInt_AsSsize_t              PyLong_AsSsize_t
  #define PyInt_AsUnsignedLongMask     PyLong_AsUnsignedLongMask
  #define PyInt_AsUnsignedLongLongMask PyLong_AsUnsignedLongLongMask
  #define PyNumber_Int                 PyNumber_Long
#endif
#if PY_MAJOR_VERSION >= 3
  #define PyBoolObject                 PyLongObject
#endif
#if PY_MAJOR_VERSION >= 3 && CYTHON_COMPILING_IN_PYPY
  #ifndef PyUnicode_InternFromString
    #define PyUnicode_InternFromString(s) PyUnicode_FromString(s)
  #endif
#endif
#if PY_VERSION_HEX < 0x030200A4
  typedef long Py_hash_t;
  #define __Pyx_PyInt_FromHash_t PyInt_FromLong
  #define __Pyx_PyInt_AsHash_t   PyInt_AsLong
#else
  #define __Pyx_PyInt_FromHash_t PyInt_FromSsize_t
  #define __Pyx_PyInt_AsHash_t   PyInt_AsSsize_t
#endif
#if PY_MAJOR_VERSION >= 3
  #define __Pyx_PyMethod_New(func, self, klass) ((self) ? PyMethod_New(func, self) : PyInstanceMethod_New(func))
#else
  #define __Pyx_PyMethod_New(func, self, klass) PyMethod_New(func, self, klass)
#endif
#if CYTHON_USE_ASYNC_SLOTS
  #if PY_VERSION_HEX >= 0x030500B1
    #define __Pyx_PyAsyncMethodsStruct PyAsyncMethods
    #define __Pyx_PyType_AsAsync(obj) (Py_TYPE(obj)->tp_as_async)
  #else
    typedef struct {
        unaryfunc am_await;
        unaryfunc am_aiter;
        unaryfunc am_anext;
    } __Pyx_PyAsyncMethodsStruct;
    #define __Pyx_PyType_AsAsync(obj) ((__Pyx_PyAsyncMethodsStruct*) (Py_TYPE(obj)->tp_reserved))
  #endif
#else
  #define __Pyx_PyType_AsAsync(obj) NULL
#endif
#ifndef CYTHON_RESTRICT
  #if defined(__GNUC__)
    #define CYTHON_RESTRICT __restrict__
  #elif defined(_MSC_VER) && _MSC_VER >= 1400
    #define CYTHON_RESTRICT __restrict
  #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define CYTHON_RESTRICT restrict
  #else
    #define CYTHON_RESTRICT
  #endif
#endif
#ifndef CYTHON_UNUSED
# if defined(__GNUC__)
#   if !(defined(__cplusplus)) || (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#     define CYTHON_UNUSED __attribute__ ((__unused__))
#   else
#     define CYTHON_UNUSED
#   endif
# elif defined(__ICC) || (defined(__INTEL_COMPILER) && !defined(_MSC_VER))
#   define CYTHON_UNUSED __attribute__ ((__unused__))
# else
#   define CYTHON_UNUSED
# endif
#endif
#ifndef CYTHON_MAYBE_UNUSED_VAR
#  if defined(__cplusplus)
     template<class T> void CYTHON_MAYBE_UNUSED_VAR( const T& ) { }
#  else
#    define CYTHON_MAYBE_UNUSED_VAR(x) (void)(x)
#  endif
#endif
#ifndef CYTHON_NCP_UNUSED
# if CYTHON_COMPILING_IN_CPYTHON
#  define CYTHON_NCP_UNUSED
# else
#  define CYTHON_NCP_UNUSED CYTHON_UNUSED
# endif
#endif
#define __Pyx_void_to_None(void_result) ((void)(void_result), Py_INCREF(Py_None), Py_None)

#ifndef CYTHON_INLINE
  #if defined(__clang__)
    #define CYTHON_INLINE __inline__ __attribute__ ((__unused__))
  #elif defined(__GNUC__)
    #define CYTHON_INLINE __inline__
  #elif defined(_MSC_VER)
    #define CYTHON_INLINE __inline
  #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define CYTHON_INLINE inline
  #else
    #define CYTHON_INLINE
  #endif
#endif

#if defined(WIN32) || defined(MS_WINDOWS)
  #define _USE_MATH_DEFINES
#endif
#include <math.h>
#ifdef NAN
#define __PYX_NAN() ((float) NAN)
#else
static CYTHON_INLINE float __PYX_NAN() {
  float value;
  memset(&value, 0xFF, sizeof(value));
  return value;
}
#endif
#if defined(__CYGWIN__) && defined(_LDBL_EQ_DBL)
#define __Pyx_truncl trunc
#else
#define __Pyx_truncl truncl
#endif


#define __PYX_ERR(f_index, lineno, Ln_error) \
{ \
  __pyx_filename = __pyx_f[f_index]; __pyx_lineno = lineno; __pyx_clineno = __LINE__; goto Ln_error; \
}

#if PY_MAJOR_VERSION >= 3
  #define __Pyx_PyNumber_Divide(x,y)         PyNumber_TrueDivide(x,y)
  #define __Pyx_PyNumber_InPlaceDivide(x,y)  PyNumber_InPlaceTrueDivide(x,y)
#else
  #define __Pyx_PyNumber_Divide(x,y)         PyNumber_Divide(x,y)
  #define __Pyx_PyNumber_InPlaceDivide(x,y)  PyNumber_InPlaceDivide(x,y)
#endif

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#define __PYX_HAVE__nms__cpu_nms
#define __PYX_HAVE_API__nms__cpu_nms
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

#ifdef PYREX_WITHOUT_ASSERTIONS
#define CYTHON_WITHOUT_ASSERTIONS
#endif

typedef struct {PyObject **p; const char *s; const Py_ssize_t n; const char* encoding;
                const char is_unicode; const char is_str; const char intern; } __Pyx_StringTabEntry;

#define __PYX_DEFAULT_STRING_ENCODING_IS_ASCII 0
#define __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT 0
#define __PYX_DEFAULT_STRING_ENCODING ""
#define __Pyx_PyObject_FromString __Pyx_PyBytes_FromString
#define __Pyx_PyObject_FromStringAndSize __Pyx_PyBytes_FromStringAndSize
#define __Pyx_uchar_cast(c) ((unsigned char)c)
#define __Pyx_long_cast(x) ((long)x)
#define __Pyx_fits_Py_ssize_t(v, type, is_signed)  (\
    (sizeof(type) < sizeof(Py_ssize_t))  ||\
    (sizeof(type) > sizeof(Py_ssize_t) &&\
          likely(v < (type)PY_SSIZE_T_MAX ||\
                 v == (type)PY_SSIZE_T_MAX)  &&\
          (!is_signed || likely(v > (type)PY_SSIZE_T_MIN ||\
                                v == (type)PY_SSIZE_T_MIN)))  ||\
    (sizeof(type) == sizeof(Py_ssize_t) &&\
          (is_signed || likely(v < (type)PY_SSIZE_T_MAX ||\
                               v == (type)PY_SSIZE_T_MAX)))  )
#if defined (__cplusplus) && __cplusplus >= 201103L
    #include <cstdlib>
    #define __Pyx_sst_abs(value) std::abs(value)
#elif SIZEOF_INT >= SIZEOF_SIZE_T
    #define __Pyx_sst_abs(value) abs(value)
#elif SIZEOF_LONG >= SIZEOF_SIZE_T
    #define __Pyx_sst_abs(value) labs(value)
#elif defined (_MSC_VER) && defined (_M_X64)
    #define __Pyx_sst_abs(value) _abs64(value)
#elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define __Pyx_sst_abs(value) llabs(value)
#elif defined (__GNUC__)
    #define __Pyx_sst_abs(value) __builtin_llabs(value)
#else
    #define __Pyx_sst_abs(value) ((value<0) ? -value : value)
#endif
static CYTHON_INLINE char* __Pyx_PyObject_AsString(PyObject*);
static CYTHON_INLINE char* __Pyx_PyObject_AsStringAndSize(PyObject*, Py_ssize_t* length);
#define __Pyx_PyByteArray_FromString(s) PyByteArray_FromStringAndSize((const char*)s, strlen((const char*)s))
#define __Pyx_PyByteArray_FromStringAndSize(s, l) PyByteArray_FromStringAndSize((const char*)s, l)
#define __Pyx_PyBytes_FromString        PyBytes_FromString
#define __Pyx_PyBytes_FromStringAndSize PyBytes_FromStringAndSize
static CYTHON_INLINE PyObject* __Pyx_PyUnicode_FromString(const char*);
#if PY_MAJOR_VERSION < 3
    #define __Pyx_PyStr_FromString        __Pyx_PyBytes_FromString
    #define __Pyx_PyStr_FromStringAndSize __Pyx_PyBytes_FromStringAndSize
#else
    #define __Pyx_PyStr_FromString        __Pyx_PyUnicode_FromString
    #define __Pyx_PyStr_FromStringAndSize __Pyx_PyUnicode_FromStringAndSize
#endif
#define __Pyx_PyObject_AsSString(s)    ((signed char*) __Pyx_PyObject_AsString(s))
#define __Pyx_PyObject_AsUString(s)    ((unsigned char*) __Pyx_PyObject_AsString(s))
#define __Pyx_PyObject_FromCString(s)  __Pyx_PyObject_FromString((const char*)s)
#define __Pyx_PyBytes_FromCString(s)   __Pyx_PyBytes_FromString((const char*)s)
#define __Pyx_PyByteArray_FromCString(s)   __Pyx_PyByteArray_FromString((const char*)s)
#define __Pyx_PyStr_FromCString(s)     __Pyx_PyStr_FromString((const char*)s)
#define __Pyx_PyUnicode_FromCString(s) __Pyx_PyUnicode_FromString((const char*)s)
#if PY_MAJOR_VERSION < 3
static CYTHON_INLINE size_t __Pyx_Py_UNICODE_strlen(const Py_UNICODE *u)
{
    const Py_UNICODE *u_end = u;
    while (*u_end++) ;
    return (size_t)(u_end - u - 1);
}
#else
#define __Pyx_Py_UNICODE_strlen Py_UNICODE_strlen
#endif
#define __Pyx_PyUnicode_FromUnicode(u)       PyUnicode_FromUnicode(u, __Pyx_Py_UNICODE_strlen(u))
#define __Pyx_PyUnicode_FromUnicodeAndLength PyUnicode_FromUnicode
#define __Pyx_PyUnicode_AsUnicode            PyUnicode_AsUnicode
#define __Pyx_NewRef(obj) (Py_INCREF(obj), obj)
#define __Pyx_Owned_Py_None(b) __Pyx_NewRef(Py_None)
#define __Pyx_PyBool_FromLong(b) ((b) ? __Pyx_NewRef(Py_True) : __Pyx_NewRef(Py_False))
static CYTHON_INLINE int __Pyx_PyObject_IsTrue(PyObject*);
static CYTHON_INLINE PyObject* __Pyx_PyNumber_IntOrLong(PyObject* x);
static CYTHON_INLINE Py_ssize_t __Pyx_PyIndex_AsSsize_t(PyObject*);
static CYTHON_INLINE PyObject * __Pyx_PyInt_FromSize_t(size_t);
#if CYTHON_ASSUME_SAFE_MACROS
#define __pyx_PyFloat_AsDouble(x) (PyFloat_CheckExact(x) ? PyFloat_AS_DOUBLE(x) : PyFloat_AsDouble(x))
#else
#define __pyx_PyFloat_AsDouble(x) PyFloat_AsDouble(x)
#endif
#define __pyx_PyFloat_AsFloat(x) ((float) __pyx_PyFloat_AsDouble(x))
#if PY_MAJOR_VERSION >= 3
#define __Pyx_PyNumber_Int(x) (PyLong_CheckExact(x) ? __Pyx_NewRef(x) : PyNumber_Long(x))
#else
#define __Pyx_PyNumber_Int(x) (PyInt_CheckExact(x) ? __Pyx_NewRef(x) : PyNumber_Int(x))
#endif
#define __Pyx_PyNumber_Float(x) (PyFloat_CheckExact(x) ? __Pyx_NewRef(x) : PyNumber_Float(x))
#if PY_MAJOR_VERSION < 3 && __PYX_DEFAULT_STRING_ENCODING_IS_ASCII
static int __Pyx_sys_getdefaultencoding_not_ascii;
static int __Pyx_init_sys_getdefaultencoding_params(void) {
    PyObject* sys;
    PyObject* default_encoding = NULL;
    PyObject* ascii_chars_u = NULL;
    PyObject* ascii_chars_b = NULL;
    const char* default_encoding_c;
    sys = PyImport_ImportModule("sys");
    if (!sys) goto bad;
    default_encoding = PyObject_CallMethod(sys, (char*) "getdefaultencoding", NULL);
    Py_DECREF(sys);
    if (!default_encoding) goto bad;
    default_encoding_c = PyBytes_AsString(default_encoding);
    if (!default_encoding_c) goto bad;
    if (strcmp(default_encoding_c, "ascii") == 0) {
        __Pyx_sys_getdefaultencoding_not_ascii = 0;
    } else {
        char ascii_chars[128];
        int c;
        for (c = 0; c < 128; c++) {
            ascii_chars[c] = c;
        }
        __Pyx_sys_getdefaultencoding_not_ascii = 1;
        ascii_chars_u = PyUnicode_DecodeASCII(ascii_chars, 128, NULL);
        if (!ascii_chars_u) goto bad;
        ascii_chars_b = PyUnicode_AsEncodedString(ascii_chars_u, default_encoding_c, NULL);
        if (!ascii_chars_b || !PyBytes_Check(ascii_chars_b) || memcmp(ascii_chars, PyBytes_AS_STRING(ascii_chars_b), 128) != 0) {
            PyErr_Format(
                PyExc_ValueError,
                "This module compiled with c_string_encoding=ascii, but default encoding '%.200s' is not a superset of ascii.",
                default_encoding_c);
            goto bad;
        }
        Py_DECREF(ascii_chars_u);
        Py_DECREF(ascii_chars_b);
    }
    Py_DECREF(default_encoding);
    return 0;
bad:
    Py_XDECREF(default_encoding);
    Py_XDECREF(ascii_chars_u);
    Py_XDECREF(ascii_chars_b);
    return -1;
}
#endif
#if __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT && PY_MAJOR_VERSION >= 3
#define __Pyx_PyUnicode_FromStringAndSize(c_str, size) PyUnicode_DecodeUTF8(c_str, size, NULL)
#else
#define __Pyx_PyUnicode_FromStringAndSize(c_str, size) PyUnicode_Decode(c_str, size, __PYX_DEFAULT_STRING_ENCODING, NULL)
#if __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT
static char* __PYX_DEFAULT_STRING_ENCODING;
static int __Pyx_init_sys_getdefaultencoding_params(void) {
    PyObject* sys;
    PyObject* default_encoding = NULL;
    char* default_encoding_c;
    sys = PyImport_ImportModule("sys");
    if (!sys) goto bad;
    default_encoding = PyObject_CallMethod(sys, (char*) (const char*) "getdefaultencoding", NULL);
    Py_DECREF(sys);
    if (!default_encoding) goto bad;
    default_encoding_c = PyBytes_AsString(default_encoding);
    if (!default_encoding_c) goto bad;
    __PYX_DEFAULT_STRING_ENCODING = (char*) malloc(strlen(default_encoding_c));
    if (!__PYX_DEFAULT_STRING_ENCODING) goto bad;
    strcpy(__PYX_DEFAULT_STRING_ENCODING, default_encoding_c);
    Py_DECREF(default_encoding);
    return 0;
bad:
    Py_XDECREF(default_encoding);
    return -1;
}
#endif
#endif


/* Test for GCC > 2.95 */
#if defined(__GNUC__)     && (__GNUC__ > 2 || (__GNUC__ == 2 && (__GNUC_MINOR__ > 95)))
  #define likely(x)   __builtin_expect(!!(x), 1)
  #define unlikely(x) __builtin_expect(!!(x), 0)
#else /* !__GNUC__ or GCC < 2.95 */
  #define likely(x)   (x)
  #define unlikely(x) (x)
#endif /* __GNUC__ */

static PyObject *__pyx_m;
static PyObject *__pyx_d;
static PyObject *__pyx_b;
static PyObject *__pyx_empty_tuple;
static PyObject *__pyx_empty_bytes;
static PyObject *__pyx_empty_unicode;
static int __pyx_lineno;
static int __pyx_clineno = 0;
static const char * __pyx_cfilenm= __FILE__;
static const char *__pyx_filename;

/* Header.proto */
#if !defined(CYTHON_CCOMPLEX)
  #if defined(__cplusplus)
    #define CYTHON_CCOMPLEX 1
  #elif defined(_Complex_I)
    #define CYTHON_CCOMPLEX 1
  #else
    #define CYTHON_CCOMPLEX 0
  #endif
#endif
#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    #include <complex>
  #else
    #include <complex.h>
  #endif
#endif
#if CYTHON_CCOMPLEX && !defined(__cplusplus) && defined(__sun__) && defined(__GNUC__)
  #undef _Complex_I
  #define _Complex_I 1.0fj
#endif


static const char *__pyx_f[] = {
  "nms/cpu_nms.pyx",
  "__init__.pxd",
  "type.pxd",
};
/* BufferFormatStructs.proto */
#define IS_UNSIGNED(type) (((type) -1) > 0)
struct __Pyx_StructField_;
#define __PYX_BUF_FLAGS_PACKED_STRUCT (1 << 0)
typedef struct {
  const char* name;
  struct __Pyx_StructField_* fields;
  size_t size;
  size_t arraysize[8];
  int ndim;
  char typegroup;
  char is_unsigned;
  int flags;
} __Pyx_TypeInfo;
typedef struct __Pyx_StructField_ {
  __Pyx_TypeInfo* type;
  const char* name;
  size_t offset;
} __Pyx_StructField;
typedef struct {
  __Pyx_StructField* field;
  size_t parent_offset;
} __Pyx_BufFmt_StackElem;
typedef struct {
  __Pyx_StructField root;
  __Pyx_BufFmt_StackElem* head;
  size_t fmt_offset;
  size_t new_count, enc_count;
  size_t struct_alignment;
  int is_complex;
  char enc_type;
  char new_packmode;
  char enc_packmode;
  char is_valid_array;
} __Pyx_BufFmt_Context;


/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":725
 * # in Cython to enable them only on the right systems.
 * 
 * ctypedef npy_int8       int8_t             # <<<<<<<<<<<<<<
 * ctypedef npy_int16      int16_t
 * ctypedef npy_int32      int32_t
 */
typedef npy_int8 __pyx_t_5numpy_int8_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":726
 * 
 * ctypedef npy_int8       int8_t
 * ctypedef npy_int16      int16_t             # <<<<<<<<<<<<<<
 * ctypedef npy_int32      int32_t
 * ctypedef npy_int64      int64_t
 */
typedef npy_int16 __pyx_t_5numpy_int16_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":727
 * ctypedef npy_int8       int8_t
 * ctypedef npy_int16      int16_t
 * ctypedef npy_int32      int32_t             # <<<<<<<<<<<<<<
 * ctypedef npy_int64      int64_t
 * #ctypedef npy_int96      int96_t
 */
typedef npy_int32 __pyx_t_5numpy_int32_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":728
 * ctypedef npy_int16      int16_t
 * ctypedef npy_int32      int32_t
 * ctypedef npy_int64      int64_t             # <<<<<<<<<<<<<<
 * #ctypedef npy_int96      int96_t
 * #ctypedef npy_int128     int128_t
 */
typedef npy_int64 __pyx_t_5numpy_int64_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":732
 * #ctypedef npy_int128     int128_t
 * 
 * ctypedef npy_uint8      uint8_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uint16     uint16_t
 * ctypedef npy_uint32     uint32_t
 */
typedef npy_uint8 __pyx_t_5numpy_uint8_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":733
 * 
 * ctypedef npy_uint8      uint8_t
 * ctypedef npy_uint16     uint16_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uint32     uint32_t
 * ctypedef npy_uint64     uint64_t
 */
typedef npy_uint16 __pyx_t_5numpy_uint16_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":734
 * ctypedef npy_uint8      uint8_t
 * ctypedef npy_uint16     uint16_t
 * ctypedef npy_uint32     uint32_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uint64     uint64_t
 * #ctypedef npy_uint96     uint96_t
 */
typedef npy_uint32 __pyx_t_5numpy_uint32_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":735
 * ctypedef npy_uint16     uint16_t
 * ctypedef npy_uint32     uint32_t
 * ctypedef npy_uint64     uint64_t             # <<<<<<<<<<<<<<
 * #ctypedef npy_uint96     uint96_t
 * #ctypedef npy_uint128    uint128_t
 */
typedef npy_uint64 __pyx_t_5numpy_uint64_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":739
 * #ctypedef npy_uint128    uint128_t
 * 
 * ctypedef npy_float32    float32_t             # <<<<<<<<<<<<<<
 * ctypedef npy_float64    float64_t
 * #ctypedef npy_float80    float80_t
 */
typedef npy_float32 __pyx_t_5numpy_float32_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":740
 * 
 * ctypedef npy_float32    float32_t
 * ctypedef npy_float64    float64_t             # <<<<<<<<<<<<<<
 * #ctypedef npy_float80    float80_t
 * #ctypedef npy_float128   float128_t
 */
typedef npy_float64 __pyx_t_5numpy_float64_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":749
 * # The int types are mapped a bit surprising --
 * # numpy.int corresponds to 'l' and numpy.long to 'q'
 * ctypedef npy_long       int_t             # <<<<<<<<<<<<<<
 * ctypedef npy_longlong   long_t
 * ctypedef npy_longlong   longlong_t
 */
typedef npy_long __pyx_t_5numpy_int_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":750
 * # numpy.int corresponds to 'l' and numpy.long to 'q'
 * ctypedef npy_long       int_t
 * ctypedef npy_longlong   long_t             # <<<<<<<<<<<<<<
 * ctypedef npy_longlong   longlong_t
 * 
 */
typedef npy_longlong __pyx_t_5numpy_long_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":751
 * ctypedef npy_long       int_t
 * ctypedef npy_longlong   long_t
 * ctypedef npy_longlong   longlong_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_ulong      uint_t
 */
typedef npy_longlong __pyx_t_5numpy_longlong_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":753
 * ctypedef npy_longlong   longlong_t
 * 
 * ctypedef npy_ulong      uint_t             # <<<<<<<<<<<<<<
 * ctypedef npy_ulonglong  ulong_t
 * ctypedef npy_ulonglong  ulonglong_t
 */
typedef npy_ulong __pyx_t_5numpy_uint_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":754
 * 
 * ctypedef npy_ulong      uint_t
 * ctypedef npy_ulonglong  ulong_t             # <<<<<<<<<<<<<<
 * ctypedef npy_ulonglong  ulonglong_t
 * 
 */
typedef npy_ulonglong __pyx_t_5numpy_ulong_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":755
 * ctypedef npy_ulong      uint_t
 * ctypedef npy_ulonglong  ulong_t
 * ctypedef npy_ulonglong  ulonglong_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_intp       intp_t
 */
typedef npy_ulonglong __pyx_t_5numpy_ulonglong_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":757
 * ctypedef npy_ulonglong  ulonglong_t
 * 
 * ctypedef npy_intp       intp_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uintp      uintp_t
 * 
 */
typedef npy_intp __pyx_t_5numpy_intp_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":758
 * 
 * ctypedef npy_intp       intp_t
 * ctypedef npy_uintp      uintp_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_double     float_t
 */
typedef npy_uintp __pyx_t_5numpy_uintp_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":760
 * ctypedef npy_uintp      uintp_t
 * 
 * ctypedef npy_double     float_t             # <<<<<<<<<<<<<<
 * ctypedef npy_double     double_t
 * ctypedef npy_longdouble longdouble_t
 */
typedef npy_double __pyx_t_5numpy_float_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":761
 * 
 * ctypedef npy_double     float_t
 * ctypedef npy_double     double_t             # <<<<<<<<<<<<<<
 * ctypedef npy_longdouble longdouble_t
 * 
 */
typedef npy_double __pyx_t_5numpy_double_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":762
 * ctypedef npy_double     float_t
 * ctypedef npy_double     double_t
 * ctypedef npy_longdouble longdouble_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_cfloat      cfloat_t
 */
typedef npy_longdouble __pyx_t_5numpy_longdouble_t;
/* Declarations.proto */
#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    typedef ::std::complex< float > __pyx_t_float_complex;
  #else
    typedef float _Complex __pyx_t_float_complex;
  #endif
#else
    typedef struct { float real, imag; } __pyx_t_float_complex;
#endif
static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float, float);

/* Declarations.proto */
#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    typedef ::std::complex< double > __pyx_t_double_complex;
  #else
    typedef double _Complex __pyx_t_double_complex;
  #endif
#else
    typedef struct { double real, imag; } __pyx_t_double_complex;
#endif
static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double, double);


/*--- Type declarations ---*/

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":764
 * ctypedef npy_longdouble longdouble_t
 * 
 * ctypedef npy_cfloat      cfloat_t             # <<<<<<<<<<<<<<
 * ctypedef npy_cdouble     cdouble_t
 * ctypedef npy_clongdouble clongdouble_t
 */
typedef npy_cfloat __pyx_t_5numpy_cfloat_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":765
 * 
 * ctypedef npy_cfloat      cfloat_t
 * ctypedef npy_cdouble     cdouble_t             # <<<<<<<<<<<<<<
 * ctypedef npy_clongdouble clongdouble_t
 * 
 */
typedef npy_cdouble __pyx_t_5numpy_cdouble_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":766
 * ctypedef npy_cfloat      cfloat_t
 * ctypedef npy_cdouble     cdouble_t
 * ctypedef npy_clongdouble clongdouble_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_cdouble     complex_t
 */
typedef npy_clongdouble __pyx_t_5numpy_clongdouble_t;

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":768
 * ctypedef npy_clongdouble clongdouble_t
 * 
 * ctypedef npy_cdouble     complex_t             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew1(a):
 */
typedef npy_cdouble __pyx_t_5numpy_complex_t;

/* --- Runtime support code (head) --- */
/* Refnanny.proto */
#ifndef CYTHON_REFNANNY
  #define CYTHON_REFNANNY 0
#endif
#if CYTHON_REFNANNY
  typedef struct {
    void (*INCREF)(void*, PyObject*, int);
    void (*DECREF)(void*, PyObject*, int);
    void (*GOTREF)(void*, PyObject*, int);
    void (*GIVEREF)(void*, PyObject*, int);
    void* (*SetupContext)(const char*, int, const char*);
    void (*FinishContext)(void**);
  } __Pyx_RefNannyAPIStruct;
  static __Pyx_RefNannyAPIStruct *__Pyx_RefNanny = NULL;
  static __Pyx_RefNannyAPIStruct *__Pyx_RefNannyImportAPI(const char *modname);
  #define __Pyx_RefNannyDeclarations void *__pyx_refnanny = NULL;
#ifdef WITH_THREAD
  #define __Pyx_RefNannySetupContext(name, acquire_gil)\
          if (acquire_gil) {\
              PyGILState_STATE __pyx_gilstate_save = PyGILState_Ensure();\
              __pyx_refnanny = __Pyx_RefNanny->SetupContext((name), __LINE__, __FILE__);\
              PyGILState_Release(__pyx_gilstate_save);\
          } else {\
              __pyx_refnanny = __Pyx_RefNanny->SetupContext((name), __LINE__, __FILE__);\
          }
#else
  #define __Pyx_RefNannySetupContext(name, acquire_gil)\
          __pyx_refnanny = __Pyx_RefNanny->SetupContext((name), __LINE__, __FILE__)
#endif
  #define __Pyx_RefNannyFinishContext()\
          __Pyx_RefNanny->FinishContext(&__pyx_refnanny)
  #define __Pyx_INCREF(r)  __Pyx_RefNanny->INCREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_DECREF(r)  __Pyx_RefNanny->DECREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_GOTREF(r)  __Pyx_RefNanny->GOTREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_GIVEREF(r) __Pyx_RefNanny->GIVEREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_XINCREF(r)  do { if((r) != NULL) {__Pyx_INCREF(r); }} while(0)
  #define __Pyx_XDECREF(r)  do { if((r) != NULL) {__Pyx_DECREF(r); }} while(0)
  #define __Pyx_XGOTREF(r)  do { if((r) != NULL) {__Pyx_GOTREF(r); }} while(0)
  #define __Pyx_XGIVEREF(r) do { if((r) != NULL) {__Pyx_GIVEREF(r);}} while(0)
#else
  #define __Pyx_RefNannyDeclarations
  #define __Pyx_RefNannySetupContext(name, acquire_gil)
  #define __Pyx_RefNannyFinishContext()
  #define __Pyx_INCREF(r) Py_INCREF(r)
  #define __Pyx_DECREF(r) Py_DECREF(r)
  #define __Pyx_GOTREF(r)
  #define __Pyx_GIVEREF(r)
  #define __Pyx_XINCREF(r) Py_XINCREF(r)
  #define __Pyx_XDECREF(r) Py_XDECREF(r)
  #define __Pyx_XGOTREF(r)
  #define __Pyx_XGIVEREF(r)
#endif
#define __Pyx_XDECREF_SET(r, v) do {\
        PyObject *tmp = (PyObject *) r;\
        r = v; __Pyx_XDECREF(tmp);\
    } while (0)
#define __Pyx_DECREF_SET(r, v) do {\
        PyObject *tmp = (PyObject *) r;\
        r = v; __Pyx_DECREF(tmp);\
    } while (0)
#define __Pyx_CLEAR(r)    do { PyObject* tmp = ((PyObject*)(r)); r = NULL; __Pyx_DECREF(tmp);} while(0)
#define __Pyx_XCLEAR(r)   do { if((r) != NULL) {PyObject* tmp = ((PyObject*)(r)); r = NULL; __Pyx_DECREF(tmp);}} while(0)

/* PyObjectGetAttrStr.proto */
#if CYTHON_USE_TYPE_SLOTS
static CYTHON_INLINE PyObject* __Pyx_PyObject_GetAttrStr(PyObject* obj, PyObject* attr_name) {
    PyTypeObject* tp = Py_TYPE(obj);
    if (likely(tp->tp_getattro))
        return tp->tp_getattro(obj, attr_name);
#if PY_MAJOR_VERSION < 3
    if (likely(tp->tp_getattr))
        return tp->tp_getattr(obj, PyString_AS_STRING(attr_name));
#endif
    return PyObject_GetAttr(obj, attr_name);
}
#else
#define __Pyx_PyObject_GetAttrStr(o,n) PyObject_GetAttr(o,n)
#endif

/* GetBuiltinName.proto */
static PyObject *__Pyx_GetBuiltinName(PyObject *name);

/* RaiseArgTupleInvalid.proto */
static void __Pyx_RaiseArgtupleInvalid(const char* func_name, int exact,
    Py_ssize_t num_min, Py_ssize_t num_max, Py_ssize_t num_found);

/* RaiseDoubleKeywords.proto */
static void __Pyx_RaiseDoubleKeywordsError(const char* func_name, PyObject* kw_name);

/* ParseKeywords.proto */
static int __Pyx_ParseOptionalKeywords(PyObject *kwds, PyObject **argnames[],\
    PyObject *kwds2, PyObject *values[], Py_ssize_t num_pos_args,\
    const char* function_name);

/* ArgTypeTest.proto */
static CYTHON_INLINE int __Pyx_ArgTypeTest(PyObject *obj, PyTypeObject *type, int none_allowed,
    const char *name, int exact);

/* BufferFormatCheck.proto */
static CYTHON_INLINE int  __Pyx_GetBufferAndValidate(Py_buffer* buf, PyObject* obj,
    __Pyx_TypeInfo* dtype, int flags, int nd, int cast, __Pyx_BufFmt_StackElem* stack);
static CYTHON_INLINE void __Pyx_SafeReleaseBuffer(Py_buffer* info);
static const char* __Pyx_BufFmt_CheckString(__Pyx_BufFmt_Context* ctx, const char* ts);
static void __Pyx_BufFmt_Init(__Pyx_BufFmt_Context* ctx,
                              __Pyx_BufFmt_StackElem* stack,
                              __Pyx_TypeInfo* type); // PROTO

/* ExtTypeTest.proto */
static CYTHON_INLINE int __Pyx_TypeTest(PyObject *obj, PyTypeObject *type);

/* PyIntBinop.proto */
#if !CYTHON_COMPILING_IN_PYPY
static PyObject* __Pyx_PyInt_AddObjC(PyObject *op1, PyObject *op2, long intval, int inplace);
#else
#define __Pyx_PyInt_AddObjC(op1, op2, intval, inplace)\
    (inplace ? PyNumber_InPlaceAdd(op1, op2) : PyNumber_Add(op1, op2))
#endif

/* PyCFunctionFastCall.proto */
#if CYTHON_FAST_PYCCALL
static CYTHON_INLINE PyObject *__Pyx_PyCFunction_FastCall(PyObject *func, PyObject **args, Py_ssize_t nargs);
#else
#define __Pyx_PyCFunction_FastCall(func, args, nargs)  (assert(0), NULL)
#endif

/* PyFunctionFastCall.proto */
#if CYTHON_FAST_PYCALL
#define __Pyx_PyFunction_FastCall(func, args, nargs)\
    __Pyx_PyFunction_FastCallDict((func), (args), (nargs), NULL)
#if 1 || PY_VERSION_HEX < 0x030600B1
static PyObject *__Pyx_PyFunction_FastCallDict(PyObject *func, PyObject **args, int nargs, PyObject *kwargs);
#else
#define __Pyx_PyFunction_FastCallDict(func, args, nargs, kwargs) _PyFunction_FastCallDict(func, args, nargs, kwargs)
#endif
#endif

/* PyObjectCall.proto */
#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_Call(PyObject *func, PyObject *arg, PyObject *kw);
#else
#define __Pyx_PyObject_Call(func, arg, kw) PyObject_Call(func, arg, kw)
#endif

/* PyObjectCallMethO.proto */
#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_CallMethO(PyObject *func, PyObject *arg);
#endif

/* PyObjectCallOneArg.proto */
static CYTHON_INLINE PyObject* __Pyx_PyObject_CallOneArg(PyObject *func, PyObject *arg);

/* PyObjectCallNoArg.proto */
#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_CallNoArg(PyObject *func);
#else
#define __Pyx_PyObject_CallNoArg(func) __Pyx_PyObject_Call(func, __pyx_empty_tuple, NULL)
#endif

/* GetModuleGlobalName.proto */
static CYTHON_INLINE PyObject *__Pyx_GetModuleGlobalName(PyObject *name);

/* BufferIndexError.proto */
static void __Pyx_RaiseBufferIndexError(int axis);

#define __Pyx_BufPtrStrided1d(type, buf, i0, s0) (type)((char*)buf + i0 * s0)
/* ListAppend.proto */
#if CYTHON_USE_PYLIST_INTERNALS && CYTHON_ASSUME_SAFE_MACROS
static CYTHON_INLINE int __Pyx_PyList_Append(PyObject* list, PyObject* x) {
    PyListObject* L = (PyListObject*) list;
    Py_ssize_t len = Py_SIZE(list);
    if (likely(L->allocated > len) & likely(len > (L->allocated >> 1))) {
        Py_INCREF(x);
        PyList_SET_ITEM(list, len, x);
        Py_SIZE(list) = len+1;
        return 0;
    }
    return PyList_Append(list, x);
}
#else
#define __Pyx_PyList_Append(L,x) PyList_Append(L,x)
#endif

/* PyThreadStateGet.proto */
#if CYTHON_FAST_THREAD_STATE
#define __Pyx_PyThreadState_declare  PyThreadState *__pyx_tstate;
#define __Pyx_PyThreadState_assign  __pyx_tstate = PyThreadState_GET();
#else
#define __Pyx_PyThreadState_declare
#define __Pyx_PyThreadState_assign
#endif

/* PyErrFetchRestore.proto */
#if CYTHON_FAST_THREAD_STATE
#define __Pyx_ErrRestoreWithState(type, value, tb)  __Pyx_ErrRestoreInState(PyThreadState_GET(), type, value, tb)
#define __Pyx_ErrFetchWithState(type, value, tb)    __Pyx_ErrFetchInState(PyThreadState_GET(), type, value, tb)
#define __Pyx_ErrRestore(type, value, tb)  __Pyx_ErrRestoreInState(__pyx_tstate, type, value, tb)
#define __Pyx_ErrFetch(type, value, tb)    __Pyx_ErrFetchInState(__pyx_tstate, type, value, tb)
static CYTHON_INLINE void __Pyx_ErrRestoreInState(PyThreadState *tstate, PyObject *type, PyObject *value, PyObject *tb);
static CYTHON_INLINE void __Pyx_ErrFetchInState(PyThreadState *tstate, PyObject **type, PyObject **value, PyObject **tb);
#else
#define __Pyx_ErrRestoreWithState(type, value, tb)  PyErr_Restore(type, value, tb)
#define __Pyx_ErrFetchWithState(type, value, tb)  PyErr_Fetch(type, value, tb)
#define __Pyx_ErrRestore(type, value, tb)  PyErr_Restore(type, value, tb)
#define __Pyx_ErrFetch(type, value, tb)  PyErr_Fetch(type, value, tb)
#endif

#define __Pyx_BufPtrStrided2d(type, buf, i0, s0, i1, s1) (type)((char*)buf + i0 * s0 + i1 * s1)
/* ListCompAppend.proto */
#if CYTHON_USE_PYLIST_INTERNALS && CYTHON_ASSUME_SAFE_MACROS
static CYTHON_INLINE int __Pyx_ListComp_Append(PyObject* list, PyObject* x) {
    PyListObject* L = (PyListObject*) list;
    Py_ssize_t len = Py_SIZE(list);
    if (likely(L->allocated > len)) {
        Py_INCREF(x);
        PyList_SET_ITEM(list, len, x);
        Py_SIZE(list) = len+1;
        return 0;
    }
    return PyList_Append(list, x);
}
#else
#define __Pyx_ListComp_Append(L,x) PyList_Append(L,x)
#endif

/* RaiseException.proto */
static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb, PyObject *cause);

/* DictGetItem.proto */
#if PY_MAJOR_VERSION >= 3 && !CYTHON_COMPILING_IN_PYPY
static PyObject *__Pyx_PyDict_GetItem(PyObject *d, PyObject* key) {
    PyObject *value;
    value = PyDict_GetItemWithError(d, key);
    if (unlikely(!value)) {
        if (!PyErr_Occurred()) {
            PyObject* args = PyTuple_Pack(1, key);
            if (likely(args))
                PyErr_SetObject(PyExc_KeyError, args);
            Py_XDECREF(args);
        }
        return NULL;
    }
    Py_INCREF(value);
    return value;
}
#else
    #define __Pyx_PyDict_GetItem(d, key) PyObject_GetItem(d, key)
#endif

/* RaiseTooManyValuesToUnpack.proto */
static CYTHON_INLINE void __Pyx_RaiseTooManyValuesError(Py_ssize_t expected);

/* RaiseNeedMoreValuesToUnpack.proto */
static CYTHON_INLINE void __Pyx_RaiseNeedMoreValuesError(Py_ssize_t index);

/* RaiseNoneIterError.proto */
static CYTHON_INLINE void __Pyx_RaiseNoneNotIterableError(void);

/* SaveResetException.proto */
#if CYTHON_FAST_THREAD_STATE
#define __Pyx_ExceptionSave(type, value, tb)  __Pyx__ExceptionSave(__pyx_tstate, type, value, tb)
static CYTHON_INLINE void __Pyx__ExceptionSave(PyThreadState *tstate, PyObject **type, PyObject **value, PyObject **tb);
#define __Pyx_ExceptionReset(type, value, tb)  __Pyx__ExceptionReset(__pyx_tstate, type, value, tb)
static CYTHON_INLINE void __Pyx__ExceptionReset(PyThreadState *tstate, PyObject *type, PyObject *value, PyObject *tb);
#else
#define __Pyx_ExceptionSave(type, value, tb)   PyErr_GetExcInfo(type, value, tb)
#define __Pyx_ExceptionReset(type, value, tb)  PyErr_SetExcInfo(type, value, tb)
#endif

/* PyErrExceptionMatches.proto */
#if CYTHON_FAST_THREAD_STATE
#define __Pyx_PyErr_ExceptionMatches(err) __Pyx_PyErr_ExceptionMatchesInState(__pyx_tstate, err)
static CYTHON_INLINE int __Pyx_PyErr_ExceptionMatchesInState(PyThreadState* tstate, PyObject* err);
#else
#define __Pyx_PyErr_ExceptionMatches(err)  PyErr_ExceptionMatches(err)
#endif

/* GetException.proto */
#if CYTHON_FAST_THREAD_STATE
#define __Pyx_GetException(type, value, tb)  __Pyx__GetException(__pyx_tstate, type, value, tb)
static int __Pyx__GetException(PyThreadState *tstate, PyObject **type, PyObject **value, PyObject **tb);
#else
static int __Pyx_GetException(PyObject **type, PyObject **value, PyObject **tb);
#endif

/* Import.proto */
static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list, int level);

/* CodeObjectCache.proto */
typedef struct {
    PyCodeObject* code_object;
    int code_line;
} __Pyx_CodeObjectCacheEntry;
struct __Pyx_CodeObjectCache {
    int count;
    int max_count;
    __Pyx_CodeObjectCacheEntry* entries;
};
static struct __Pyx_CodeObjectCache __pyx_code_cache = {0,0,NULL};
static int __pyx_bisect_code_objects(__Pyx_CodeObjectCacheEntry* entries, int count, int code_line);
static PyCodeObject *__pyx_find_code_object(int code_line);
static void __pyx_insert_code_object(int code_line, PyCodeObject* code_object);

/* AddTraceback.proto */
static void __Pyx_AddTraceback(const char *funcname, int c_line,
                               int py_line, const char *filename);

/* BufferStructDeclare.proto */
typedef struct {
  Py_ssize_t shape, strides, suboffsets;
} __Pyx_Buf_DimInfo;
typedef struct {
  size_t refcount;
  Py_buffer pybuffer;
} __Pyx_Buffer;
typedef struct {
  __Pyx_Buffer *rcbuffer;
  char *data;
  __Pyx_Buf_DimInfo diminfo[8];
} __Pyx_LocalBuf_ND;

#if PY_MAJOR_VERSION < 3
    static int __Pyx_GetBuffer(PyObject *obj, Py_buffer *view, int flags);
    static void __Pyx_ReleaseBuffer(Py_buffer *view);
#else
    #define __Pyx_GetBuffer PyObject_GetBuffer
    #define __Pyx_ReleaseBuffer PyBuffer_Release
#endif


/* None.proto */
static Py_ssize_t __Pyx_zeros[] = {0, 0, 0, 0, 0, 0, 0, 0};
static Py_ssize_t __Pyx_minusones[] = {-1, -1, -1, -1, -1, -1, -1, -1};

/* CIntToPy.proto */
static CYTHON_INLINE PyObject* __Pyx_PyInt_From_int(int value);

/* CIntToPy.proto */
static CYTHON_INLINE PyObject* __Pyx_PyInt_From_long(long value);

/* CIntToPy.proto */
static CYTHON_INLINE PyObject* __Pyx_PyInt_From_unsigned_int(unsigned int value);

/* RealImag.proto */
#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    #define __Pyx_CREAL(z) ((z).real())
    #define __Pyx_CIMAG(z) ((z).imag())
  #else
    #define __Pyx_CREAL(z) (__real__(z))
    #define __Pyx_CIMAG(z) (__imag__(z))
  #endif
#else
    #define __Pyx_CREAL(z) ((z).real)
    #define __Pyx_CIMAG(z) ((z).imag)
#endif
#if defined(__cplusplus) && CYTHON_CCOMPLEX\
        && (defined(_WIN32) || defined(__clang__) || (defined(__GNUC__) && (__GNUC__ >= 5 || __GNUC__ == 4 && __GNUC_MINOR__ >= 4 )) || __cplusplus >= 201103)
    #define __Pyx_SET_CREAL(z,x) ((z).real(x))
    #define __Pyx_SET_CIMAG(z,y) ((z).imag(y))
#else
    #define __Pyx_SET_CREAL(z,x) __Pyx_CREAL(z) = (x)
    #define __Pyx_SET_CIMAG(z,y) __Pyx_CIMAG(z) = (y)
#endif

/* Arithmetic.proto */
#if CYTHON_CCOMPLEX
    #define __Pyx_c_eq_float(a, b)   ((a)==(b))
    #define __Pyx_c_sum_float(a, b)  ((a)+(b))
    #define __Pyx_c_diff_float(a, b) ((a)-(b))
    #define __Pyx_c_prod_float(a, b) ((a)*(b))
    #define __Pyx_c_quot_float(a, b) ((a)/(b))
    #define __Pyx_c_neg_float(a)     (-(a))
  #ifdef __cplusplus
    #define __Pyx_c_is_zero_float(z) ((z)==(float)0)
    #define __Pyx_c_conj_float(z)    (::std::conj(z))
    #if 1
        #define __Pyx_c_abs_float(z)     (::std::abs(z))
        #define __Pyx_c_pow_float(a, b)  (::std::pow(a, b))
    #endif
  #else
    #define __Pyx_c_is_zero_float(z) ((z)==0)
    #define __Pyx_c_conj_float(z)    (conjf(z))
    #if 1
        #define __Pyx_c_abs_float(z)     (cabsf(z))
        #define __Pyx_c_pow_float(a, b)  (cpowf(a, b))
    #endif
 #endif
#else
    static CYTHON_INLINE int __Pyx_c_eq_float(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_sum_float(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_diff_float(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_prod_float(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_quot_float(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_neg_float(__pyx_t_float_complex);
    static CYTHON_INLINE int __Pyx_c_is_zero_float(__pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_conj_float(__pyx_t_float_complex);
    #if 1
        static CYTHON_INLINE float __Pyx_c_abs_float(__pyx_t_float_complex);
        static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_pow_float(__pyx_t_float_complex, __pyx_t_float_complex);
    #endif
#endif

/* Arithmetic.proto */
#if CYTHON_CCOMPLEX
    #define __Pyx_c_eq_double(a, b)   ((a)==(b))
    #define __Pyx_c_sum_double(a, b)  ((a)+(b))
    #define __Pyx_c_diff_double(a, b) ((a)-(b))
    #define __Pyx_c_prod_double(a, b) ((a)*(b))
    #define __Pyx_c_quot_double(a, b) ((a)/(b))
    #define __Pyx_c_neg_double(a)     (-(a))
  #ifdef __cplusplus
    #define __Pyx_c_is_zero_double(z) ((z)==(double)0)
    #define __Pyx_c_conj_double(z)    (::std::conj(z))
    #if 1
        #define __Pyx_c_abs_double(z)     (::std::abs(z))
        #define __Pyx_c_pow_double(a, b)  (::std::pow(a, b))
    #endif
  #else
    #define __Pyx_c_is_zero_double(z) ((z)==0)
    #define __Pyx_c_conj_double(z)    (conj(z))
    #if 1
        #define __Pyx_c_abs_double(z)     (cabs(z))
        #define __Pyx_c_pow_double(a, b)  (cpow(a, b))
    #endif
 #endif
#else
    static CYTHON_INLINE int __Pyx_c_eq_double(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_sum_double(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_diff_double(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_prod_double(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_quot_double(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_neg_double(__pyx_t_double_complex);
    static CYTHON_INLINE int __Pyx_c_is_zero_double(__pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_conj_double(__pyx_t_double_complex);
    #if 1
        static CYTHON_INLINE double __Pyx_c_abs_double(__pyx_t_double_complex);
        static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_pow_double(__pyx_t_double_complex, __pyx_t_double_complex);
    #endif
#endif

/* CIntToPy.proto */
static CYTHON_INLINE PyObject* __Pyx_PyInt_From_enum__NPY_TYPES(enum NPY_TYPES value);

/* CIntFromPy.proto */
static CYTHON_INLINE unsigned int __Pyx_PyInt_As_unsigned_int(PyObject *);

/* CIntFromPy.proto */
static CYTHON_INLINE int __Pyx_PyInt_As_int(PyObject *);

/* CIntFromPy.proto */
static CYTHON_INLINE long __Pyx_PyInt_As_long(PyObject *);

/* CheckBinaryVersion.proto */
static int __Pyx_check_binary_version(void);

/* PyIdentifierFromString.proto */
#if !defined(__Pyx_PyIdentifier_FromString)
#if PY_MAJOR_VERSION < 3
  #define __Pyx_PyIdentifier_FromString(s) PyString_FromString(s)
#else
  #define __Pyx_PyIdentifier_FromString(s) PyUnicode_FromString(s)
#endif
#endif

/* ModuleImport.proto */
static PyObject *__Pyx_ImportModule(const char *name);

/* TypeImport.proto */
static PyTypeObject *__Pyx_ImportType(const char *module_name, const char *class_name, size_t size, int strict);

/* InitStrings.proto */
static int __Pyx_InitStrings(__Pyx_StringTabEntry *t);


/* Module declarations from 'cpython.buffer' */

/* Module declarations from 'libc.string' */

/* Module declarations from 'libc.stdio' */

/* Module declarations from '__builtin__' */

/* Module declarations from 'cpython.type' */
static PyTypeObject *__pyx_ptype_7cpython_4type_type = 0;

/* Module declarations from 'cpython' */

/* Module declarations from 'cpython.object' */

/* Module declarations from 'cpython.ref' */

/* Module declarations from 'libc.stdlib' */

/* Module declarations from 'numpy' */

/* Module declarations from 'numpy' */
static PyTypeObject *__pyx_ptype_5numpy_dtype = 0;
static PyTypeObject *__pyx_ptype_5numpy_flatiter = 0;
static PyTypeObject *__pyx_ptype_5numpy_broadcast = 0;
static PyTypeObject *__pyx_ptype_5numpy_ndarray = 0;
static PyTypeObject *__pyx_ptype_5numpy_ufunc = 0;
static CYTHON_INLINE char *__pyx_f_5numpy__util_dtypestring(PyArray_Descr *, char *, char *, int *); /*proto*/

/* Module declarations from 'nms.cpu_nms' */
static CYTHON_INLINE __pyx_t_5numpy_float32_t __pyx_f_3nms_7cpu_nms_max(__pyx_t_5numpy_float32_t, __pyx_t_5numpy_float32_t); /*proto*/
static CYTHON_INLINE __pyx_t_5numpy_float32_t __pyx_f_3nms_7cpu_nms_min(__pyx_t_5numpy_float32_t, __pyx_t_5numpy_float32_t); /*proto*/
static __Pyx_TypeInfo __Pyx_TypeInfo_nn___pyx_t_5numpy_float32_t = { "float32_t", NULL, sizeof(__pyx_t_5numpy_float32_t), { 0 }, 0, 'R', 0, 0 };
static __Pyx_TypeInfo __Pyx_TypeInfo_nn___pyx_t_5numpy_int_t = { "int_t", NULL, sizeof(__pyx_t_5numpy_int_t), { 0 }, 0, IS_UNSIGNED(__pyx_t_5numpy_int_t) ? 'U' : 'I', IS_UNSIGNED(__pyx_t_5numpy_int_t), 0 };
static __Pyx_TypeInfo __Pyx_TypeInfo_float = { "float", NULL, sizeof(float), { 0 }, 0, 'R', 0, 0 };
#define __Pyx_MODULE_NAME "nms.cpu_nms"
int __pyx_module_is_main_nms__cpu_nms = 0;

/* Implementation of 'nms.cpu_nms' */
static PyObject *__pyx_builtin_range;
static PyObject *__pyx_builtin_ValueError;
static PyObject *__pyx_builtin_RuntimeError;
static PyObject *__pyx_builtin_ImportError;
static const char __pyx_k_N[] = "N";
static const char __pyx_k_h[] = "h";
static const char __pyx_k_i[] = "_i";
static const char __pyx_k_j[] = "_j";
static const char __pyx_k_s[] = "s";
static const char __pyx_k_w[] = "w";
static const char __pyx_k_Nt[] = "Nt";
static const char __pyx_k_ih[] = "ih";
static const char __pyx_k_iw[] = "iw";
static const char __pyx_k_np[] = "np";
static const char __pyx_k_ov[] = "ov";
static const char __pyx_k_ts[] = "ts";
static const char __pyx_k_ua[] = "ua";
static const char __pyx_k_x1[] = "x1";
static const char __pyx_k_x2[] = "x2";
static const char __pyx_k_y1[] = "y1";
static const char __pyx_k_y2[] = "y2";
static const char __pyx_k_exp[] = "exp";
static const char __pyx_k_i_2[] = "i";
static const char __pyx_k_int[] = "int";
static const char __pyx_k_ix1[] = "ix1";
static const char __pyx_k_ix2[] = "ix2";
static const char __pyx_k_iy1[] = "iy1";
static const char __pyx_k_iy2[] = "iy2";
static const char __pyx_k_j_2[] = "j";
static const char __pyx_k_ovr[] = "ovr";
static const char __pyx_k_pos[] = "pos";
static const char __pyx_k_tx1[] = "tx1";
static const char __pyx_k_tx2[] = "tx2";
static const char __pyx_k_ty1[] = "ty1";
static const char __pyx_k_ty2[] = "ty2";
static const char __pyx_k_xx1[] = "xx1";
static const char __pyx_k_xx2[] = "xx2";
static const char __pyx_k_yy1[] = "yy1";
static const char __pyx_k_yy2[] = "yy2";
static const char __pyx_k_area[] = "area";
static const char __pyx_k_dets[] = "dets";
static const char __pyx_k_keep[] = "keep";
static const char __pyx_k_main[] = "__main__";
static const char __pyx_k_test[] = "__test__";
static const char __pyx_k_areas[] = "areas";
static const char __pyx_k_boxes[] = "boxes";
static const char __pyx_k_dtype[] = "dtype";
static const char __pyx_k_iarea[] = "iarea";
static const char __pyx_k_inter[] = "inter";
static const char __pyx_k_ndets[] = "ndets";
static const char __pyx_k_numpy[] = "numpy";
static const char __pyx_k_order[] = "order";
static const char __pyx_k_range[] = "range";
static const char __pyx_k_sigma[] = "sigma";
static const char __pyx_k_zeros[] = "zeros";
static const char __pyx_k_import[] = "__import__";
static const char __pyx_k_maxpos[] = "maxpos";
static const char __pyx_k_method[] = "method";
static const char __pyx_k_scores[] = "scores";
static const char __pyx_k_thresh[] = "thresh";
static const char __pyx_k_weight[] = "weight";
static const char __pyx_k_argsort[] = "argsort";
static const char __pyx_k_cpu_nms[] = "cpu_nms";
static const char __pyx_k_box_area[] = "box_area";
static const char __pyx_k_maxscore[] = "maxscore";
static const char __pyx_k_threshold[] = "threshold";
static const char __pyx_k_ValueError[] = "ValueError";
static const char __pyx_k_suppressed[] = "suppressed";
static const char __pyx_k_ImportError[] = "ImportError";
static const char __pyx_k_nms_cpu_nms[] = "nms.cpu_nms";
static const char __pyx_k_RuntimeError[] = "RuntimeError";
static const char __pyx_k_cpu_soft_nms[] = "cpu_soft_nms";
static const char __pyx_k_ndarray_is_not_C_contiguous[] = "ndarray is not C contiguous";
static const char __pyx_k_home_messi_RFBNet_utils_nms_cpu[] = "/home/messi/RFBNet/utils/nms/cpu_nms.pyx";
static const char __pyx_k_numpy_core_multiarray_failed_to[] = "numpy.core.multiarray failed to import";
static const char __pyx_k_unknown_dtype_code_in_numpy_pxd[] = "unknown dtype code in numpy.pxd (%d)";
static const char __pyx_k_Format_string_allocated_too_shor[] = "Format string allocated too short, see comment in numpy.pxd";
static const char __pyx_k_Non_native_byte_order_not_suppor[] = "Non-native byte order not supported";
static const char __pyx_k_ndarray_is_not_Fortran_contiguou[] = "ndarray is not Fortran contiguous";
static const char __pyx_k_numpy_core_umath_failed_to_impor[] = "numpy.core.umath failed to import";
static const char __pyx_k_Format_string_allocated_too_shor_2[] = "Format string allocated too short.";
static PyObject *__pyx_kp_u_Format_string_allocated_too_shor;
static PyObject *__pyx_kp_u_Format_string_allocated_too_shor_2;
static PyObject *__pyx_n_s_ImportError;
static PyObject *__pyx_n_s_N;
static PyObject *__pyx_kp_u_Non_native_byte_order_not_suppor;
static PyObject *__pyx_n_s_Nt;
static PyObject *__pyx_n_s_RuntimeError;
static PyObject *__pyx_n_s_ValueError;
static PyObject *__pyx_n_s_area;
static PyObject *__pyx_n_s_areas;
static PyObject *__pyx_n_s_argsort;
static PyObject *__pyx_n_s_box_area;
static PyObject *__pyx_n_s_boxes;
static PyObject *__pyx_n_s_cpu_nms;
static PyObject *__pyx_n_s_cpu_soft_nms;
static PyObject *__pyx_n_s_dets;
static PyObject *__pyx_n_s_dtype;
static PyObject *__pyx_n_s_exp;
static PyObject *__pyx_n_s_h;
static PyObject *__pyx_kp_s_home_messi_RFBNet_utils_nms_cpu;
static PyObject *__pyx_n_s_i;
static PyObject *__pyx_n_s_i_2;
static PyObject *__pyx_n_s_iarea;
static PyObject *__pyx_n_s_ih;
static PyObject *__pyx_n_s_import;
static PyObject *__pyx_n_s_int;
static PyObject *__pyx_n_s_inter;
static PyObject *__pyx_n_s_iw;
static PyObject *__pyx_n_s_ix1;
static PyObject *__pyx_n_s_ix2;
static PyObject *__pyx_n_s_iy1;
static PyObject *__pyx_n_s_iy2;
static PyObject *__pyx_n_s_j;
static PyObject *__pyx_n_s_j_2;
static PyObject *__pyx_n_s_keep;
static PyObject *__pyx_n_s_main;
static PyObject *__pyx_n_s_maxpos;
static PyObject *__pyx_n_s_maxscore;
static PyObject *__pyx_n_s_method;
static PyObject *__pyx_kp_u_ndarray_is_not_C_contiguous;
static PyObject *__pyx_kp_u_ndarray_is_not_Fortran_contiguou;
static PyObject *__pyx_n_s_ndets;
static PyObject *__pyx_n_s_nms_cpu_nms;
static PyObject *__pyx_n_s_np;
static PyObject *__pyx_n_s_numpy;
static PyObject *__pyx_kp_s_numpy_core_multiarray_failed_to;
static PyObject *__pyx_kp_s_numpy_core_umath_failed_to_impor;
static PyObject *__pyx_n_s_order;
static PyObject *__pyx_n_s_ov;
static PyObject *__pyx_n_s_ovr;
static PyObject *__pyx_n_s_pos;
static PyObject *__pyx_n_s_range;
static PyObject *__pyx_n_s_s;
static PyObject *__pyx_n_s_scores;
static PyObject *__pyx_n_s_sigma;
static PyObject *__pyx_n_s_suppressed;
static PyObject *__pyx_n_s_test;
static PyObject *__pyx_n_s_thresh;
static PyObject *__pyx_n_s_threshold;
static PyObject *__pyx_n_s_ts;
static PyObject *__pyx_n_s_tx1;
static PyObject *__pyx_n_s_tx2;
static PyObject *__pyx_n_s_ty1;
static PyObject *__pyx_n_s_ty2;
static PyObject *__pyx_n_s_ua;
static PyObject *__pyx_kp_u_unknown_dtype_code_in_numpy_pxd;
static PyObject *__pyx_n_s_w;
static PyObject *__pyx_n_s_weight;
static PyObject *__pyx_n_s_x1;
static PyObject *__pyx_n_s_x2;
static PyObject *__pyx_n_s_xx1;
static PyObject *__pyx_n_s_xx2;
static PyObject *__pyx_n_s_y1;
static PyObject *__pyx_n_s_y2;
static PyObject *__pyx_n_s_yy1;
static PyObject *__pyx_n_s_yy2;
static PyObject *__pyx_n_s_zeros;
static PyObject *__pyx_pf_3nms_7cpu_nms_cpu_nms(CYTHON_UNUSED PyObject *__pyx_self, PyArrayObject *__pyx_v_dets, PyObject *__pyx_v_thresh); /* proto */
static PyObject *__pyx_pf_3nms_7cpu_nms_2cpu_soft_nms(CYTHON_UNUSED PyObject *__pyx_self, PyArrayObject *__pyx_v_boxes, float __pyx_v_sigma, float __pyx_v_Nt, float __pyx_v_threshold, unsigned int __pyx_v_method); /* proto */
static int __pyx_pf_5numpy_7ndarray___getbuffer__(PyArrayObject *__pyx_v_self, Py_buffer *__pyx_v_info, int __pyx_v_flags); /* proto */
static void __pyx_pf_5numpy_7ndarray_2__releasebuffer__(PyArrayObject *__pyx_v_self, Py_buffer *__pyx_v_info); /* proto */
static PyObject *__pyx_int_0;
static PyObject *__pyx_int_1;
static PyObject *__pyx_int_2;
static PyObject *__pyx_int_3;
static PyObject *__pyx_int_4;
static PyObject *__pyx_int_neg_1;
static PyObject *__pyx_slice_;
static PyObject *__pyx_slice__3;
static PyObject *__pyx_slice__5;
static PyObject *__pyx_slice__7;
static PyObject *__pyx_slice__9;
static PyObject *__pyx_tuple__2;
static PyObject *__pyx_tuple__4;
static PyObject *__pyx_tuple__6;
static PyObject *__pyx_tuple__8;
static PyObject *__pyx_slice__11;
static PyObject *__pyx_tuple__10;
static PyObject *__pyx_tuple__12;
static PyObject *__pyx_tuple__13;
static PyObject *__pyx_tuple__14;
static PyObject *__pyx_tuple__15;
static PyObject *__pyx_tuple__16;
static PyObject *__pyx_tuple__17;
static PyObject *__pyx_tuple__18;
static PyObject *__pyx_tuple__19;
static PyObject *__pyx_tuple__20;
static PyObject *__pyx_tuple__21;
static PyObject *__pyx_tuple__23;
static PyObject *__pyx_codeobj__22;
static PyObject *__pyx_codeobj__24;

/* "nms/cpu_nms.pyx":11
 * cimport numpy as np
 * 
 * cdef inline np.float32_t max(np.float32_t a, np.float32_t b):             # <<<<<<<<<<<<<<
 *     return a if a >= b else b
 * 
 */

static CYTHON_INLINE __pyx_t_5numpy_float32_t __pyx_f_3nms_7cpu_nms_max(__pyx_t_5numpy_float32_t __pyx_v_a, __pyx_t_5numpy_float32_t __pyx_v_b) {
  __pyx_t_5numpy_float32_t __pyx_r;
  __Pyx_RefNannyDeclarations
  __pyx_t_5numpy_float32_t __pyx_t_1;
  __Pyx_RefNannySetupContext("max", 0);

  /* "nms/cpu_nms.pyx":12
 * 
 * cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
 *     return a if a >= b else b             # <<<<<<<<<<<<<<
 * 
 * cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
 */
  if (((__pyx_v_a >= __pyx_v_b) != 0)) {
    __pyx_t_1 = __pyx_v_a;
  } else {
    __pyx_t_1 = __pyx_v_b;
  }
  __pyx_r = __pyx_t_1;
  goto __pyx_L0;

  /* "nms/cpu_nms.pyx":11
 * cimport numpy as np
 * 
 * cdef inline np.float32_t max(np.float32_t a, np.float32_t b):             # <<<<<<<<<<<<<<
 *     return a if a >= b else b
 * 
 */

  /* function exit code */
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "nms/cpu_nms.pyx":14
 *     return a if a >= b else b
 * 
 * cdef inline np.float32_t min(np.float32_t a, np.float32_t b):             # <<<<<<<<<<<<<<
 *     return a if a <= b else b
 * 
 */

static CYTHON_INLINE __pyx_t_5numpy_float32_t __pyx_f_3nms_7cpu_nms_min(__pyx_t_5numpy_float32_t __pyx_v_a, __pyx_t_5numpy_float32_t __pyx_v_b) {
  __pyx_t_5numpy_float32_t __pyx_r;
  __Pyx_RefNannyDeclarations
  __pyx_t_5numpy_float32_t __pyx_t_1;
  __Pyx_RefNannySetupContext("min", 0);

  /* "nms/cpu_nms.pyx":15
 * 
 * cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
 *     return a if a <= b else b             # <<<<<<<<<<<<<<
 * 
 * def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
 */
  if (((__pyx_v_a <= __pyx_v_b) != 0)) {
    __pyx_t_1 = __pyx_v_a;
  } else {
    __pyx_t_1 = __pyx_v_b;
  }
  __pyx_r = __pyx_t_1;
  goto __pyx_L0;

  /* "nms/cpu_nms.pyx":14
 *     return a if a >= b else b
 * 
 * cdef inline np.float32_t min(np.float32_t a, np.float32_t b):             # <<<<<<<<<<<<<<
 *     return a if a <= b else b
 * 
 */

  /* function exit code */
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "nms/cpu_nms.pyx":17
 *     return a if a <= b else b
 * 
 * def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
 *     cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
 */

/* Python wrapper */
static PyObject *__pyx_pw_3nms_7cpu_nms_1cpu_nms(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_mdef_3nms_7cpu_nms_1cpu_nms = {"cpu_nms", (PyCFunction)__pyx_pw_3nms_7cpu_nms_1cpu_nms, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_pw_3nms_7cpu_nms_1cpu_nms(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyArrayObject *__pyx_v_dets = 0;
  PyObject *__pyx_v_thresh = 0;
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("cpu_nms (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&__pyx_n_s_dets,&__pyx_n_s_thresh,0};
    PyObject* values[2] = {0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = PyTuple_GET_SIZE(__pyx_args);
      switch (pos_args) {
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s_dets)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        if (likely((values[1] = PyDict_GetItem(__pyx_kwds, __pyx_n_s_thresh)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("cpu_nms", 1, 2, 2, 1); __PYX_ERR(0, 17, __pyx_L3_error)
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "cpu_nms") < 0)) __PYX_ERR(0, 17, __pyx_L3_error)
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 2) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
    }
    __pyx_v_dets = ((PyArrayObject *)values[0]);
    __pyx_v_thresh = ((PyObject*)values[1]);
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("cpu_nms", 1, 2, 2, PyTuple_GET_SIZE(__pyx_args)); __PYX_ERR(0, 17, __pyx_L3_error)
  __pyx_L3_error:;
  __Pyx_AddTraceback("nms.cpu_nms.cpu_nms", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_dets), __pyx_ptype_5numpy_ndarray, 1, "dets", 0))) __PYX_ERR(0, 17, __pyx_L1_error)
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_thresh), (&PyFloat_Type), 1, "thresh", 1))) __PYX_ERR(0, 17, __pyx_L1_error)
  __pyx_r = __pyx_pf_3nms_7cpu_nms_cpu_nms(__pyx_self, __pyx_v_dets, __pyx_v_thresh);

  /* function exit code */
  goto __pyx_L0;
  __pyx_L1_error:;
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_3nms_7cpu_nms_cpu_nms(CYTHON_UNUSED PyObject *__pyx_self, PyArrayObject *__pyx_v_dets, PyObject *__pyx_v_thresh) {
  PyArrayObject *__pyx_v_x1 = 0;
  PyArrayObject *__pyx_v_y1 = 0;
  PyArrayObject *__pyx_v_x2 = 0;
  PyArrayObject *__pyx_v_y2 = 0;
  PyArrayObject *__pyx_v_scores = 0;
  PyArrayObject *__pyx_v_areas = 0;
  PyArrayObject *__pyx_v_order = 0;
  int __pyx_v_ndets;
  PyArrayObject *__pyx_v_suppressed = 0;
  int __pyx_v__i;
  int __pyx_v__j;
  int __pyx_v_i;
  int __pyx_v_j;
  __pyx_t_5numpy_float32_t __pyx_v_ix1;
  __pyx_t_5numpy_float32_t __pyx_v_iy1;
  __pyx_t_5numpy_float32_t __pyx_v_ix2;
  __pyx_t_5numpy_float32_t __pyx_v_iy2;
  __pyx_t_5numpy_float32_t __pyx_v_iarea;
  __pyx_t_5numpy_float32_t __pyx_v_xx1;
  __pyx_t_5numpy_float32_t __pyx_v_yy1;
  __pyx_t_5numpy_float32_t __pyx_v_xx2;
  __pyx_t_5numpy_float32_t __pyx_v_yy2;
  __pyx_t_5numpy_float32_t __pyx_v_w;
  __pyx_t_5numpy_float32_t __pyx_v_h;
  __pyx_t_5numpy_float32_t __pyx_v_inter;
  __pyx_t_5numpy_float32_t __pyx_v_ovr;
  PyObject *__pyx_v_keep = NULL;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_areas;
  __Pyx_Buffer __pyx_pybuffer_areas;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_dets;
  __Pyx_Buffer __pyx_pybuffer_dets;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_order;
  __Pyx_Buffer __pyx_pybuffer_order;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_scores;
  __Pyx_Buffer __pyx_pybuffer_scores;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_suppressed;
  __Pyx_Buffer __pyx_pybuffer_suppressed;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_x1;
  __Pyx_Buffer __pyx_pybuffer_x1;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_x2;
  __Pyx_Buffer __pyx_pybuffer_x2;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_y1;
  __Pyx_Buffer __pyx_pybuffer_y1;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_y2;
  __Pyx_Buffer __pyx_pybuffer_y2;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyArrayObject *__pyx_t_2 = NULL;
  PyArrayObject *__pyx_t_3 = NULL;
  PyArrayObject *__pyx_t_4 = NULL;
  PyArrayObject *__pyx_t_5 = NULL;
  PyArrayObject *__pyx_t_6 = NULL;
  PyObject *__pyx_t_7 = NULL;
  PyObject *__pyx_t_8 = NULL;
  PyArrayObject *__pyx_t_9 = NULL;
  PyArrayObject *__pyx_t_10 = NULL;
  PyObject *__pyx_t_11 = NULL;
  PyObject *__pyx_t_12 = NULL;
  PyArrayObject *__pyx_t_13 = NULL;
  int __pyx_t_14;
  int __pyx_t_15;
  Py_ssize_t __pyx_t_16;
  int __pyx_t_17;
  Py_ssize_t __pyx_t_18;
  int __pyx_t_19;
  int __pyx_t_20;
  Py_ssize_t __pyx_t_21;
  Py_ssize_t __pyx_t_22;
  Py_ssize_t __pyx_t_23;
  Py_ssize_t __pyx_t_24;
  Py_ssize_t __pyx_t_25;
  int __pyx_t_26;
  Py_ssize_t __pyx_t_27;
  int __pyx_t_28;
  Py_ssize_t __pyx_t_29;
  Py_ssize_t __pyx_t_30;
  Py_ssize_t __pyx_t_31;
  Py_ssize_t __pyx_t_32;
  Py_ssize_t __pyx_t_33;
  Py_ssize_t __pyx_t_34;
  __pyx_t_5numpy_float32_t __pyx_t_35;
  Py_ssize_t __pyx_t_36;
  __Pyx_RefNannySetupContext("cpu_nms", 0);
  __pyx_pybuffer_x1.pybuffer.buf = NULL;
  __pyx_pybuffer_x1.refcount = 0;
  __pyx_pybuffernd_x1.data = NULL;
  __pyx_pybuffernd_x1.rcbuffer = &__pyx_pybuffer_x1;
  __pyx_pybuffer_y1.pybuffer.buf = NULL;
  __pyx_pybuffer_y1.refcount = 0;
  __pyx_pybuffernd_y1.data = NULL;
  __pyx_pybuffernd_y1.rcbuffer = &__pyx_pybuffer_y1;
  __pyx_pybuffer_x2.pybuffer.buf = NULL;
  __pyx_pybuffer_x2.refcount = 0;
  __pyx_pybuffernd_x2.data = NULL;
  __pyx_pybuffernd_x2.rcbuffer = &__pyx_pybuffer_x2;
  __pyx_pybuffer_y2.pybuffer.buf = NULL;
  __pyx_pybuffer_y2.refcount = 0;
  __pyx_pybuffernd_y2.data = NULL;
  __pyx_pybuffernd_y2.rcbuffer = &__pyx_pybuffer_y2;
  __pyx_pybuffer_scores.pybuffer.buf = NULL;
  __pyx_pybuffer_scores.refcount = 0;
  __pyx_pybuffernd_scores.data = NULL;
  __pyx_pybuffernd_scores.rcbuffer = &__pyx_pybuffer_scores;
  __pyx_pybuffer_areas.pybuffer.buf = NULL;
  __pyx_pybuffer_areas.refcount = 0;
  __pyx_pybuffernd_areas.data = NULL;
  __pyx_pybuffernd_areas.rcbuffer = &__pyx_pybuffer_areas;
  __pyx_pybuffer_order.pybuffer.buf = NULL;
  __pyx_pybuffer_order.refcount = 0;
  __pyx_pybuffernd_order.data = NULL;
  __pyx_pybuffernd_order.rcbuffer = &__pyx_pybuffer_order;
  __pyx_pybuffer_suppressed.pybuffer.buf = NULL;
  __pyx_pybuffer_suppressed.refcount = 0;
  __pyx_pybuffernd_suppressed.data = NULL;
  __pyx_pybuffernd_suppressed.rcbuffer = &__pyx_pybuffer_suppressed;
  __pyx_pybuffer_dets.pybuffer.buf = NULL;
  __pyx_pybuffer_dets.refcount = 0;
  __pyx_pybuffernd_dets.data = NULL;
  __pyx_pybuffernd_dets.rcbuffer = &__pyx_pybuffer_dets;
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_dets.rcbuffer->pybuffer, (PyObject*)__pyx_v_dets, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float32_t, PyBUF_FORMAT| PyBUF_STRIDES, 2, 0, __pyx_stack) == -1)) __PYX_ERR(0, 17, __pyx_L1_error)
  }
  __pyx_pybuffernd_dets.diminfo[0].strides = __pyx_pybuffernd_dets.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_dets.diminfo[0].shape = __pyx_pybuffernd_dets.rcbuffer->pybuffer.shape[0]; __pyx_pybuffernd_dets.diminfo[1].strides = __pyx_pybuffernd_dets.rcbuffer->pybuffer.strides[1]; __pyx_pybuffernd_dets.diminfo[1].shape = __pyx_pybuffernd_dets.rcbuffer->pybuffer.shape[1];

  /* "nms/cpu_nms.pyx":18
 * 
 * def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
 *     cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
 *     cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
 */
  __pyx_t_1 = PyObject_GetItem(((PyObject *)__pyx_v_dets), __pyx_tuple__2); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 18, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (!(likely(((__pyx_t_1) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_1, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 18, __pyx_L1_error)
  __pyx_t_2 = ((PyArrayObject *)__pyx_t_1);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_x1.rcbuffer->pybuffer, (PyObject*)__pyx_t_2, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float32_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_x1 = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_pybuffernd_x1.rcbuffer->pybuffer.buf = NULL;
      __PYX_ERR(0, 18, __pyx_L1_error)
    } else {__pyx_pybuffernd_x1.diminfo[0].strides = __pyx_pybuffernd_x1.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_x1.diminfo[0].shape = __pyx_pybuffernd_x1.rcbuffer->pybuffer.shape[0];
    }
  }
  __pyx_t_2 = 0;
  __pyx_v_x1 = ((PyArrayObject *)__pyx_t_1);
  __pyx_t_1 = 0;

  /* "nms/cpu_nms.pyx":19
 * def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
 *     cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
 *     cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
 *     cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
 */
  __pyx_t_1 = PyObject_GetItem(((PyObject *)__pyx_v_dets), __pyx_tuple__4); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 19, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (!(likely(((__pyx_t_1) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_1, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 19, __pyx_L1_error)
  __pyx_t_3 = ((PyArrayObject *)__pyx_t_1);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_y1.rcbuffer->pybuffer, (PyObject*)__pyx_t_3, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float32_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_y1 = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_pybuffernd_y1.rcbuffer->pybuffer.buf = NULL;
      __PYX_ERR(0, 19, __pyx_L1_error)
    } else {__pyx_pybuffernd_y1.diminfo[0].strides = __pyx_pybuffernd_y1.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_y1.diminfo[0].shape = __pyx_pybuffernd_y1.rcbuffer->pybuffer.shape[0];
    }
  }
  __pyx_t_3 = 0;
  __pyx_v_y1 = ((PyArrayObject *)__pyx_t_1);
  __pyx_t_1 = 0;

  /* "nms/cpu_nms.pyx":20
 *     cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
 *     cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
 *     cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
 *     cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]
 */
  __pyx_t_1 = PyObject_GetItem(((PyObject *)__pyx_v_dets), __pyx_tuple__6); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 20, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (!(likely(((__pyx_t_1) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_1, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 20, __pyx_L1_error)
  __pyx_t_4 = ((PyArrayObject *)__pyx_t_1);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_x2.rcbuffer->pybuffer, (PyObject*)__pyx_t_4, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float32_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_x2 = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_pybuffernd_x2.rcbuffer->pybuffer.buf = NULL;
      __PYX_ERR(0, 20, __pyx_L1_error)
    } else {__pyx_pybuffernd_x2.diminfo[0].strides = __pyx_pybuffernd_x2.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_x2.diminfo[0].shape = __pyx_pybuffernd_x2.rcbuffer->pybuffer.shape[0];
    }
  }
  __pyx_t_4 = 0;
  __pyx_v_x2 = ((PyArrayObject *)__pyx_t_1);
  __pyx_t_1 = 0;

  /* "nms/cpu_nms.pyx":21
 *     cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
 *     cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
 *     cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]
 * 
 */
  __pyx_t_1 = PyObject_GetItem(((PyObject *)__pyx_v_dets), __pyx_tuple__8); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 21, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (!(likely(((__pyx_t_1) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_1, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 21, __pyx_L1_error)
  __pyx_t_5 = ((PyArrayObject *)__pyx_t_1);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_y2.rcbuffer->pybuffer, (PyObject*)__pyx_t_5, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float32_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_y2 = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_pybuffernd_y2.rcbuffer->pybuffer.buf = NULL;
      __PYX_ERR(0, 21, __pyx_L1_error)
    } else {__pyx_pybuffernd_y2.diminfo[0].strides = __pyx_pybuffernd_y2.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_y2.diminfo[0].shape = __pyx_pybuffernd_y2.rcbuffer->pybuffer.shape[0];
    }
  }
  __pyx_t_5 = 0;
  __pyx_v_y2 = ((PyArrayObject *)__pyx_t_1);
  __pyx_t_1 = 0;

  /* "nms/cpu_nms.pyx":22
 *     cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
 *     cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
 *     cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
 */
  __pyx_t_1 = PyObject_GetItem(((PyObject *)__pyx_v_dets), __pyx_tuple__10); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 22, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (!(likely(((__pyx_t_1) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_1, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 22, __pyx_L1_error)
  __pyx_t_6 = ((PyArrayObject *)__pyx_t_1);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_scores.rcbuffer->pybuffer, (PyObject*)__pyx_t_6, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float32_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_scores = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_pybuffernd_scores.rcbuffer->pybuffer.buf = NULL;
      __PYX_ERR(0, 22, __pyx_L1_error)
    } else {__pyx_pybuffernd_scores.diminfo[0].strides = __pyx_pybuffernd_scores.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_scores.diminfo[0].shape = __pyx_pybuffernd_scores.rcbuffer->pybuffer.shape[0];
    }
  }
  __pyx_t_6 = 0;
  __pyx_v_scores = ((PyArrayObject *)__pyx_t_1);
  __pyx_t_1 = 0;

  /* "nms/cpu_nms.pyx":24
 *     cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]
 * 
 *     cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]
 * 
 */
  __pyx_t_1 = PyNumber_Subtract(((PyObject *)__pyx_v_x2), ((PyObject *)__pyx_v_x1)); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 24, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_7 = __Pyx_PyInt_AddObjC(__pyx_t_1, __pyx_int_1, 1, 0); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 24, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_7);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyNumber_Subtract(((PyObject *)__pyx_v_y2), ((PyObject *)__pyx_v_y1)); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 24, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_8 = __Pyx_PyInt_AddObjC(__pyx_t_1, __pyx_int_1, 1, 0); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 24, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_8);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyNumber_Multiply(__pyx_t_7, __pyx_t_8); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 24, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
  __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
  if (!(likely(((__pyx_t_1) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_1, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 24, __pyx_L1_error)
  __pyx_t_9 = ((PyArrayObject *)__pyx_t_1);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_areas.rcbuffer->pybuffer, (PyObject*)__pyx_t_9, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float32_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_areas = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_pybuffernd_areas.rcbuffer->pybuffer.buf = NULL;
      __PYX_ERR(0, 24, __pyx_L1_error)
    } else {__pyx_pybuffernd_areas.diminfo[0].strides = __pyx_pybuffernd_areas.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_areas.diminfo[0].shape = __pyx_pybuffernd_areas.rcbuffer->pybuffer.shape[0];
    }
  }
  __pyx_t_9 = 0;
  __pyx_v_areas = ((PyArrayObject *)__pyx_t_1);
  __pyx_t_1 = 0;

  /* "nms/cpu_nms.pyx":25
 * 
 *     cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
 *     cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]             # <<<<<<<<<<<<<<
 * 
 *     cdef int ndets = dets.shape[0]
 */
  __pyx_t_8 = __Pyx_PyObject_GetAttrStr(((PyObject *)__pyx_v_scores), __pyx_n_s_argsort); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 25, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_8);
  __pyx_t_7 = NULL;
  if (CYTHON_UNPACK_METHODS && likely(PyMethod_Check(__pyx_t_8))) {
    __pyx_t_7 = PyMethod_GET_SELF(__pyx_t_8);
    if (likely(__pyx_t_7)) {
      PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_8);
      __Pyx_INCREF(__pyx_t_7);
      __Pyx_INCREF(function);
      __Pyx_DECREF_SET(__pyx_t_8, function);
    }
  }
  if (__pyx_t_7) {
    __pyx_t_1 = __Pyx_PyObject_CallOneArg(__pyx_t_8, __pyx_t_7); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 25, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
  } else {
    __pyx_t_1 = __Pyx_PyObject_CallNoArg(__pyx_t_8); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 25, __pyx_L1_error)
  }
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
  __pyx_t_8 = PyObject_GetItem(__pyx_t_1, __pyx_slice__11); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 25, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_8);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  if (!(likely(((__pyx_t_8) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_8, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 25, __pyx_L1_error)
  __pyx_t_10 = ((PyArrayObject *)__pyx_t_8);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_order.rcbuffer->pybuffer, (PyObject*)__pyx_t_10, &__Pyx_TypeInfo_nn___pyx_t_5numpy_int_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_order = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_pybuffernd_order.rcbuffer->pybuffer.buf = NULL;
      __PYX_ERR(0, 25, __pyx_L1_error)
    } else {__pyx_pybuffernd_order.diminfo[0].strides = __pyx_pybuffernd_order.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_order.diminfo[0].shape = __pyx_pybuffernd_order.rcbuffer->pybuffer.shape[0];
    }
  }
  __pyx_t_10 = 0;
  __pyx_v_order = ((PyArrayObject *)__pyx_t_8);
  __pyx_t_8 = 0;

  /* "nms/cpu_nms.pyx":27
 *     cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]
 * 
 *     cdef int ndets = dets.shape[0]             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.int_t, ndim=1] suppressed = \
 *             np.zeros((ndets), dtype=np.int)
 */
  __pyx_v_ndets = (__pyx_v_dets->dimensions[0]);

  /* "nms/cpu_nms.pyx":29
 *     cdef int ndets = dets.shape[0]
 *     cdef np.ndarray[np.int_t, ndim=1] suppressed = \
 *             np.zeros((ndets), dtype=np.int)             # <<<<<<<<<<<<<<
 * 
 *     # nominal indices
 */
  __pyx_t_8 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 29, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_8);
  __pyx_t_1 = __Pyx_PyObject_GetAttrStr(__pyx_t_8, __pyx_n_s_zeros); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 29, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
  __pyx_t_8 = __Pyx_PyInt_From_int(__pyx_v_ndets); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 29, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_8);
  __pyx_t_7 = PyTuple_New(1); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 29, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_7);
  __Pyx_GIVEREF(__pyx_t_8);
  PyTuple_SET_ITEM(__pyx_t_7, 0, __pyx_t_8);
  __pyx_t_8 = 0;
  __pyx_t_8 = PyDict_New(); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 29, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_8);
  __pyx_t_11 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_11)) __PYX_ERR(0, 29, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_11);
  __pyx_t_12 = __Pyx_PyObject_GetAttrStr(__pyx_t_11, __pyx_n_s_int); if (unlikely(!__pyx_t_12)) __PYX_ERR(0, 29, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_12);
  __Pyx_DECREF(__pyx_t_11); __pyx_t_11 = 0;
  if (PyDict_SetItem(__pyx_t_8, __pyx_n_s_dtype, __pyx_t_12) < 0) __PYX_ERR(0, 29, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_12); __pyx_t_12 = 0;
  __pyx_t_12 = __Pyx_PyObject_Call(__pyx_t_1, __pyx_t_7, __pyx_t_8); if (unlikely(!__pyx_t_12)) __PYX_ERR(0, 29, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_12);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
  __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
  if (!(likely(((__pyx_t_12) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_12, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 29, __pyx_L1_error)
  __pyx_t_13 = ((PyArrayObject *)__pyx_t_12);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_suppressed.rcbuffer->pybuffer, (PyObject*)__pyx_t_13, &__Pyx_TypeInfo_nn___pyx_t_5numpy_int_t, PyBUF_FORMAT| PyBUF_STRIDES| PyBUF_WRITABLE, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_suppressed = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_pybuffernd_suppressed.rcbuffer->pybuffer.buf = NULL;
      __PYX_ERR(0, 28, __pyx_L1_error)
    } else {__pyx_pybuffernd_suppressed.diminfo[0].strides = __pyx_pybuffernd_suppressed.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_suppressed.diminfo[0].shape = __pyx_pybuffernd_suppressed.rcbuffer->pybuffer.shape[0];
    }
  }
  __pyx_t_13 = 0;
  __pyx_v_suppressed = ((PyArrayObject *)__pyx_t_12);
  __pyx_t_12 = 0;

  /* "nms/cpu_nms.pyx":42
 *     cdef np.float32_t inter, ovr
 * 
 *     keep = []             # <<<<<<<<<<<<<<
 *     for _i in range(ndets):
 *         i = order[_i]
 */
  __pyx_t_12 = PyList_New(0); if (unlikely(!__pyx_t_12)) __PYX_ERR(0, 42, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_12);
  __pyx_v_keep = ((PyObject*)__pyx_t_12);
  __pyx_t_12 = 0;

  /* "nms/cpu_nms.pyx":43
 * 
 *     keep = []
 *     for _i in range(ndets):             # <<<<<<<<<<<<<<
 *         i = order[_i]
 *         if suppressed[i] == 1:
 */
  __pyx_t_14 = __pyx_v_ndets;
  for (__pyx_t_15 = 0; __pyx_t_15 < __pyx_t_14; __pyx_t_15+=1) {
    __pyx_v__i = __pyx_t_15;

    /* "nms/cpu_nms.pyx":44
 *     keep = []
 *     for _i in range(ndets):
 *         i = order[_i]             # <<<<<<<<<<<<<<
 *         if suppressed[i] == 1:
 *             continue
 */
    __pyx_t_16 = __pyx_v__i;
    __pyx_t_17 = -1;
    if (__pyx_t_16 < 0) {
      __pyx_t_16 += __pyx_pybuffernd_order.diminfo[0].shape;
      if (unlikely(__pyx_t_16 < 0)) __pyx_t_17 = 0;
    } else if (unlikely(__pyx_t_16 >= __pyx_pybuffernd_order.diminfo[0].shape)) __pyx_t_17 = 0;
    if (unlikely(__pyx_t_17 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_17);
      __PYX_ERR(0, 44, __pyx_L1_error)
    }
    __pyx_v_i = (*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_int_t *, __pyx_pybuffernd_order.rcbuffer->pybuffer.buf, __pyx_t_16, __pyx_pybuffernd_order.diminfo[0].strides));

    /* "nms/cpu_nms.pyx":45
 *     for _i in range(ndets):
 *         i = order[_i]
 *         if suppressed[i] == 1:             # <<<<<<<<<<<<<<
 *             continue
 *         keep.append(i)
 */
    __pyx_t_18 = __pyx_v_i;
    __pyx_t_17 = -1;
    if (__pyx_t_18 < 0) {
      __pyx_t_18 += __pyx_pybuffernd_suppressed.diminfo[0].shape;
      if (unlikely(__pyx_t_18 < 0)) __pyx_t_17 = 0;
    } else if (unlikely(__pyx_t_18 >= __pyx_pybuffernd_suppressed.diminfo[0].shape)) __pyx_t_17 = 0;
    if (unlikely(__pyx_t_17 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_17);
      __PYX_ERR(0, 45, __pyx_L1_error)
    }
    __pyx_t_19 = (((*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_int_t *, __pyx_pybuffernd_suppressed.rcbuffer->pybuffer.buf, __pyx_t_18, __pyx_pybuffernd_suppressed.diminfo[0].strides)) == 1) != 0);
    if (__pyx_t_19) {

      /* "nms/cpu_nms.pyx":46
 *         i = order[_i]
 *         if suppressed[i] == 1:
 *             continue             # <<<<<<<<<<<<<<
 *         keep.append(i)
 *         ix1 = x1[i]
 */
      goto __pyx_L3_continue;

      /* "nms/cpu_nms.pyx":45
 *     for _i in range(ndets):
 *         i = order[_i]
 *         if suppressed[i] == 1:             # <<<<<<<<<<<<<<
 *             continue
 *         keep.append(i)
 */
    }

    /* "nms/cpu_nms.pyx":47
 *         if suppressed[i] == 1:
 *             continue
 *         keep.append(i)             # <<<<<<<<<<<<<<
 *         ix1 = x1[i]
 *         iy1 = y1[i]
 */
    __pyx_t_12 = __Pyx_PyInt_From_int(__pyx_v_i); if (unlikely(!__pyx_t_12)) __PYX_ERR(0, 47, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_12);
    __pyx_t_20 = __Pyx_PyList_Append(__pyx_v_keep, __pyx_t_12); if (unlikely(__pyx_t_20 == -1)) __PYX_ERR(0, 47, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_12); __pyx_t_12 = 0;

    /* "nms/cpu_nms.pyx":48
 *             continue
 *         keep.append(i)
 *         ix1 = x1[i]             # <<<<<<<<<<<<<<
 *         iy1 = y1[i]
 *         ix2 = x2[i]
 */
    __pyx_t_21 = __pyx_v_i;
    __pyx_t_17 = -1;
    if (__pyx_t_21 < 0) {
      __pyx_t_21 += __pyx_pybuffernd_x1.diminfo[0].shape;
      if (unlikely(__pyx_t_21 < 0)) __pyx_t_17 = 0;
    } else if (unlikely(__pyx_t_21 >= __pyx_pybuffernd_x1.diminfo[0].shape)) __pyx_t_17 = 0;
    if (unlikely(__pyx_t_17 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_17);
      __PYX_ERR(0, 48, __pyx_L1_error)
    }
    __pyx_v_ix1 = (*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_x1.rcbuffer->pybuffer.buf, __pyx_t_21, __pyx_pybuffernd_x1.diminfo[0].strides));

    /* "nms/cpu_nms.pyx":49
 *         keep.append(i)
 *         ix1 = x1[i]
 *         iy1 = y1[i]             # <<<<<<<<<<<<<<
 *         ix2 = x2[i]
 *         iy2 = y2[i]
 */
    __pyx_t_22 = __pyx_v_i;
    __pyx_t_17 = -1;
    if (__pyx_t_22 < 0) {
      __pyx_t_22 += __pyx_pybuffernd_y1.diminfo[0].shape;
      if (unlikely(__pyx_t_22 < 0)) __pyx_t_17 = 0;
    } else if (unlikely(__pyx_t_22 >= __pyx_pybuffernd_y1.diminfo[0].shape)) __pyx_t_17 = 0;
    if (unlikely(__pyx_t_17 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_17);
      __PYX_ERR(0, 49, __pyx_L1_error)
    }
    __pyx_v_iy1 = (*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_y1.rcbuffer->pybuffer.buf, __pyx_t_22, __pyx_pybuffernd_y1.diminfo[0].strides));

    /* "nms/cpu_nms.pyx":50
 *         ix1 = x1[i]
 *         iy1 = y1[i]
 *         ix2 = x2[i]             # <<<<<<<<<<<<<<
 *         iy2 = y2[i]
 *         iarea = areas[i]
 */
    __pyx_t_23 = __pyx_v_i;
    __pyx_t_17 = -1;
    if (__pyx_t_23 < 0) {
      __pyx_t_23 += __pyx_pybuffernd_x2.diminfo[0].shape;
      if (unlikely(__pyx_t_23 < 0)) __pyx_t_17 = 0;
    } else if (unlikely(__pyx_t_23 >= __pyx_pybuffernd_x2.diminfo[0].shape)) __pyx_t_17 = 0;
    if (unlikely(__pyx_t_17 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_17);
      __PYX_ERR(0, 50, __pyx_L1_error)
    }
    __pyx_v_ix2 = (*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_x2.rcbuffer->pybuffer.buf, __pyx_t_23, __pyx_pybuffernd_x2.diminfo[0].strides));

    /* "nms/cpu_nms.pyx":51
 *         iy1 = y1[i]
 *         ix2 = x2[i]
 *         iy2 = y2[i]             # <<<<<<<<<<<<<<
 *         iarea = areas[i]
 *         for _j in range(_i + 1, ndets):
 */
    __pyx_t_24 = __pyx_v_i;
    __pyx_t_17 = -1;
    if (__pyx_t_24 < 0) {
      __pyx_t_24 += __pyx_pybuffernd_y2.diminfo[0].shape;
      if (unlikely(__pyx_t_24 < 0)) __pyx_t_17 = 0;
    } else if (unlikely(__pyx_t_24 >= __pyx_pybuffernd_y2.diminfo[0].shape)) __pyx_t_17 = 0;
    if (unlikely(__pyx_t_17 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_17);
      __PYX_ERR(0, 51, __pyx_L1_error)
    }
    __pyx_v_iy2 = (*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_y2.rcbuffer->pybuffer.buf, __pyx_t_24, __pyx_pybuffernd_y2.diminfo[0].strides));

    /* "nms/cpu_nms.pyx":52
 *         ix2 = x2[i]
 *         iy2 = y2[i]
 *         iarea = areas[i]             # <<<<<<<<<<<<<<
 *         for _j in range(_i + 1, ndets):
 *             j = order[_j]
 */
    __pyx_t_25 = __pyx_v_i;
    __pyx_t_17 = -1;
    if (__pyx_t_25 < 0) {
      __pyx_t_25 += __pyx_pybuffernd_areas.diminfo[0].shape;
      if (unlikely(__pyx_t_25 < 0)) __pyx_t_17 = 0;
    } else if (unlikely(__pyx_t_25 >= __pyx_pybuffernd_areas.diminfo[0].shape)) __pyx_t_17 = 0;
    if (unlikely(__pyx_t_17 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_17);
      __PYX_ERR(0, 52, __pyx_L1_error)
    }
    __pyx_v_iarea = (*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_areas.rcbuffer->pybuffer.buf, __pyx_t_25, __pyx_pybuffernd_areas.diminfo[0].strides));

    /* "nms/cpu_nms.pyx":53
 *         iy2 = y2[i]
 *         iarea = areas[i]
 *         for _j in range(_i + 1, ndets):             # <<<<<<<<<<<<<<
 *             j = order[_j]
 *             if suppressed[j] == 1:
 */
    __pyx_t_17 = __pyx_v_ndets;
    for (__pyx_t_26 = (__pyx_v__i + 1); __pyx_t_26 < __pyx_t_17; __pyx_t_26+=1) {
      __pyx_v__j = __pyx_t_26;

      /* "nms/cpu_nms.pyx":54
 *         iarea = areas[i]
 *         for _j in range(_i + 1, ndets):
 *             j = order[_j]             # <<<<<<<<<<<<<<
 *             if suppressed[j] == 1:
 *                 continue
 */
      __pyx_t_27 = __pyx_v__j;
      __pyx_t_28 = -1;
      if (__pyx_t_27 < 0) {
        __pyx_t_27 += __pyx_pybuffernd_order.diminfo[0].shape;
        if (unlikely(__pyx_t_27 < 0)) __pyx_t_28 = 0;
      } else if (unlikely(__pyx_t_27 >= __pyx_pybuffernd_order.diminfo[0].shape)) __pyx_t_28 = 0;
      if (unlikely(__pyx_t_28 != -1)) {
        __Pyx_RaiseBufferIndexError(__pyx_t_28);
        __PYX_ERR(0, 54, __pyx_L1_error)
      }
      __pyx_v_j = (*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_int_t *, __pyx_pybuffernd_order.rcbuffer->pybuffer.buf, __pyx_t_27, __pyx_pybuffernd_order.diminfo[0].strides));

      /* "nms/cpu_nms.pyx":55
 *         for _j in range(_i + 1, ndets):
 *             j = order[_j]
 *             if suppressed[j] == 1:             # <<<<<<<<<<<<<<
 *                 continue
 *             xx1 = max(ix1, x1[j])
 */
      __pyx_t_29 = __pyx_v_j;
      __pyx_t_28 = -1;
      if (__pyx_t_29 < 0) {
        __pyx_t_29 += __pyx_pybuffernd_suppressed.diminfo[0].shape;
        if (unlikely(__pyx_t_29 < 0)) __pyx_t_28 = 0;
      } else if (unlikely(__pyx_t_29 >= __pyx_pybuffernd_suppressed.diminfo[0].shape)) __pyx_t_28 = 0;
      if (unlikely(__pyx_t_28 != -1)) {
        __Pyx_RaiseBufferIndexError(__pyx_t_28);
        __PYX_ERR(0, 55, __pyx_L1_error)
      }
      __pyx_t_19 = (((*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_int_t *, __pyx_pybuffernd_suppressed.rcbuffer->pybuffer.buf, __pyx_t_29, __pyx_pybuffernd_suppressed.diminfo[0].strides)) == 1) != 0);
      if (__pyx_t_19) {

        /* "nms/cpu_nms.pyx":56
 *             j = order[_j]
 *             if suppressed[j] == 1:
 *                 continue             # <<<<<<<<<<<<<<
 *             xx1 = max(ix1, x1[j])
 *             yy1 = max(iy1, y1[j])
 */
        goto __pyx_L6_continue;

        /* "nms/cpu_nms.pyx":55
 *         for _j in range(_i + 1, ndets):
 *             j = order[_j]
 *             if suppressed[j] == 1:             # <<<<<<<<<<<<<<
 *                 continue
 *             xx1 = max(ix1, x1[j])
 */
      }

      /* "nms/cpu_nms.pyx":57
 *             if suppressed[j] == 1:
 *                 continue
 *             xx1 = max(ix1, x1[j])             # <<<<<<<<<<<<<<
 *             yy1 = max(iy1, y1[j])
 *             xx2 = min(ix2, x2[j])
 */
      __pyx_t_30 = __pyx_v_j;
      __pyx_t_28 = -1;
      if (__pyx_t_30 < 0) {
        __pyx_t_30 += __pyx_pybuffernd_x1.diminfo[0].shape;
        if (unlikely(__pyx_t_30 < 0)) __pyx_t_28 = 0;
      } else if (unlikely(__pyx_t_30 >= __pyx_pybuffernd_x1.diminfo[0].shape)) __pyx_t_28 = 0;
      if (unlikely(__pyx_t_28 != -1)) {
        __Pyx_RaiseBufferIndexError(__pyx_t_28);
        __PYX_ERR(0, 57, __pyx_L1_error)
      }
      __pyx_v_xx1 = __pyx_f_3nms_7cpu_nms_max(__pyx_v_ix1, (*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_x1.rcbuffer->pybuffer.buf, __pyx_t_30, __pyx_pybuffernd_x1.diminfo[0].strides)));

      /* "nms/cpu_nms.pyx":58
 *                 continue
 *             xx1 = max(ix1, x1[j])
 *             yy1 = max(iy1, y1[j])             # <<<<<<<<<<<<<<
 *             xx2 = min(ix2, x2[j])
 *             yy2 = min(iy2, y2[j])
 */
      __pyx_t_31 = __pyx_v_j;
      __pyx_t_28 = -1;
      if (__pyx_t_31 < 0) {
        __pyx_t_31 += __pyx_pybuffernd_y1.diminfo[0].shape;
        if (unlikely(__pyx_t_31 < 0)) __pyx_t_28 = 0;
      } else if (unlikely(__pyx_t_31 >= __pyx_pybuffernd_y1.diminfo[0].shape)) __pyx_t_28 = 0;
      if (unlikely(__pyx_t_28 != -1)) {
        __Pyx_RaiseBufferIndexError(__pyx_t_28);
        __PYX_ERR(0, 58, __pyx_L1_error)
      }
      __pyx_v_yy1 = __pyx_f_3nms_7cpu_nms_max(__pyx_v_iy1, (*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_y1.rcbuffer->pybuffer.buf, __pyx_t_31, __pyx_pybuffernd_y1.diminfo[0].strides)));

      /* "nms/cpu_nms.pyx":59
 *             xx1 = max(ix1, x1[j])
 *             yy1 = max(iy1, y1[j])
 *             xx2 = min(ix2, x2[j])             # <<<<<<<<<<<<<<
 *             yy2 = min(iy2, y2[j])
 *             w = max(0.0, xx2 - xx1 + 1)
 */
      __pyx_t_32 = __pyx_v_j;
      __pyx_t_28 = -1;
      if (__pyx_t_32 < 0) {
        __pyx_t_32 += __pyx_pybuffernd_x2.diminfo[0].shape;
        if (unlikely(__pyx_t_32 < 0)) __pyx_t_28 = 0;
      } else if (unlikely(__pyx_t_32 >= __pyx_pybuffernd_x2.diminfo[0].shape)) __pyx_t_28 = 0;
      if (unlikely(__pyx_t_28 != -1)) {
        __Pyx_RaiseBufferIndexError(__pyx_t_28);
        __PYX_ERR(0, 59, __pyx_L1_error)
      }
      __pyx_v_xx2 = __pyx_f_3nms_7cpu_nms_min(__pyx_v_ix2, (*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_x2.rcbuffer->pybuffer.buf, __pyx_t_32, __pyx_pybuffernd_x2.diminfo[0].strides)));

      /* "nms/cpu_nms.pyx":60
 *             yy1 = max(iy1, y1[j])
 *             xx2 = min(ix2, x2[j])
 *             yy2 = min(iy2, y2[j])             # <<<<<<<<<<<<<<
 *             w = max(0.0, xx2 - xx1 + 1)
 *             h = max(0.0, yy2 - yy1 + 1)
 */
      __pyx_t_33 = __pyx_v_j;
      __pyx_t_28 = -1;
      if (__pyx_t_33 < 0) {
        __pyx_t_33 += __pyx_pybuffernd_y2.diminfo[0].shape;
        if (unlikely(__pyx_t_33 < 0)) __pyx_t_28 = 0;
      } else if (unlikely(__pyx_t_33 >= __pyx_pybuffernd_y2.diminfo[0].shape)) __pyx_t_28 = 0;
      if (unlikely(__pyx_t_28 != -1)) {
        __Pyx_RaiseBufferIndexError(__pyx_t_28);
        __PYX_ERR(0, 60, __pyx_L1_error)
      }
      __pyx_v_yy2 = __pyx_f_3nms_7cpu_nms_min(__pyx_v_iy2, (*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_y2.rcbuffer->pybuffer.buf, __pyx_t_33, __pyx_pybuffernd_y2.diminfo[0].strides)));

      /* "nms/cpu_nms.pyx":61
 *             xx2 = min(ix2, x2[j])
 *             yy2 = min(iy2, y2[j])
 *             w = max(0.0, xx2 - xx1 + 1)             # <<<<<<<<<<<<<<
 *             h = max(0.0, yy2 - yy1 + 1)
 *             inter = w * h
 */
      __pyx_v_w = __pyx_f_3nms_7cpu_nms_max(0.0, ((__pyx_v_xx2 - __pyx_v_xx1) + 1.0));

      /* "nms/cpu_nms.pyx":62
 *             yy2 = min(iy2, y2[j])
 *             w = max(0.0, xx2 - xx1 + 1)
 *             h = max(0.0, yy2 - yy1 + 1)             # <<<<<<<<<<<<<<
 *             inter = w * h
 *             ovr = inter / (iarea + areas[j] - inter)
 */
      __pyx_v_h = __pyx_f_3nms_7cpu_nms_max(0.0, ((__pyx_v_yy2 - __pyx_v_yy1) + 1.0));

      /* "nms/cpu_nms.pyx":63
 *             w = max(0.0, xx2 - xx1 + 1)
 *             h = max(0.0, yy2 - yy1 + 1)
 *             inter = w * h             # <<<<<<<<<<<<<<
 *             ovr = inter / (iarea + areas[j] - inter)
 *             if ovr >= thresh:
 */
      __pyx_v_inter = (__pyx_v_w * __pyx_v_h);

      /* "nms/cpu_nms.pyx":64
 *             h = max(0.0, yy2 - yy1 + 1)
 *             inter = w * h
 *             ovr = inter / (iarea + areas[j] - inter)             # <<<<<<<<<<<<<<
 *             if ovr >= thresh:
 *                 suppressed[j] = 1
 */
      __pyx_t_34 = __pyx_v_j;
      __pyx_t_28 = -1;
      if (__pyx_t_34 < 0) {
        __pyx_t_34 += __pyx_pybuffernd_areas.diminfo[0].shape;
        if (unlikely(__pyx_t_34 < 0)) __pyx_t_28 = 0;
      } else if (unlikely(__pyx_t_34 >= __pyx_pybuffernd_areas.diminfo[0].shape)) __pyx_t_28 = 0;
      if (unlikely(__pyx_t_28 != -1)) {
        __Pyx_RaiseBufferIndexError(__pyx_t_28);
        __PYX_ERR(0, 64, __pyx_L1_error)
      }
      __pyx_t_35 = ((__pyx_v_iarea + (*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_areas.rcbuffer->pybuffer.buf, __pyx_t_34, __pyx_pybuffernd_areas.diminfo[0].strides))) - __pyx_v_inter);
      if (unlikely(__pyx_t_35 == 0)) {
        PyErr_SetString(PyExc_ZeroDivisionError, "float division");
        __PYX_ERR(0, 64, __pyx_L1_error)
      }
      __pyx_v_ovr = (__pyx_v_inter / __pyx_t_35);

      /* "nms/cpu_nms.pyx":65
 *             inter = w * h
 *             ovr = inter / (iarea + areas[j] - inter)
 *             if ovr >= thresh:             # <<<<<<<<<<<<<<
 *                 suppressed[j] = 1
 * 
 */
      __pyx_t_12 = PyFloat_FromDouble(__pyx_v_ovr); if (unlikely(!__pyx_t_12)) __PYX_ERR(0, 65, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_12);
      __pyx_t_8 = PyObject_RichCompare(__pyx_t_12, __pyx_v_thresh, Py_GE); __Pyx_XGOTREF(__pyx_t_8); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 65, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_12); __pyx_t_12 = 0;
      __pyx_t_19 = __Pyx_PyObject_IsTrue(__pyx_t_8); if (unlikely(__pyx_t_19 < 0)) __PYX_ERR(0, 65, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
      if (__pyx_t_19) {

        /* "nms/cpu_nms.pyx":66
 *             ovr = inter / (iarea + areas[j] - inter)
 *             if ovr >= thresh:
 *                 suppressed[j] = 1             # <<<<<<<<<<<<<<
 * 
 *     return keep
 */
        __pyx_t_36 = __pyx_v_j;
        __pyx_t_28 = -1;
        if (__pyx_t_36 < 0) {
          __pyx_t_36 += __pyx_pybuffernd_suppressed.diminfo[0].shape;
          if (unlikely(__pyx_t_36 < 0)) __pyx_t_28 = 0;
        } else if (unlikely(__pyx_t_36 >= __pyx_pybuffernd_suppressed.diminfo[0].shape)) __pyx_t_28 = 0;
        if (unlikely(__pyx_t_28 != -1)) {
          __Pyx_RaiseBufferIndexError(__pyx_t_28);
          __PYX_ERR(0, 66, __pyx_L1_error)
        }
        *__Pyx_BufPtrStrided1d(__pyx_t_5numpy_int_t *, __pyx_pybuffernd_suppressed.rcbuffer->pybuffer.buf, __pyx_t_36, __pyx_pybuffernd_suppressed.diminfo[0].strides) = 1;

        /* "nms/cpu_nms.pyx":65
 *             inter = w * h
 *             ovr = inter / (iarea + areas[j] - inter)
 *             if ovr >= thresh:             # <<<<<<<<<<<<<<
 *                 suppressed[j] = 1
 * 
 */
      }
      __pyx_L6_continue:;
    }
    __pyx_L3_continue:;
  }

  /* "nms/cpu_nms.pyx":68
 *                 suppressed[j] = 1
 * 
 *     return keep             # <<<<<<<<<<<<<<
 * 
 * def cpu_soft_nms(np.ndarray[float, ndim=2] boxes, float sigma=0.5, float Nt=0.3, float threshold=0.001, unsigned int method=0):
 */
  __Pyx_XDECREF(__pyx_r);
  __Pyx_INCREF(__pyx_v_keep);
  __pyx_r = __pyx_v_keep;
  goto __pyx_L0;

  /* "nms/cpu_nms.pyx":17
 *     return a if a <= b else b
 * 
 * def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
 *     cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_7);
  __Pyx_XDECREF(__pyx_t_8);
  __Pyx_XDECREF(__pyx_t_11);
  __Pyx_XDECREF(__pyx_t_12);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_areas.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_dets.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_order.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_scores.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_suppressed.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_x1.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_x2.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_y1.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_y2.rcbuffer->pybuffer);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("nms.cpu_nms.cpu_nms", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_areas.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_dets.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_order.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_scores.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_suppressed.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_x1.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_x2.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_y1.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_y2.rcbuffer->pybuffer);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_x1);
  __Pyx_XDECREF((PyObject *)__pyx_v_y1);
  __Pyx_XDECREF((PyObject *)__pyx_v_x2);
  __Pyx_XDECREF((PyObject *)__pyx_v_y2);
  __Pyx_XDECREF((PyObject *)__pyx_v_scores);
  __Pyx_XDECREF((PyObject *)__pyx_v_areas);
  __Pyx_XDECREF((PyObject *)__pyx_v_order);
  __Pyx_XDECREF((PyObject *)__pyx_v_suppressed);
  __Pyx_XDECREF(__pyx_v_keep);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "nms/cpu_nms.pyx":70
 *     return keep
 * 
 * def cpu_soft_nms(np.ndarray[float, ndim=2] boxes, float sigma=0.5, float Nt=0.3, float threshold=0.001, unsigned int method=0):             # <<<<<<<<<<<<<<
 *     cdef unsigned int N = boxes.shape[0]
 *     cdef float iw, ih, box_area
 */

/* Python wrapper */
static PyObject *__pyx_pw_3nms_7cpu_nms_3cpu_soft_nms(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_mdef_3nms_7cpu_nms_3cpu_soft_nms = {"cpu_soft_nms", (PyCFunction)__pyx_pw_3nms_7cpu_nms_3cpu_soft_nms, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_pw_3nms_7cpu_nms_3cpu_soft_nms(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyArrayObject *__pyx_v_boxes = 0;
  float __pyx_v_sigma;
  float __pyx_v_Nt;
  float __pyx_v_threshold;
  unsigned int __pyx_v_method;
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("cpu_soft_nms (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&__pyx_n_s_boxes,&__pyx_n_s_sigma,&__pyx_n_s_Nt,&__pyx_n_s_threshold,&__pyx_n_s_method,0};
    PyObject* values[5] = {0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = PyTuple_GET_SIZE(__pyx_args);
      switch (pos_args) {
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s_boxes)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s_sigma);
          if (value) { values[1] = value; kw_args--; }
        }
        case  2:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s_Nt);
          if (value) { values[2] = value; kw_args--; }
        }
        case  3:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s_threshold);
          if (value) { values[3] = value; kw_args--; }
        }
        case  4:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s_method);
          if (value) { values[4] = value; kw_args--; }
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "cpu_soft_nms") < 0)) __PYX_ERR(0, 70, __pyx_L3_error)
      }
    } else {
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        break;
        default: goto __pyx_L5_argtuple_error;
      }
    }
    __pyx_v_boxes = ((PyArrayObject *)values[0]);
    if (values[1]) {
      __pyx_v_sigma = __pyx_PyFloat_AsFloat(values[1]); if (unlikely((__pyx_v_sigma == (float)-1) && PyErr_Occurred())) __PYX_ERR(0, 70, __pyx_L3_error)
    } else {
      __pyx_v_sigma = ((float)0.5);
    }
    if (values[2]) {
      __pyx_v_Nt = __pyx_PyFloat_AsFloat(values[2]); if (unlikely((__pyx_v_Nt == (float)-1) && PyErr_Occurred())) __PYX_ERR(0, 70, __pyx_L3_error)
    } else {
      __pyx_v_Nt = ((float)0.3);
    }
    if (values[3]) {
      __pyx_v_threshold = __pyx_PyFloat_AsFloat(values[3]); if (unlikely((__pyx_v_threshold == (float)-1) && PyErr_Occurred())) __PYX_ERR(0, 70, __pyx_L3_error)
    } else {
      __pyx_v_threshold = ((float)0.001);
    }
    if (values[4]) {
      __pyx_v_method = __Pyx_PyInt_As_unsigned_int(values[4]); if (unlikely((__pyx_v_method == (unsigned int)-1) && PyErr_Occurred())) __PYX_ERR(0, 70, __pyx_L3_error)
    } else {
      __pyx_v_method = ((unsigned int)0);
    }
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("cpu_soft_nms", 0, 1, 5, PyTuple_GET_SIZE(__pyx_args)); __PYX_ERR(0, 70, __pyx_L3_error)
  __pyx_L3_error:;
  __Pyx_AddTraceback("nms.cpu_nms.cpu_soft_nms", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_boxes), __pyx_ptype_5numpy_ndarray, 1, "boxes", 0))) __PYX_ERR(0, 70, __pyx_L1_error)
  __pyx_r = __pyx_pf_3nms_7cpu_nms_2cpu_soft_nms(__pyx_self, __pyx_v_boxes, __pyx_v_sigma, __pyx_v_Nt, __pyx_v_threshold, __pyx_v_method);

  /* function exit code */
  goto __pyx_L0;
  __pyx_L1_error:;
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_3nms_7cpu_nms_2cpu_soft_nms(CYTHON_UNUSED PyObject *__pyx_self, PyArrayObject *__pyx_v_boxes, float __pyx_v_sigma, float __pyx_v_Nt, float __pyx_v_threshold, unsigned int __pyx_v_method) {
  unsigned int __pyx_v_N;
  float __pyx_v_iw;
  float __pyx_v_ih;
  float __pyx_v_ua;
  int __pyx_v_pos;
  float __pyx_v_maxscore;
  int __pyx_v_maxpos;
  float __pyx_v_x1;
  float __pyx_v_x2;
  float __pyx_v_y1;
  float __pyx_v_y2;
  float __pyx_v_tx1;
  float __pyx_v_tx2;
  float __pyx_v_ty1;
  float __pyx_v_ty2;
  float __pyx_v_ts;
  float __pyx_v_area;
  float __pyx_v_weight;
  float __pyx_v_ov;
  PyObject *__pyx_v_i = NULL;
  CYTHON_UNUSED PyObject *__pyx_v_s = NULL;
  PyObject *__pyx_v_keep = NULL;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_boxes;
  __Pyx_Buffer __pyx_pybuffer_boxes;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  Py_ssize_t __pyx_t_3;
  PyObject *(*__pyx_t_4)(PyObject *);
  PyObject *__pyx_t_5 = NULL;
  float __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  Py_ssize_t __pyx_t_9;
  Py_ssize_t __pyx_t_10;
  Py_ssize_t __pyx_t_11;
  Py_ssize_t __pyx_t_12;
  Py_ssize_t __pyx_t_13;
  Py_ssize_t __pyx_t_14;
  Py_ssize_t __pyx_t_15;
  Py_ssize_t __pyx_t_16;
  Py_ssize_t __pyx_t_17;
  Py_ssize_t __pyx_t_18;
  Py_ssize_t __pyx_t_19;
  Py_ssize_t __pyx_t_20;
  Py_ssize_t __pyx_t_21;
  Py_ssize_t __pyx_t_22;
  Py_ssize_t __pyx_t_23;
  Py_ssize_t __pyx_t_24;
  Py_ssize_t __pyx_t_25;
  Py_ssize_t __pyx_t_26;
  Py_ssize_t __pyx_t_27;
  Py_ssize_t __pyx_t_28;
  Py_ssize_t __pyx_t_29;
  Py_ssize_t __pyx_t_30;
  Py_ssize_t __pyx_t_31;
  Py_ssize_t __pyx_t_32;
  Py_ssize_t __pyx_t_33;
  Py_ssize_t __pyx_t_34;
  Py_ssize_t __pyx_t_35;
  Py_ssize_t __pyx_t_36;
  Py_ssize_t __pyx_t_37;
  Py_ssize_t __pyx_t_38;
  Py_ssize_t __pyx_t_39;
  Py_ssize_t __pyx_t_40;
  Py_ssize_t __pyx_t_41;
  Py_ssize_t __pyx_t_42;
  PyObject *__pyx_t_43 = NULL;
  PyObject *__pyx_t_44 = NULL;
  PyObject *__pyx_t_45 = NULL;
  Py_ssize_t __pyx_t_46;
  Py_ssize_t __pyx_t_47;
  Py_ssize_t __pyx_t_48;
  Py_ssize_t __pyx_t_49;
  Py_ssize_t __pyx_t_50;
  Py_ssize_t __pyx_t_51;
  Py_ssize_t __pyx_t_52;
  Py_ssize_t __pyx_t_53;
  Py_ssize_t __pyx_t_54;
  Py_ssize_t __pyx_t_55;
  Py_ssize_t __pyx_t_56;
  Py_ssize_t __pyx_t_57;
  Py_ssize_t __pyx_t_58;
  Py_ssize_t __pyx_t_59;
  Py_ssize_t __pyx_t_60;
  Py_ssize_t __pyx_t_61;
  Py_ssize_t __pyx_t_62;
  Py_ssize_t __pyx_t_63;
  Py_ssize_t __pyx_t_64;
  Py_ssize_t __pyx_t_65;
  Py_ssize_t __pyx_t_66;
  Py_ssize_t __pyx_t_67;
  Py_ssize_t __pyx_t_68;
  Py_ssize_t __pyx_t_69;
  Py_ssize_t __pyx_t_70;
  Py_ssize_t __pyx_t_71;
  __Pyx_RefNannySetupContext("cpu_soft_nms", 0);
  __pyx_pybuffer_boxes.pybuffer.buf = NULL;
  __pyx_pybuffer_boxes.refcount = 0;
  __pyx_pybuffernd_boxes.data = NULL;
  __pyx_pybuffernd_boxes.rcbuffer = &__pyx_pybuffer_boxes;
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_boxes.rcbuffer->pybuffer, (PyObject*)__pyx_v_boxes, &__Pyx_TypeInfo_float, PyBUF_FORMAT| PyBUF_STRIDES| PyBUF_WRITABLE, 2, 0, __pyx_stack) == -1)) __PYX_ERR(0, 70, __pyx_L1_error)
  }
  __pyx_pybuffernd_boxes.diminfo[0].strides = __pyx_pybuffernd_boxes.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_boxes.diminfo[0].shape = __pyx_pybuffernd_boxes.rcbuffer->pybuffer.shape[0]; __pyx_pybuffernd_boxes.diminfo[1].strides = __pyx_pybuffernd_boxes.rcbuffer->pybuffer.strides[1]; __pyx_pybuffernd_boxes.diminfo[1].shape = __pyx_pybuffernd_boxes.rcbuffer->pybuffer.shape[1];

  /* "nms/cpu_nms.pyx":71
 * 
 * def cpu_soft_nms(np.ndarray[float, ndim=2] boxes, float sigma=0.5, float Nt=0.3, float threshold=0.001, unsigned int method=0):
 *     cdef unsigned int N = boxes.shape[0]             # <<<<<<<<<<<<<<
 *     cdef float iw, ih, box_area
 *     cdef float ua
 */
  __pyx_v_N = (__pyx_v_boxes->dimensions[0]);

  /* "nms/cpu_nms.pyx":74
 *     cdef float iw, ih, box_area
 *     cdef float ua
 *     cdef int pos = 0             # <<<<<<<<<<<<<<
 *     cdef float maxscore = 0
 *     cdef int maxpos = 0
 */
  __pyx_v_pos = 0;

  /* "nms/cpu_nms.pyx":75
 *     cdef float ua
 *     cdef int pos = 0
 *     cdef float maxscore = 0             # <<<<<<<<<<<<<<
 *     cdef int maxpos = 0
 *     cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov
 */
  __pyx_v_maxscore = 0.0;

  /* "nms/cpu_nms.pyx":76
 *     cdef int pos = 0
 *     cdef float maxscore = 0
 *     cdef int maxpos = 0             # <<<<<<<<<<<<<<
 *     cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov
 * 
 */
  __pyx_v_maxpos = 0;

  /* "nms/cpu_nms.pyx":79
 *     cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov
 * 
 *     for i in range(N):             # <<<<<<<<<<<<<<
 *         maxscore = boxes[i, 4]
 *         maxpos = i
 */
  __pyx_t_1 = __Pyx_PyInt_From_unsigned_int(__pyx_v_N); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 79, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyTuple_New(1); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 79, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_GIVEREF(__pyx_t_1);
  PyTuple_SET_ITEM(__pyx_t_2, 0, __pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = __Pyx_PyObject_Call(__pyx_builtin_range, __pyx_t_2, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 79, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  if (likely(PyList_CheckExact(__pyx_t_1)) || PyTuple_CheckExact(__pyx_t_1)) {
    __pyx_t_2 = __pyx_t_1; __Pyx_INCREF(__pyx_t_2); __pyx_t_3 = 0;
    __pyx_t_4 = NULL;
  } else {
    __pyx_t_3 = -1; __pyx_t_2 = PyObject_GetIter(__pyx_t_1); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 79, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_2);
    __pyx_t_4 = Py_TYPE(__pyx_t_2)->tp_iternext; if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 79, __pyx_L1_error)
  }
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  for (;;) {
    if (likely(!__pyx_t_4)) {
      if (likely(PyList_CheckExact(__pyx_t_2))) {
        if (__pyx_t_3 >= PyList_GET_SIZE(__pyx_t_2)) break;
        #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
        __pyx_t_1 = PyList_GET_ITEM(__pyx_t_2, __pyx_t_3); __Pyx_INCREF(__pyx_t_1); __pyx_t_3++; if (unlikely(0 < 0)) __PYX_ERR(0, 79, __pyx_L1_error)
        #else
        __pyx_t_1 = PySequence_ITEM(__pyx_t_2, __pyx_t_3); __pyx_t_3++; if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 79, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_1);
        #endif
      } else {
        if (__pyx_t_3 >= PyTuple_GET_SIZE(__pyx_t_2)) break;
        #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
        __pyx_t_1 = PyTuple_GET_ITEM(__pyx_t_2, __pyx_t_3); __Pyx_INCREF(__pyx_t_1); __pyx_t_3++; if (unlikely(0 < 0)) __PYX_ERR(0, 79, __pyx_L1_error)
        #else
        __pyx_t_1 = PySequence_ITEM(__pyx_t_2, __pyx_t_3); __pyx_t_3++; if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 79, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_1);
        #endif
      }
    } else {
      __pyx_t_1 = __pyx_t_4(__pyx_t_2);
      if (unlikely(!__pyx_t_1)) {
        PyObject* exc_type = PyErr_Occurred();
        if (exc_type) {
          if (likely(exc_type == PyExc_StopIteration || PyErr_GivenExceptionMatches(exc_type, PyExc_StopIteration))) PyErr_Clear();
          else __PYX_ERR(0, 79, __pyx_L1_error)
        }
        break;
      }
      __Pyx_GOTREF(__pyx_t_1);
    }
    __Pyx_XDECREF_SET(__pyx_v_i, __pyx_t_1);
    __pyx_t_1 = 0;

    /* "nms/cpu_nms.pyx":80
 * 
 *     for i in range(N):
 *         maxscore = boxes[i, 4]             # <<<<<<<<<<<<<<
 *         maxpos = i
 * 
 */
    __pyx_t_1 = PyTuple_New(2); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 80, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_1, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_4);
    __Pyx_GIVEREF(__pyx_int_4);
    PyTuple_SET_ITEM(__pyx_t_1, 1, __pyx_int_4);
    __pyx_t_5 = PyObject_GetItem(((PyObject *)__pyx_v_boxes), __pyx_t_1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 80, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_t_6 = __pyx_PyFloat_AsFloat(__pyx_t_5); if (unlikely((__pyx_t_6 == (float)-1) && PyErr_Occurred())) __PYX_ERR(0, 80, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_v_maxscore = __pyx_t_6;

    /* "nms/cpu_nms.pyx":81
 *     for i in range(N):
 *         maxscore = boxes[i, 4]
 *         maxpos = i             # <<<<<<<<<<<<<<
 * 
 *         tx1 = boxes[i,0]
 */
    __pyx_t_7 = __Pyx_PyInt_As_int(__pyx_v_i); if (unlikely((__pyx_t_7 == (int)-1) && PyErr_Occurred())) __PYX_ERR(0, 81, __pyx_L1_error)
    __pyx_v_maxpos = __pyx_t_7;

    /* "nms/cpu_nms.pyx":83
 *         maxpos = i
 * 
 *         tx1 = boxes[i,0]             # <<<<<<<<<<<<<<
 *         ty1 = boxes[i,1]
 *         tx2 = boxes[i,2]
 */
    __pyx_t_5 = PyTuple_New(2); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 83, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_0);
    __Pyx_GIVEREF(__pyx_int_0);
    PyTuple_SET_ITEM(__pyx_t_5, 1, __pyx_int_0);
    __pyx_t_1 = PyObject_GetItem(((PyObject *)__pyx_v_boxes), __pyx_t_5); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 83, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_t_6 = __pyx_PyFloat_AsFloat(__pyx_t_1); if (unlikely((__pyx_t_6 == (float)-1) && PyErr_Occurred())) __PYX_ERR(0, 83, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_v_tx1 = __pyx_t_6;

    /* "nms/cpu_nms.pyx":84
 * 
 *         tx1 = boxes[i,0]
 *         ty1 = boxes[i,1]             # <<<<<<<<<<<<<<
 *         tx2 = boxes[i,2]
 *         ty2 = boxes[i,3]
 */
    __pyx_t_1 = PyTuple_New(2); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 84, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_1, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_1);
    __Pyx_GIVEREF(__pyx_int_1);
    PyTuple_SET_ITEM(__pyx_t_1, 1, __pyx_int_1);
    __pyx_t_5 = PyObject_GetItem(((PyObject *)__pyx_v_boxes), __pyx_t_1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 84, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_t_6 = __pyx_PyFloat_AsFloat(__pyx_t_5); if (unlikely((__pyx_t_6 == (float)-1) && PyErr_Occurred())) __PYX_ERR(0, 84, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_v_ty1 = __pyx_t_6;

    /* "nms/cpu_nms.pyx":85
 *         tx1 = boxes[i,0]
 *         ty1 = boxes[i,1]
 *         tx2 = boxes[i,2]             # <<<<<<<<<<<<<<
 *         ty2 = boxes[i,3]
 *         ts = boxes[i,4]
 */
    __pyx_t_5 = PyTuple_New(2); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 85, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_2);
    __Pyx_GIVEREF(__pyx_int_2);
    PyTuple_SET_ITEM(__pyx_t_5, 1, __pyx_int_2);
    __pyx_t_1 = PyObject_GetItem(((PyObject *)__pyx_v_boxes), __pyx_t_5); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 85, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_t_6 = __pyx_PyFloat_AsFloat(__pyx_t_1); if (unlikely((__pyx_t_6 == (float)-1) && PyErr_Occurred())) __PYX_ERR(0, 85, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_v_tx2 = __pyx_t_6;

    /* "nms/cpu_nms.pyx":86
 *         ty1 = boxes[i,1]
 *         tx2 = boxes[i,2]
 *         ty2 = boxes[i,3]             # <<<<<<<<<<<<<<
 *         ts = boxes[i,4]
 * 
 */
    __pyx_t_1 = PyTuple_New(2); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 86, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_1, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_3);
    __Pyx_GIVEREF(__pyx_int_3);
    PyTuple_SET_ITEM(__pyx_t_1, 1, __pyx_int_3);
    __pyx_t_5 = PyObject_GetItem(((PyObject *)__pyx_v_boxes), __pyx_t_1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 86, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_t_6 = __pyx_PyFloat_AsFloat(__pyx_t_5); if (unlikely((__pyx_t_6 == (float)-1) && PyErr_Occurred())) __PYX_ERR(0, 86, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_v_ty2 = __pyx_t_6;

    /* "nms/cpu_nms.pyx":87
 *         tx2 = boxes[i,2]
 *         ty2 = boxes[i,3]
 *         ts = boxes[i,4]             # <<<<<<<<<<<<<<
 * 
 *         pos = i + 1
 */
    __pyx_t_5 = PyTuple_New(2); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 87, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_4);
    __Pyx_GIVEREF(__pyx_int_4);
    PyTuple_SET_ITEM(__pyx_t_5, 1, __pyx_int_4);
    __pyx_t_1 = PyObject_GetItem(((PyObject *)__pyx_v_boxes), __pyx_t_5); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 87, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_t_6 = __pyx_PyFloat_AsFloat(__pyx_t_1); if (unlikely((__pyx_t_6 == (float)-1) && PyErr_Occurred())) __PYX_ERR(0, 87, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_v_ts = __pyx_t_6;

    /* "nms/cpu_nms.pyx":89
 *         ts = boxes[i,4]
 * 
 *         pos = i + 1             # <<<<<<<<<<<<<<
 * 	# get max box
 *         while pos < N:
 */
    __pyx_t_1 = __Pyx_PyInt_AddObjC(__pyx_v_i, __pyx_int_1, 1, 0); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 89, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __pyx_t_7 = __Pyx_PyInt_As_int(__pyx_t_1); if (unlikely((__pyx_t_7 == (int)-1) && PyErr_Occurred())) __PYX_ERR(0, 89, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_v_pos = __pyx_t_7;

    /* "nms/cpu_nms.pyx":91
 *         pos = i + 1
 * 	# get max box
 *         while pos < N:             # <<<<<<<<<<<<<<
 *             if maxscore < boxes[pos, 4]:
 *                 maxscore = boxes[pos, 4]
 */
    while (1) {
      __pyx_t_8 = ((__pyx_v_pos < __pyx_v_N) != 0);
      if (!__pyx_t_8) break;

      /* "nms/cpu_nms.pyx":92
 * 	# get max box
 *         while pos < N:
 *             if maxscore < boxes[pos, 4]:             # <<<<<<<<<<<<<<
 *                 maxscore = boxes[pos, 4]
 *                 maxpos = pos
 */
      __pyx_t_9 = __pyx_v_pos;
      __pyx_t_10 = 4;
      __pyx_t_7 = -1;
      if (__pyx_t_9 < 0) {
        __pyx_t_9 += __pyx_pybuffernd_boxes.diminfo[0].shape;
        if (unlikely(__pyx_t_9 < 0)) __pyx_t_7 = 0;
      } else if (unlikely(__pyx_t_9 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
      if (__pyx_t_10 < 0) {
        __pyx_t_10 += __pyx_pybuffernd_boxes.diminfo[1].shape;
        if (unlikely(__pyx_t_10 < 0)) __pyx_t_7 = 1;
      } else if (unlikely(__pyx_t_10 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
      if (unlikely(__pyx_t_7 != -1)) {
        __Pyx_RaiseBufferIndexError(__pyx_t_7);
        __PYX_ERR(0, 92, __pyx_L1_error)
      }
      __pyx_t_8 = ((__pyx_v_maxscore < (*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_9, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_10, __pyx_pybuffernd_boxes.diminfo[1].strides))) != 0);
      if (__pyx_t_8) {

        /* "nms/cpu_nms.pyx":93
 *         while pos < N:
 *             if maxscore < boxes[pos, 4]:
 *                 maxscore = boxes[pos, 4]             # <<<<<<<<<<<<<<
 *                 maxpos = pos
 *             pos = pos + 1
 */
        __pyx_t_11 = __pyx_v_pos;
        __pyx_t_12 = 4;
        __pyx_t_7 = -1;
        if (__pyx_t_11 < 0) {
          __pyx_t_11 += __pyx_pybuffernd_boxes.diminfo[0].shape;
          if (unlikely(__pyx_t_11 < 0)) __pyx_t_7 = 0;
        } else if (unlikely(__pyx_t_11 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
        if (__pyx_t_12 < 0) {
          __pyx_t_12 += __pyx_pybuffernd_boxes.diminfo[1].shape;
          if (unlikely(__pyx_t_12 < 0)) __pyx_t_7 = 1;
        } else if (unlikely(__pyx_t_12 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
        if (unlikely(__pyx_t_7 != -1)) {
          __Pyx_RaiseBufferIndexError(__pyx_t_7);
          __PYX_ERR(0, 93, __pyx_L1_error)
        }
        __pyx_v_maxscore = (*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_11, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_12, __pyx_pybuffernd_boxes.diminfo[1].strides));

        /* "nms/cpu_nms.pyx":94
 *             if maxscore < boxes[pos, 4]:
 *                 maxscore = boxes[pos, 4]
 *                 maxpos = pos             # <<<<<<<<<<<<<<
 *             pos = pos + 1
 * 
 */
        __pyx_v_maxpos = __pyx_v_pos;

        /* "nms/cpu_nms.pyx":92
 * 	# get max box
 *         while pos < N:
 *             if maxscore < boxes[pos, 4]:             # <<<<<<<<<<<<<<
 *                 maxscore = boxes[pos, 4]
 *                 maxpos = pos
 */
      }

      /* "nms/cpu_nms.pyx":95
 *                 maxscore = boxes[pos, 4]
 *                 maxpos = pos
 *             pos = pos + 1             # <<<<<<<<<<<<<<
 * 
 * 	# add max box as a detection
 */
      __pyx_v_pos = (__pyx_v_pos + 1);
    }

    /* "nms/cpu_nms.pyx":98
 * 
 * 	# add max box as a detection
 *         boxes[i,0] = boxes[maxpos,0]             # <<<<<<<<<<<<<<
 *         boxes[i,1] = boxes[maxpos,1]
 *         boxes[i,2] = boxes[maxpos,2]
 */
    __pyx_t_13 = __pyx_v_maxpos;
    __pyx_t_14 = 0;
    __pyx_t_7 = -1;
    if (__pyx_t_13 < 0) {
      __pyx_t_13 += __pyx_pybuffernd_boxes.diminfo[0].shape;
      if (unlikely(__pyx_t_13 < 0)) __pyx_t_7 = 0;
    } else if (unlikely(__pyx_t_13 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
    if (__pyx_t_14 < 0) {
      __pyx_t_14 += __pyx_pybuffernd_boxes.diminfo[1].shape;
      if (unlikely(__pyx_t_14 < 0)) __pyx_t_7 = 1;
    } else if (unlikely(__pyx_t_14 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
    if (unlikely(__pyx_t_7 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_7);
      __PYX_ERR(0, 98, __pyx_L1_error)
    }
    __pyx_t_1 = PyFloat_FromDouble((*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_13, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_14, __pyx_pybuffernd_boxes.diminfo[1].strides))); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 98, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __pyx_t_5 = PyTuple_New(2); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 98, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_0);
    __Pyx_GIVEREF(__pyx_int_0);
    PyTuple_SET_ITEM(__pyx_t_5, 1, __pyx_int_0);
    if (unlikely(PyObject_SetItem(((PyObject *)__pyx_v_boxes), __pyx_t_5, __pyx_t_1) < 0)) __PYX_ERR(0, 98, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

    /* "nms/cpu_nms.pyx":99
 * 	# add max box as a detection
 *         boxes[i,0] = boxes[maxpos,0]
 *         boxes[i,1] = boxes[maxpos,1]             # <<<<<<<<<<<<<<
 *         boxes[i,2] = boxes[maxpos,2]
 *         boxes[i,3] = boxes[maxpos,3]
 */
    __pyx_t_15 = __pyx_v_maxpos;
    __pyx_t_16 = 1;
    __pyx_t_7 = -1;
    if (__pyx_t_15 < 0) {
      __pyx_t_15 += __pyx_pybuffernd_boxes.diminfo[0].shape;
      if (unlikely(__pyx_t_15 < 0)) __pyx_t_7 = 0;
    } else if (unlikely(__pyx_t_15 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
    if (__pyx_t_16 < 0) {
      __pyx_t_16 += __pyx_pybuffernd_boxes.diminfo[1].shape;
      if (unlikely(__pyx_t_16 < 0)) __pyx_t_7 = 1;
    } else if (unlikely(__pyx_t_16 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
    if (unlikely(__pyx_t_7 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_7);
      __PYX_ERR(0, 99, __pyx_L1_error)
    }
    __pyx_t_1 = PyFloat_FromDouble((*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_15, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_16, __pyx_pybuffernd_boxes.diminfo[1].strides))); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 99, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __pyx_t_5 = PyTuple_New(2); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 99, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_1);
    __Pyx_GIVEREF(__pyx_int_1);
    PyTuple_SET_ITEM(__pyx_t_5, 1, __pyx_int_1);
    if (unlikely(PyObject_SetItem(((PyObject *)__pyx_v_boxes), __pyx_t_5, __pyx_t_1) < 0)) __PYX_ERR(0, 99, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

    /* "nms/cpu_nms.pyx":100
 *         boxes[i,0] = boxes[maxpos,0]
 *         boxes[i,1] = boxes[maxpos,1]
 *         boxes[i,2] = boxes[maxpos,2]             # <<<<<<<<<<<<<<
 *         boxes[i,3] = boxes[maxpos,3]
 *         boxes[i,4] = boxes[maxpos,4]
 */
    __pyx_t_17 = __pyx_v_maxpos;
    __pyx_t_18 = 2;
    __pyx_t_7 = -1;
    if (__pyx_t_17 < 0) {
      __pyx_t_17 += __pyx_pybuffernd_boxes.diminfo[0].shape;
      if (unlikely(__pyx_t_17 < 0)) __pyx_t_7 = 0;
    } else if (unlikely(__pyx_t_17 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
    if (__pyx_t_18 < 0) {
      __pyx_t_18 += __pyx_pybuffernd_boxes.diminfo[1].shape;
      if (unlikely(__pyx_t_18 < 0)) __pyx_t_7 = 1;
    } else if (unlikely(__pyx_t_18 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
    if (unlikely(__pyx_t_7 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_7);
      __PYX_ERR(0, 100, __pyx_L1_error)
    }
    __pyx_t_1 = PyFloat_FromDouble((*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_17, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_18, __pyx_pybuffernd_boxes.diminfo[1].strides))); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 100, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __pyx_t_5 = PyTuple_New(2); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 100, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_2);
    __Pyx_GIVEREF(__pyx_int_2);
    PyTuple_SET_ITEM(__pyx_t_5, 1, __pyx_int_2);
    if (unlikely(PyObject_SetItem(((PyObject *)__pyx_v_boxes), __pyx_t_5, __pyx_t_1) < 0)) __PYX_ERR(0, 100, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

    /* "nms/cpu_nms.pyx":101
 *         boxes[i,1] = boxes[maxpos,1]
 *         boxes[i,2] = boxes[maxpos,2]
 *         boxes[i,3] = boxes[maxpos,3]             # <<<<<<<<<<<<<<
 *         boxes[i,4] = boxes[maxpos,4]
 * 
 */
    __pyx_t_19 = __pyx_v_maxpos;
    __pyx_t_20 = 3;
    __pyx_t_7 = -1;
    if (__pyx_t_19 < 0) {
      __pyx_t_19 += __pyx_pybuffernd_boxes.diminfo[0].shape;
      if (unlikely(__pyx_t_19 < 0)) __pyx_t_7 = 0;
    } else if (unlikely(__pyx_t_19 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
    if (__pyx_t_20 < 0) {
      __pyx_t_20 += __pyx_pybuffernd_boxes.diminfo[1].shape;
      if (unlikely(__pyx_t_20 < 0)) __pyx_t_7 = 1;
    } else if (unlikely(__pyx_t_20 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
    if (unlikely(__pyx_t_7 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_7);
      __PYX_ERR(0, 101, __pyx_L1_error)
    }
    __pyx_t_1 = PyFloat_FromDouble((*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_19, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_20, __pyx_pybuffernd_boxes.diminfo[1].strides))); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 101, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __pyx_t_5 = PyTuple_New(2); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 101, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_3);
    __Pyx_GIVEREF(__pyx_int_3);
    PyTuple_SET_ITEM(__pyx_t_5, 1, __pyx_int_3);
    if (unlikely(PyObject_SetItem(((PyObject *)__pyx_v_boxes), __pyx_t_5, __pyx_t_1) < 0)) __PYX_ERR(0, 101, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

    /* "nms/cpu_nms.pyx":102
 *         boxes[i,2] = boxes[maxpos,2]
 *         boxes[i,3] = boxes[maxpos,3]
 *         boxes[i,4] = boxes[maxpos,4]             # <<<<<<<<<<<<<<
 * 
 * 	# swap ith box with position of max box
 */
    __pyx_t_21 = __pyx_v_maxpos;
    __pyx_t_22 = 4;
    __pyx_t_7 = -1;
    if (__pyx_t_21 < 0) {
      __pyx_t_21 += __pyx_pybuffernd_boxes.diminfo[0].shape;
      if (unlikely(__pyx_t_21 < 0)) __pyx_t_7 = 0;
    } else if (unlikely(__pyx_t_21 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
    if (__pyx_t_22 < 0) {
      __pyx_t_22 += __pyx_pybuffernd_boxes.diminfo[1].shape;
      if (unlikely(__pyx_t_22 < 0)) __pyx_t_7 = 1;
    } else if (unlikely(__pyx_t_22 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
    if (unlikely(__pyx_t_7 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_7);
      __PYX_ERR(0, 102, __pyx_L1_error)
    }
    __pyx_t_1 = PyFloat_FromDouble((*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_21, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_22, __pyx_pybuffernd_boxes.diminfo[1].strides))); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 102, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __pyx_t_5 = PyTuple_New(2); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 102, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_4);
    __Pyx_GIVEREF(__pyx_int_4);
    PyTuple_SET_ITEM(__pyx_t_5, 1, __pyx_int_4);
    if (unlikely(PyObject_SetItem(((PyObject *)__pyx_v_boxes), __pyx_t_5, __pyx_t_1) < 0)) __PYX_ERR(0, 102, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

    /* "nms/cpu_nms.pyx":105
 * 
 * 	# swap ith box with position of max box
 *         boxes[maxpos,0] = tx1             # <<<<<<<<<<<<<<
 *         boxes[maxpos,1] = ty1
 *         boxes[maxpos,2] = tx2
 */
    __pyx_t_23 = __pyx_v_maxpos;
    __pyx_t_24 = 0;
    __pyx_t_7 = -1;
    if (__pyx_t_23 < 0) {
      __pyx_t_23 += __pyx_pybuffernd_boxes.diminfo[0].shape;
      if (unlikely(__pyx_t_23 < 0)) __pyx_t_7 = 0;
    } else if (unlikely(__pyx_t_23 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
    if (__pyx_t_24 < 0) {
      __pyx_t_24 += __pyx_pybuffernd_boxes.diminfo[1].shape;
      if (unlikely(__pyx_t_24 < 0)) __pyx_t_7 = 1;
    } else if (unlikely(__pyx_t_24 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
    if (unlikely(__pyx_t_7 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_7);
      __PYX_ERR(0, 105, __pyx_L1_error)
    }
    *__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_23, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_24, __pyx_pybuffernd_boxes.diminfo[1].strides) = __pyx_v_tx1;

    /* "nms/cpu_nms.pyx":106
 * 	# swap ith box with position of max box
 *         boxes[maxpos,0] = tx1
 *         boxes[maxpos,1] = ty1             # <<<<<<<<<<<<<<
 *         boxes[maxpos,2] = tx2
 *         boxes[maxpos,3] = ty2
 */
    __pyx_t_25 = __pyx_v_maxpos;
    __pyx_t_26 = 1;
    __pyx_t_7 = -1;
    if (__pyx_t_25 < 0) {
      __pyx_t_25 += __pyx_pybuffernd_boxes.diminfo[0].shape;
      if (unlikely(__pyx_t_25 < 0)) __pyx_t_7 = 0;
    } else if (unlikely(__pyx_t_25 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
    if (__pyx_t_26 < 0) {
      __pyx_t_26 += __pyx_pybuffernd_boxes.diminfo[1].shape;
      if (unlikely(__pyx_t_26 < 0)) __pyx_t_7 = 1;
    } else if (unlikely(__pyx_t_26 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
    if (unlikely(__pyx_t_7 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_7);
      __PYX_ERR(0, 106, __pyx_L1_error)
    }
    *__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_25, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_26, __pyx_pybuffernd_boxes.diminfo[1].strides) = __pyx_v_ty1;

    /* "nms/cpu_nms.pyx":107
 *         boxes[maxpos,0] = tx1
 *         boxes[maxpos,1] = ty1
 *         boxes[maxpos,2] = tx2             # <<<<<<<<<<<<<<
 *         boxes[maxpos,3] = ty2
 *         boxes[maxpos,4] = ts
 */
    __pyx_t_27 = __pyx_v_maxpos;
    __pyx_t_28 = 2;
    __pyx_t_7 = -1;
    if (__pyx_t_27 < 0) {
      __pyx_t_27 += __pyx_pybuffernd_boxes.diminfo[0].shape;
      if (unlikely(__pyx_t_27 < 0)) __pyx_t_7 = 0;
    } else if (unlikely(__pyx_t_27 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
    if (__pyx_t_28 < 0) {
      __pyx_t_28 += __pyx_pybuffernd_boxes.diminfo[1].shape;
      if (unlikely(__pyx_t_28 < 0)) __pyx_t_7 = 1;
    } else if (unlikely(__pyx_t_28 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
    if (unlikely(__pyx_t_7 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_7);
      __PYX_ERR(0, 107, __pyx_L1_error)
    }
    *__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_27, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_28, __pyx_pybuffernd_boxes.diminfo[1].strides) = __pyx_v_tx2;

    /* "nms/cpu_nms.pyx":108
 *         boxes[maxpos,1] = ty1
 *         boxes[maxpos,2] = tx2
 *         boxes[maxpos,3] = ty2             # <<<<<<<<<<<<<<
 *         boxes[maxpos,4] = ts
 * 
 */
    __pyx_t_29 = __pyx_v_maxpos;
    __pyx_t_30 = 3;
    __pyx_t_7 = -1;
    if (__pyx_t_29 < 0) {
      __pyx_t_29 += __pyx_pybuffernd_boxes.diminfo[0].shape;
      if (unlikely(__pyx_t_29 < 0)) __pyx_t_7 = 0;
    } else if (unlikely(__pyx_t_29 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
    if (__pyx_t_30 < 0) {
      __pyx_t_30 += __pyx_pybuffernd_boxes.diminfo[1].shape;
      if (unlikely(__pyx_t_30 < 0)) __pyx_t_7 = 1;
    } else if (unlikely(__pyx_t_30 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
    if (unlikely(__pyx_t_7 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_7);
      __PYX_ERR(0, 108, __pyx_L1_error)
    }
    *__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_29, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_30, __pyx_pybuffernd_boxes.diminfo[1].strides) = __pyx_v_ty2;

    /* "nms/cpu_nms.pyx":109
 *         boxes[maxpos,2] = tx2
 *         boxes[maxpos,3] = ty2
 *         boxes[maxpos,4] = ts             # <<<<<<<<<<<<<<
 * 
 *         tx1 = boxes[i,0]
 */
    __pyx_t_31 = __pyx_v_maxpos;
    __pyx_t_32 = 4;
    __pyx_t_7 = -1;
    if (__pyx_t_31 < 0) {
      __pyx_t_31 += __pyx_pybuffernd_boxes.diminfo[0].shape;
      if (unlikely(__pyx_t_31 < 0)) __pyx_t_7 = 0;
    } else if (unlikely(__pyx_t_31 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
    if (__pyx_t_32 < 0) {
      __pyx_t_32 += __pyx_pybuffernd_boxes.diminfo[1].shape;
      if (unlikely(__pyx_t_32 < 0)) __pyx_t_7 = 1;
    } else if (unlikely(__pyx_t_32 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
    if (unlikely(__pyx_t_7 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_7);
      __PYX_ERR(0, 109, __pyx_L1_error)
    }
    *__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_31, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_32, __pyx_pybuffernd_boxes.diminfo[1].strides) = __pyx_v_ts;

    /* "nms/cpu_nms.pyx":111
 *         boxes[maxpos,4] = ts
 * 
 *         tx1 = boxes[i,0]             # <<<<<<<<<<<<<<
 *         ty1 = boxes[i,1]
 *         tx2 = boxes[i,2]
 */
    __pyx_t_1 = PyTuple_New(2); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 111, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_1, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_0);
    __Pyx_GIVEREF(__pyx_int_0);
    PyTuple_SET_ITEM(__pyx_t_1, 1, __pyx_int_0);
    __pyx_t_5 = PyObject_GetItem(((PyObject *)__pyx_v_boxes), __pyx_t_1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 111, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_t_6 = __pyx_PyFloat_AsFloat(__pyx_t_5); if (unlikely((__pyx_t_6 == (float)-1) && PyErr_Occurred())) __PYX_ERR(0, 111, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_v_tx1 = __pyx_t_6;

    /* "nms/cpu_nms.pyx":112
 * 
 *         tx1 = boxes[i,0]
 *         ty1 = boxes[i,1]             # <<<<<<<<<<<<<<
 *         tx2 = boxes[i,2]
 *         ty2 = boxes[i,3]
 */
    __pyx_t_5 = PyTuple_New(2); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 112, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_1);
    __Pyx_GIVEREF(__pyx_int_1);
    PyTuple_SET_ITEM(__pyx_t_5, 1, __pyx_int_1);
    __pyx_t_1 = PyObject_GetItem(((PyObject *)__pyx_v_boxes), __pyx_t_5); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 112, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_t_6 = __pyx_PyFloat_AsFloat(__pyx_t_1); if (unlikely((__pyx_t_6 == (float)-1) && PyErr_Occurred())) __PYX_ERR(0, 112, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_v_ty1 = __pyx_t_6;

    /* "nms/cpu_nms.pyx":113
 *         tx1 = boxes[i,0]
 *         ty1 = boxes[i,1]
 *         tx2 = boxes[i,2]             # <<<<<<<<<<<<<<
 *         ty2 = boxes[i,3]
 *         ts = boxes[i,4]
 */
    __pyx_t_1 = PyTuple_New(2); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 113, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_1, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_2);
    __Pyx_GIVEREF(__pyx_int_2);
    PyTuple_SET_ITEM(__pyx_t_1, 1, __pyx_int_2);
    __pyx_t_5 = PyObject_GetItem(((PyObject *)__pyx_v_boxes), __pyx_t_1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 113, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_t_6 = __pyx_PyFloat_AsFloat(__pyx_t_5); if (unlikely((__pyx_t_6 == (float)-1) && PyErr_Occurred())) __PYX_ERR(0, 113, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_v_tx2 = __pyx_t_6;

    /* "nms/cpu_nms.pyx":114
 *         ty1 = boxes[i,1]
 *         tx2 = boxes[i,2]
 *         ty2 = boxes[i,3]             # <<<<<<<<<<<<<<
 *         ts = boxes[i,4]
 * 
 */
    __pyx_t_5 = PyTuple_New(2); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 114, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_3);
    __Pyx_GIVEREF(__pyx_int_3);
    PyTuple_SET_ITEM(__pyx_t_5, 1, __pyx_int_3);
    __pyx_t_1 = PyObject_GetItem(((PyObject *)__pyx_v_boxes), __pyx_t_5); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 114, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_t_6 = __pyx_PyFloat_AsFloat(__pyx_t_1); if (unlikely((__pyx_t_6 == (float)-1) && PyErr_Occurred())) __PYX_ERR(0, 114, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_v_ty2 = __pyx_t_6;

    /* "nms/cpu_nms.pyx":115
 *         tx2 = boxes[i,2]
 *         ty2 = boxes[i,3]
 *         ts = boxes[i,4]             # <<<<<<<<<<<<<<
 * 
 *         pos = i + 1
 */
    __pyx_t_1 = PyTuple_New(2); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 115, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __Pyx_INCREF(__pyx_v_i);
    __Pyx_GIVEREF(__pyx_v_i);
    PyTuple_SET_ITEM(__pyx_t_1, 0, __pyx_v_i);
    __Pyx_INCREF(__pyx_int_4);
    __Pyx_GIVEREF(__pyx_int_4);
    PyTuple_SET_ITEM(__pyx_t_1, 1, __pyx_int_4);
    __pyx_t_5 = PyObject_GetItem(((PyObject *)__pyx_v_boxes), __pyx_t_1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 115, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_t_6 = __pyx_PyFloat_AsFloat(__pyx_t_5); if (unlikely((__pyx_t_6 == (float)-1) && PyErr_Occurred())) __PYX_ERR(0, 115, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_v_ts = __pyx_t_6;

    /* "nms/cpu_nms.pyx":117
 *         ts = boxes[i,4]
 * 
 *         pos = i + 1             # <<<<<<<<<<<<<<
 * 	# NMS iterations, note that N changes if detection boxes fall below threshold
 *         while pos < N:
 */
    __pyx_t_5 = __Pyx_PyInt_AddObjC(__pyx_v_i, __pyx_int_1, 1, 0); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 117, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __pyx_t_7 = __Pyx_PyInt_As_int(__pyx_t_5); if (unlikely((__pyx_t_7 == (int)-1) && PyErr_Occurred())) __PYX_ERR(0, 117, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_v_pos = __pyx_t_7;

    /* "nms/cpu_nms.pyx":119
 *         pos = i + 1
 * 	# NMS iterations, note that N changes if detection boxes fall below threshold
 *         while pos < N:             # <<<<<<<<<<<<<<
 *             x1 = boxes[pos, 0]
 *             y1 = boxes[pos, 1]
 */
    while (1) {
      __pyx_t_8 = ((__pyx_v_pos < __pyx_v_N) != 0);
      if (!__pyx_t_8) break;

      /* "nms/cpu_nms.pyx":120
 * 	# NMS iterations, note that N changes if detection boxes fall below threshold
 *         while pos < N:
 *             x1 = boxes[pos, 0]             # <<<<<<<<<<<<<<
 *             y1 = boxes[pos, 1]
 *             x2 = boxes[pos, 2]
 */
      __pyx_t_33 = __pyx_v_pos;
      __pyx_t_34 = 0;
      __pyx_t_7 = -1;
      if (__pyx_t_33 < 0) {
        __pyx_t_33 += __pyx_pybuffernd_boxes.diminfo[0].shape;
        if (unlikely(__pyx_t_33 < 0)) __pyx_t_7 = 0;
      } else if (unlikely(__pyx_t_33 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
      if (__pyx_t_34 < 0) {
        __pyx_t_34 += __pyx_pybuffernd_boxes.diminfo[1].shape;
        if (unlikely(__pyx_t_34 < 0)) __pyx_t_7 = 1;
      } else if (unlikely(__pyx_t_34 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
      if (unlikely(__pyx_t_7 != -1)) {
        __Pyx_RaiseBufferIndexError(__pyx_t_7);
        __PYX_ERR(0, 120, __pyx_L1_error)
      }
      __pyx_v_x1 = (*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_33, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_34, __pyx_pybuffernd_boxes.diminfo[1].strides));

      /* "nms/cpu_nms.pyx":121
 *         while pos < N:
 *             x1 = boxes[pos, 0]
 *             y1 = boxes[pos, 1]             # <<<<<<<<<<<<<<
 *             x2 = boxes[pos, 2]
 *             y2 = boxes[pos, 3]
 */
      __pyx_t_35 = __pyx_v_pos;
      __pyx_t_36 = 1;
      __pyx_t_7 = -1;
      if (__pyx_t_35 < 0) {
        __pyx_t_35 += __pyx_pybuffernd_boxes.diminfo[0].shape;
        if (unlikely(__pyx_t_35 < 0)) __pyx_t_7 = 0;
      } else if (unlikely(__pyx_t_35 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
      if (__pyx_t_36 < 0) {
        __pyx_t_36 += __pyx_pybuffernd_boxes.diminfo[1].shape;
        if (unlikely(__pyx_t_36 < 0)) __pyx_t_7 = 1;
      } else if (unlikely(__pyx_t_36 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
      if (unlikely(__pyx_t_7 != -1)) {
        __Pyx_RaiseBufferIndexError(__pyx_t_7);
        __PYX_ERR(0, 121, __pyx_L1_error)
      }
      __pyx_v_y1 = (*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_35, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_36, __pyx_pybuffernd_boxes.diminfo[1].strides));

      /* "nms/cpu_nms.pyx":122
 *             x1 = boxes[pos, 0]
 *             y1 = boxes[pos, 1]
 *             x2 = boxes[pos, 2]             # <<<<<<<<<<<<<<
 *             y2 = boxes[pos, 3]
 *             s = boxes[pos, 4]
 */
      __pyx_t_37 = __pyx_v_pos;
      __pyx_t_38 = 2;
      __pyx_t_7 = -1;
      if (__pyx_t_37 < 0) {
        __pyx_t_37 += __pyx_pybuffernd_boxes.diminfo[0].shape;
        if (unlikely(__pyx_t_37 < 0)) __pyx_t_7 = 0;
      } else if (unlikely(__pyx_t_37 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
      if (__pyx_t_38 < 0) {
        __pyx_t_38 += __pyx_pybuffernd_boxes.diminfo[1].shape;
        if (unlikely(__pyx_t_38 < 0)) __pyx_t_7 = 1;
      } else if (unlikely(__pyx_t_38 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
      if (unlikely(__pyx_t_7 != -1)) {
        __Pyx_RaiseBufferIndexError(__pyx_t_7);
        __PYX_ERR(0, 122, __pyx_L1_error)
      }
      __pyx_v_x2 = (*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_37, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_38, __pyx_pybuffernd_boxes.diminfo[1].strides));

      /* "nms/cpu_nms.pyx":123
 *             y1 = boxes[pos, 1]
 *             x2 = boxes[pos, 2]
 *             y2 = boxes[pos, 3]             # <<<<<<<<<<<<<<
 *             s = boxes[pos, 4]
 * 
 */
      __pyx_t_39 = __pyx_v_pos;
      __pyx_t_40 = 3;
      __pyx_t_7 = -1;
      if (__pyx_t_39 < 0) {
        __pyx_t_39 += __pyx_pybuffernd_boxes.diminfo[0].shape;
        if (unlikely(__pyx_t_39 < 0)) __pyx_t_7 = 0;
      } else if (unlikely(__pyx_t_39 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
      if (__pyx_t_40 < 0) {
        __pyx_t_40 += __pyx_pybuffernd_boxes.diminfo[1].shape;
        if (unlikely(__pyx_t_40 < 0)) __pyx_t_7 = 1;
      } else if (unlikely(__pyx_t_40 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
      if (unlikely(__pyx_t_7 != -1)) {
        __Pyx_RaiseBufferIndexError(__pyx_t_7);
        __PYX_ERR(0, 123, __pyx_L1_error)
      }
      __pyx_v_y2 = (*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_39, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_40, __pyx_pybuffernd_boxes.diminfo[1].strides));

      /* "nms/cpu_nms.pyx":124
 *             x2 = boxes[pos, 2]
 *             y2 = boxes[pos, 3]
 *             s = boxes[pos, 4]             # <<<<<<<<<<<<<<
 * 
 *             area = (x2 - x1 + 1) * (y2 - y1 + 1)
 */
      __pyx_t_41 = __pyx_v_pos;
      __pyx_t_42 = 4;
      __pyx_t_7 = -1;
      if (__pyx_t_41 < 0) {
        __pyx_t_41 += __pyx_pybuffernd_boxes.diminfo[0].shape;
        if (unlikely(__pyx_t_41 < 0)) __pyx_t_7 = 0;
      } else if (unlikely(__pyx_t_41 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
      if (__pyx_t_42 < 0) {
        __pyx_t_42 += __pyx_pybuffernd_boxes.diminfo[1].shape;
        if (unlikely(__pyx_t_42 < 0)) __pyx_t_7 = 1;
      } else if (unlikely(__pyx_t_42 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
      if (unlikely(__pyx_t_7 != -1)) {
        __Pyx_RaiseBufferIndexError(__pyx_t_7);
        __PYX_ERR(0, 124, __pyx_L1_error)
      }
      __pyx_t_5 = PyFloat_FromDouble((*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_41, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_42, __pyx_pybuffernd_boxes.diminfo[1].strides))); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 124, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_XDECREF_SET(__pyx_v_s, __pyx_t_5);
      __pyx_t_5 = 0;

      /* "nms/cpu_nms.pyx":126
 *             s = boxes[pos, 4]
 * 
 *             area = (x2 - x1 + 1) * (y2 - y1 + 1)             # <<<<<<<<<<<<<<
 *             iw = (min(tx2, x2) - max(tx1, x1) + 1)
 *             if iw > 0:
 */
      __pyx_v_area = (((__pyx_v_x2 - __pyx_v_x1) + 1.0) * ((__pyx_v_y2 - __pyx_v_y1) + 1.0));

      /* "nms/cpu_nms.pyx":127
 * 
 *             area = (x2 - x1 + 1) * (y2 - y1 + 1)
 *             iw = (min(tx2, x2) - max(tx1, x1) + 1)             # <<<<<<<<<<<<<<
 *             if iw > 0:
 *                 ih = (min(ty2, y2) - max(ty1, y1) + 1)
 */
      __pyx_v_iw = ((__pyx_f_3nms_7cpu_nms_min(__pyx_v_tx2, __pyx_v_x2) - __pyx_f_3nms_7cpu_nms_max(__pyx_v_tx1, __pyx_v_x1)) + 1.0);

      /* "nms/cpu_nms.pyx":128
 *             area = (x2 - x1 + 1) * (y2 - y1 + 1)
 *             iw = (min(tx2, x2) - max(tx1, x1) + 1)
 *             if iw > 0:             # <<<<<<<<<<<<<<
 *                 ih = (min(ty2, y2) - max(ty1, y1) + 1)
 *                 if ih > 0:
 */
      __pyx_t_8 = ((__pyx_v_iw > 0.0) != 0);
      if (__pyx_t_8) {

        /* "nms/cpu_nms.pyx":129
 *             iw = (min(tx2, x2) - max(tx1, x1) + 1)
 *             if iw > 0:
 *                 ih = (min(ty2, y2) - max(ty1, y1) + 1)             # <<<<<<<<<<<<<<
 *                 if ih > 0:
 *                     ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
 */
        __pyx_v_ih = ((__pyx_f_3nms_7cpu_nms_min(__pyx_v_ty2, __pyx_v_y2) - __pyx_f_3nms_7cpu_nms_max(__pyx_v_ty1, __pyx_v_y1)) + 1.0);

        /* "nms/cpu_nms.pyx":130
 *             if iw > 0:
 *                 ih = (min(ty2, y2) - max(ty1, y1) + 1)
 *                 if ih > 0:             # <<<<<<<<<<<<<<
 *                     ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
 *                     ov = iw * ih / ua #iou between max box and detection box
 */
        __pyx_t_8 = ((__pyx_v_ih > 0.0) != 0);
        if (__pyx_t_8) {

          /* "nms/cpu_nms.pyx":131
 *                 ih = (min(ty2, y2) - max(ty1, y1) + 1)
 *                 if ih > 0:
 *                     ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)             # <<<<<<<<<<<<<<
 *                     ov = iw * ih / ua #iou between max box and detection box
 * 
 */
          __pyx_v_ua = ((double)(((((__pyx_v_tx2 - __pyx_v_tx1) + 1.0) * ((__pyx_v_ty2 - __pyx_v_ty1) + 1.0)) + __pyx_v_area) - (__pyx_v_iw * __pyx_v_ih)));

          /* "nms/cpu_nms.pyx":132
 *                 if ih > 0:
 *                     ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
 *                     ov = iw * ih / ua #iou between max box and detection box             # <<<<<<<<<<<<<<
 * 
 *                     if method == 1: # linear
 */
          __pyx_t_6 = (__pyx_v_iw * __pyx_v_ih);
          if (unlikely(__pyx_v_ua == 0)) {
            PyErr_SetString(PyExc_ZeroDivisionError, "float division");
            __PYX_ERR(0, 132, __pyx_L1_error)
          }
          __pyx_v_ov = (__pyx_t_6 / __pyx_v_ua);

          /* "nms/cpu_nms.pyx":134
 *                     ov = iw * ih / ua #iou between max box and detection box
 * 
 *                     if method == 1: # linear             # <<<<<<<<<<<<<<
 *                         if ov > Nt:
 *                             weight = 1 - ov
 */
          switch (__pyx_v_method) {
            case 1:

            /* "nms/cpu_nms.pyx":135
 * 
 *                     if method == 1: # linear
 *                         if ov > Nt:             # <<<<<<<<<<<<<<
 *                             weight = 1 - ov
 *                         else:
 */
            __pyx_t_8 = ((__pyx_v_ov > __pyx_v_Nt) != 0);
            if (__pyx_t_8) {

              /* "nms/cpu_nms.pyx":136
 *                     if method == 1: # linear
 *                         if ov > Nt:
 *                             weight = 1 - ov             # <<<<<<<<<<<<<<
 *                         else:
 *                             weight = 1
 */
              __pyx_v_weight = (1.0 - __pyx_v_ov);

              /* "nms/cpu_nms.pyx":135
 * 
 *                     if method == 1: # linear
 *                         if ov > Nt:             # <<<<<<<<<<<<<<
 *                             weight = 1 - ov
 *                         else:
 */
              goto __pyx_L12;
            }

            /* "nms/cpu_nms.pyx":138
 *                             weight = 1 - ov
 *                         else:
 *                             weight = 1             # <<<<<<<<<<<<<<
 *                     elif method == 2: # gaussian
 *                         weight = np.exp(-(ov * ov)/sigma)
 */
            /*else*/ {
              __pyx_v_weight = 1.0;
            }
            __pyx_L12:;

            /* "nms/cpu_nms.pyx":134
 *                     ov = iw * ih / ua #iou between max box and detection box
 * 
 *                     if method == 1: # linear             # <<<<<<<<<<<<<<
 *                         if ov > Nt:
 *                             weight = 1 - ov
 */
            break;

            /* "nms/cpu_nms.pyx":139
 *                         else:
 *                             weight = 1
 *                     elif method == 2: # gaussian             # <<<<<<<<<<<<<<
 *                         weight = np.exp(-(ov * ov)/sigma)
 *                     else: # original NMS
 */
            case 2:

            /* "nms/cpu_nms.pyx":140
 *                             weight = 1
 *                     elif method == 2: # gaussian
 *                         weight = np.exp(-(ov * ov)/sigma)             # <<<<<<<<<<<<<<
 *                     else: # original NMS
 *                         if ov > Nt:
 */
            __pyx_t_1 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 140, __pyx_L1_error)
            __Pyx_GOTREF(__pyx_t_1);
            __pyx_t_43 = __Pyx_PyObject_GetAttrStr(__pyx_t_1, __pyx_n_s_exp); if (unlikely(!__pyx_t_43)) __PYX_ERR(0, 140, __pyx_L1_error)
            __Pyx_GOTREF(__pyx_t_43);
            __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
            __pyx_t_6 = (-(__pyx_v_ov * __pyx_v_ov));
            if (unlikely(__pyx_v_sigma == 0)) {
              PyErr_SetString(PyExc_ZeroDivisionError, "float division");
              __PYX_ERR(0, 140, __pyx_L1_error)
            }
            __pyx_t_1 = PyFloat_FromDouble((__pyx_t_6 / __pyx_v_sigma)); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 140, __pyx_L1_error)
            __Pyx_GOTREF(__pyx_t_1);
            __pyx_t_44 = NULL;
            if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_43))) {
              __pyx_t_44 = PyMethod_GET_SELF(__pyx_t_43);
              if (likely(__pyx_t_44)) {
                PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_43);
                __Pyx_INCREF(__pyx_t_44);
                __Pyx_INCREF(function);
                __Pyx_DECREF_SET(__pyx_t_43, function);
              }
            }
            if (!__pyx_t_44) {
              __pyx_t_5 = __Pyx_PyObject_CallOneArg(__pyx_t_43, __pyx_t_1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 140, __pyx_L1_error)
              __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
              __Pyx_GOTREF(__pyx_t_5);
            } else {
              #if CYTHON_FAST_PYCALL
              if (PyFunction_Check(__pyx_t_43)) {
                PyObject *__pyx_temp[2] = {__pyx_t_44, __pyx_t_1};
                __pyx_t_5 = __Pyx_PyFunction_FastCall(__pyx_t_43, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 140, __pyx_L1_error)
                __Pyx_XDECREF(__pyx_t_44); __pyx_t_44 = 0;
                __Pyx_GOTREF(__pyx_t_5);
                __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
              } else
              #endif
              #if CYTHON_FAST_PYCCALL
              if (__Pyx_PyFastCFunction_Check(__pyx_t_43)) {
                PyObject *__pyx_temp[2] = {__pyx_t_44, __pyx_t_1};
                __pyx_t_5 = __Pyx_PyCFunction_FastCall(__pyx_t_43, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 140, __pyx_L1_error)
                __Pyx_XDECREF(__pyx_t_44); __pyx_t_44 = 0;
                __Pyx_GOTREF(__pyx_t_5);
                __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
              } else
              #endif
              {
                __pyx_t_45 = PyTuple_New(1+1); if (unlikely(!__pyx_t_45)) __PYX_ERR(0, 140, __pyx_L1_error)
                __Pyx_GOTREF(__pyx_t_45);
                __Pyx_GIVEREF(__pyx_t_44); PyTuple_SET_ITEM(__pyx_t_45, 0, __pyx_t_44); __pyx_t_44 = NULL;
                __Pyx_GIVEREF(__pyx_t_1);
                PyTuple_SET_ITEM(__pyx_t_45, 0+1, __pyx_t_1);
                __pyx_t_1 = 0;
                __pyx_t_5 = __Pyx_PyObject_Call(__pyx_t_43, __pyx_t_45, NULL); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 140, __pyx_L1_error)
                __Pyx_GOTREF(__pyx_t_5);
                __Pyx_DECREF(__pyx_t_45); __pyx_t_45 = 0;
              }
            }
            __Pyx_DECREF(__pyx_t_43); __pyx_t_43 = 0;
            __pyx_t_6 = __pyx_PyFloat_AsFloat(__pyx_t_5); if (unlikely((__pyx_t_6 == (float)-1) && PyErr_Occurred())) __PYX_ERR(0, 140, __pyx_L1_error)
            __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
            __pyx_v_weight = __pyx_t_6;

            /* "nms/cpu_nms.pyx":139
 *                         else:
 *                             weight = 1
 *                     elif method == 2: # gaussian             # <<<<<<<<<<<<<<
 *                         weight = np.exp(-(ov * ov)/sigma)
 *                     else: # original NMS
 */
            break;
            default:

            /* "nms/cpu_nms.pyx":142
 *                         weight = np.exp(-(ov * ov)/sigma)
 *                     else: # original NMS
 *                         if ov > Nt:             # <<<<<<<<<<<<<<
 *                             weight = 0
 *                         else:
 */
            __pyx_t_8 = ((__pyx_v_ov > __pyx_v_Nt) != 0);
            if (__pyx_t_8) {

              /* "nms/cpu_nms.pyx":143
 *                     else: # original NMS
 *                         if ov > Nt:
 *                             weight = 0             # <<<<<<<<<<<<<<
 *                         else:
 *                             weight = 1
 */
              __pyx_v_weight = 0.0;

              /* "nms/cpu_nms.pyx":142
 *                         weight = np.exp(-(ov * ov)/sigma)
 *                     else: # original NMS
 *                         if ov > Nt:             # <<<<<<<<<<<<<<
 *                             weight = 0
 *                         else:
 */
              goto __pyx_L13;
            }

            /* "nms/cpu_nms.pyx":145
 *                             weight = 0
 *                         else:
 *                             weight = 1             # <<<<<<<<<<<<<<
 * 
 *                     boxes[pos, 4] = weight*boxes[pos, 4]
 */
            /*else*/ {
              __pyx_v_weight = 1.0;
            }
            __pyx_L13:;
            break;
          }

          /* "nms/cpu_nms.pyx":147
 *                             weight = 1
 * 
 *                     boxes[pos, 4] = weight*boxes[pos, 4]             # <<<<<<<<<<<<<<
 * 
 * 		    # if box score falls below threshold, discard the box by swapping with last box
 */
          __pyx_t_46 = __pyx_v_pos;
          __pyx_t_47 = 4;
          __pyx_t_7 = -1;
          if (__pyx_t_46 < 0) {
            __pyx_t_46 += __pyx_pybuffernd_boxes.diminfo[0].shape;
            if (unlikely(__pyx_t_46 < 0)) __pyx_t_7 = 0;
          } else if (unlikely(__pyx_t_46 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
          if (__pyx_t_47 < 0) {
            __pyx_t_47 += __pyx_pybuffernd_boxes.diminfo[1].shape;
            if (unlikely(__pyx_t_47 < 0)) __pyx_t_7 = 1;
          } else if (unlikely(__pyx_t_47 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
          if (unlikely(__pyx_t_7 != -1)) {
            __Pyx_RaiseBufferIndexError(__pyx_t_7);
            __PYX_ERR(0, 147, __pyx_L1_error)
          }
          __pyx_t_48 = __pyx_v_pos;
          __pyx_t_49 = 4;
          __pyx_t_7 = -1;
          if (__pyx_t_48 < 0) {
            __pyx_t_48 += __pyx_pybuffernd_boxes.diminfo[0].shape;
            if (unlikely(__pyx_t_48 < 0)) __pyx_t_7 = 0;
          } else if (unlikely(__pyx_t_48 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
          if (__pyx_t_49 < 0) {
            __pyx_t_49 += __pyx_pybuffernd_boxes.diminfo[1].shape;
            if (unlikely(__pyx_t_49 < 0)) __pyx_t_7 = 1;
          } else if (unlikely(__pyx_t_49 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
          if (unlikely(__pyx_t_7 != -1)) {
            __Pyx_RaiseBufferIndexError(__pyx_t_7);
            __PYX_ERR(0, 147, __pyx_L1_error)
          }
          *__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_48, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_49, __pyx_pybuffernd_boxes.diminfo[1].strides) = (__pyx_v_weight * (*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_46, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_47, __pyx_pybuffernd_boxes.diminfo[1].strides)));

          /* "nms/cpu_nms.pyx":151
 * 		    # if box score falls below threshold, discard the box by swapping with last box
 * 		    # update N
 *                     if boxes[pos, 4] < threshold:             # <<<<<<<<<<<<<<
 *                         boxes[pos,0] = boxes[N-1, 0]
 *                         boxes[pos,1] = boxes[N-1, 1]
 */
          __pyx_t_50 = __pyx_v_pos;
          __pyx_t_51 = 4;
          __pyx_t_7 = -1;
          if (__pyx_t_50 < 0) {
            __pyx_t_50 += __pyx_pybuffernd_boxes.diminfo[0].shape;
            if (unlikely(__pyx_t_50 < 0)) __pyx_t_7 = 0;
          } else if (unlikely(__pyx_t_50 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
          if (__pyx_t_51 < 0) {
            __pyx_t_51 += __pyx_pybuffernd_boxes.diminfo[1].shape;
            if (unlikely(__pyx_t_51 < 0)) __pyx_t_7 = 1;
          } else if (unlikely(__pyx_t_51 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
          if (unlikely(__pyx_t_7 != -1)) {
            __Pyx_RaiseBufferIndexError(__pyx_t_7);
            __PYX_ERR(0, 151, __pyx_L1_error)
          }
          __pyx_t_8 = (((*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_50, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_51, __pyx_pybuffernd_boxes.diminfo[1].strides)) < __pyx_v_threshold) != 0);
          if (__pyx_t_8) {

            /* "nms/cpu_nms.pyx":152
 * 		    # update N
 *                     if boxes[pos, 4] < threshold:
 *                         boxes[pos,0] = boxes[N-1, 0]             # <<<<<<<<<<<<<<
 *                         boxes[pos,1] = boxes[N-1, 1]
 *                         boxes[pos,2] = boxes[N-1, 2]
 */
            __pyx_t_52 = (__pyx_v_N - 1);
            __pyx_t_53 = 0;
            __pyx_t_7 = -1;
            if (__pyx_t_52 < 0) {
              __pyx_t_52 += __pyx_pybuffernd_boxes.diminfo[0].shape;
              if (unlikely(__pyx_t_52 < 0)) __pyx_t_7 = 0;
            } else if (unlikely(__pyx_t_52 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
            if (__pyx_t_53 < 0) {
              __pyx_t_53 += __pyx_pybuffernd_boxes.diminfo[1].shape;
              if (unlikely(__pyx_t_53 < 0)) __pyx_t_7 = 1;
            } else if (unlikely(__pyx_t_53 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
            if (unlikely(__pyx_t_7 != -1)) {
              __Pyx_RaiseBufferIndexError(__pyx_t_7);
              __PYX_ERR(0, 152, __pyx_L1_error)
            }
            __pyx_t_54 = __pyx_v_pos;
            __pyx_t_55 = 0;
            __pyx_t_7 = -1;
            if (__pyx_t_54 < 0) {
              __pyx_t_54 += __pyx_pybuffernd_boxes.diminfo[0].shape;
              if (unlikely(__pyx_t_54 < 0)) __pyx_t_7 = 0;
            } else if (unlikely(__pyx_t_54 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
            if (__pyx_t_55 < 0) {
              __pyx_t_55 += __pyx_pybuffernd_boxes.diminfo[1].shape;
              if (unlikely(__pyx_t_55 < 0)) __pyx_t_7 = 1;
            } else if (unlikely(__pyx_t_55 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
            if (unlikely(__pyx_t_7 != -1)) {
              __Pyx_RaiseBufferIndexError(__pyx_t_7);
              __PYX_ERR(0, 152, __pyx_L1_error)
            }
            *__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_54, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_55, __pyx_pybuffernd_boxes.diminfo[1].strides) = (*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_52, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_53, __pyx_pybuffernd_boxes.diminfo[1].strides));

            /* "nms/cpu_nms.pyx":153
 *                     if boxes[pos, 4] < threshold:
 *                         boxes[pos,0] = boxes[N-1, 0]
 *                         boxes[pos,1] = boxes[N-1, 1]             # <<<<<<<<<<<<<<
 *                         boxes[pos,2] = boxes[N-1, 2]
 *                         boxes[pos,3] = boxes[N-1, 3]
 */
            __pyx_t_56 = (__pyx_v_N - 1);
            __pyx_t_57 = 1;
            __pyx_t_7 = -1;
            if (__pyx_t_56 < 0) {
              __pyx_t_56 += __pyx_pybuffernd_boxes.diminfo[0].shape;
              if (unlikely(__pyx_t_56 < 0)) __pyx_t_7 = 0;
            } else if (unlikely(__pyx_t_56 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
            if (__pyx_t_57 < 0) {
              __pyx_t_57 += __pyx_pybuffernd_boxes.diminfo[1].shape;
              if (unlikely(__pyx_t_57 < 0)) __pyx_t_7 = 1;
            } else if (unlikely(__pyx_t_57 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
            if (unlikely(__pyx_t_7 != -1)) {
              __Pyx_RaiseBufferIndexError(__pyx_t_7);
              __PYX_ERR(0, 153, __pyx_L1_error)
            }
            __pyx_t_58 = __pyx_v_pos;
            __pyx_t_59 = 1;
            __pyx_t_7 = -1;
            if (__pyx_t_58 < 0) {
              __pyx_t_58 += __pyx_pybuffernd_boxes.diminfo[0].shape;
              if (unlikely(__pyx_t_58 < 0)) __pyx_t_7 = 0;
            } else if (unlikely(__pyx_t_58 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
            if (__pyx_t_59 < 0) {
              __pyx_t_59 += __pyx_pybuffernd_boxes.diminfo[1].shape;
              if (unlikely(__pyx_t_59 < 0)) __pyx_t_7 = 1;
            } else if (unlikely(__pyx_t_59 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
            if (unlikely(__pyx_t_7 != -1)) {
              __Pyx_RaiseBufferIndexError(__pyx_t_7);
              __PYX_ERR(0, 153, __pyx_L1_error)
            }
            *__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_58, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_59, __pyx_pybuffernd_boxes.diminfo[1].strides) = (*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_56, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_57, __pyx_pybuffernd_boxes.diminfo[1].strides));

            /* "nms/cpu_nms.pyx":154
 *                         boxes[pos,0] = boxes[N-1, 0]
 *                         boxes[pos,1] = boxes[N-1, 1]
 *                         boxes[pos,2] = boxes[N-1, 2]             # <<<<<<<<<<<<<<
 *                         boxes[pos,3] = boxes[N-1, 3]
 *                         boxes[pos,4] = boxes[N-1, 4]
 */
            __pyx_t_60 = (__pyx_v_N - 1);
            __pyx_t_61 = 2;
            __pyx_t_7 = -1;
            if (__pyx_t_60 < 0) {
              __pyx_t_60 += __pyx_pybuffernd_boxes.diminfo[0].shape;
              if (unlikely(__pyx_t_60 < 0)) __pyx_t_7 = 0;
            } else if (unlikely(__pyx_t_60 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
            if (__pyx_t_61 < 0) {
              __pyx_t_61 += __pyx_pybuffernd_boxes.diminfo[1].shape;
              if (unlikely(__pyx_t_61 < 0)) __pyx_t_7 = 1;
            } else if (unlikely(__pyx_t_61 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
            if (unlikely(__pyx_t_7 != -1)) {
              __Pyx_RaiseBufferIndexError(__pyx_t_7);
              __PYX_ERR(0, 154, __pyx_L1_error)
            }
            __pyx_t_62 = __pyx_v_pos;
            __pyx_t_63 = 2;
            __pyx_t_7 = -1;
            if (__pyx_t_62 < 0) {
              __pyx_t_62 += __pyx_pybuffernd_boxes.diminfo[0].shape;
              if (unlikely(__pyx_t_62 < 0)) __pyx_t_7 = 0;
            } else if (unlikely(__pyx_t_62 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
            if (__pyx_t_63 < 0) {
              __pyx_t_63 += __pyx_pybuffernd_boxes.diminfo[1].shape;
              if (unlikely(__pyx_t_63 < 0)) __pyx_t_7 = 1;
            } else if (unlikely(__pyx_t_63 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
            if (unlikely(__pyx_t_7 != -1)) {
              __Pyx_RaiseBufferIndexError(__pyx_t_7);
              __PYX_ERR(0, 154, __pyx_L1_error)
            }
            *__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_62, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_63, __pyx_pybuffernd_boxes.diminfo[1].strides) = (*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_60, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_61, __pyx_pybuffernd_boxes.diminfo[1].strides));

            /* "nms/cpu_nms.pyx":155
 *                         boxes[pos,1] = boxes[N-1, 1]
 *                         boxes[pos,2] = boxes[N-1, 2]
 *                         boxes[pos,3] = boxes[N-1, 3]             # <<<<<<<<<<<<<<
 *                         boxes[pos,4] = boxes[N-1, 4]
 *                         N = N - 1
 */
            __pyx_t_64 = (__pyx_v_N - 1);
            __pyx_t_65 = 3;
            __pyx_t_7 = -1;
            if (__pyx_t_64 < 0) {
              __pyx_t_64 += __pyx_pybuffernd_boxes.diminfo[0].shape;
              if (unlikely(__pyx_t_64 < 0)) __pyx_t_7 = 0;
            } else if (unlikely(__pyx_t_64 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
            if (__pyx_t_65 < 0) {
              __pyx_t_65 += __pyx_pybuffernd_boxes.diminfo[1].shape;
              if (unlikely(__pyx_t_65 < 0)) __pyx_t_7 = 1;
            } else if (unlikely(__pyx_t_65 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
            if (unlikely(__pyx_t_7 != -1)) {
              __Pyx_RaiseBufferIndexError(__pyx_t_7);
              __PYX_ERR(0, 155, __pyx_L1_error)
            }
            __pyx_t_66 = __pyx_v_pos;
            __pyx_t_67 = 3;
            __pyx_t_7 = -1;
            if (__pyx_t_66 < 0) {
              __pyx_t_66 += __pyx_pybuffernd_boxes.diminfo[0].shape;
              if (unlikely(__pyx_t_66 < 0)) __pyx_t_7 = 0;
            } else if (unlikely(__pyx_t_66 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
            if (__pyx_t_67 < 0) {
              __pyx_t_67 += __pyx_pybuffernd_boxes.diminfo[1].shape;
              if (unlikely(__pyx_t_67 < 0)) __pyx_t_7 = 1;
            } else if (unlikely(__pyx_t_67 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
            if (unlikely(__pyx_t_7 != -1)) {
              __Pyx_RaiseBufferIndexError(__pyx_t_7);
              __PYX_ERR(0, 155, __pyx_L1_error)
            }
            *__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_66, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_67, __pyx_pybuffernd_boxes.diminfo[1].strides) = (*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_64, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_65, __pyx_pybuffernd_boxes.diminfo[1].strides));

            /* "nms/cpu_nms.pyx":156
 *                         boxes[pos,2] = boxes[N-1, 2]
 *                         boxes[pos,3] = boxes[N-1, 3]
 *                         boxes[pos,4] = boxes[N-1, 4]             # <<<<<<<<<<<<<<
 *                         N = N - 1
 *                         pos = pos - 1
 */
            __pyx_t_68 = (__pyx_v_N - 1);
            __pyx_t_69 = 4;
            __pyx_t_7 = -1;
            if (__pyx_t_68 < 0) {
              __pyx_t_68 += __pyx_pybuffernd_boxes.diminfo[0].shape;
              if (unlikely(__pyx_t_68 < 0)) __pyx_t_7 = 0;
            } else if (unlikely(__pyx_t_68 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
            if (__pyx_t_69 < 0) {
              __pyx_t_69 += __pyx_pybuffernd_boxes.diminfo[1].shape;
              if (unlikely(__pyx_t_69 < 0)) __pyx_t_7 = 1;
            } else if (unlikely(__pyx_t_69 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
            if (unlikely(__pyx_t_7 != -1)) {
              __Pyx_RaiseBufferIndexError(__pyx_t_7);
              __PYX_ERR(0, 156, __pyx_L1_error)
            }
            __pyx_t_70 = __pyx_v_pos;
            __pyx_t_71 = 4;
            __pyx_t_7 = -1;
            if (__pyx_t_70 < 0) {
              __pyx_t_70 += __pyx_pybuffernd_boxes.diminfo[0].shape;
              if (unlikely(__pyx_t_70 < 0)) __pyx_t_7 = 0;
            } else if (unlikely(__pyx_t_70 >= __pyx_pybuffernd_boxes.diminfo[0].shape)) __pyx_t_7 = 0;
            if (__pyx_t_71 < 0) {
              __pyx_t_71 += __pyx_pybuffernd_boxes.diminfo[1].shape;
              if (unlikely(__pyx_t_71 < 0)) __pyx_t_7 = 1;
            } else if (unlikely(__pyx_t_71 >= __pyx_pybuffernd_boxes.diminfo[1].shape)) __pyx_t_7 = 1;
            if (unlikely(__pyx_t_7 != -1)) {
              __Pyx_RaiseBufferIndexError(__pyx_t_7);
              __PYX_ERR(0, 156, __pyx_L1_error)
            }
            *__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_70, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_71, __pyx_pybuffernd_boxes.diminfo[1].strides) = (*__Pyx_BufPtrStrided2d(float *, __pyx_pybuffernd_boxes.rcbuffer->pybuffer.buf, __pyx_t_68, __pyx_pybuffernd_boxes.diminfo[0].strides, __pyx_t_69, __pyx_pybuffernd_boxes.diminfo[1].strides));

            /* "nms/cpu_nms.pyx":157
 *                         boxes[pos,3] = boxes[N-1, 3]
 *                         boxes[pos,4] = boxes[N-1, 4]
 *                         N = N - 1             # <<<<<<<<<<<<<<
 *                         pos = pos - 1
 * 
 */
            __pyx_v_N = (__pyx_v_N - 1);

            /* "nms/cpu_nms.pyx":158
 *                         boxes[pos,4] = boxes[N-1, 4]
 *                         N = N - 1
 *                         pos = pos - 1             # <<<<<<<<<<<<<<
 * 
 *             pos = pos + 1
 */
            __pyx_v_pos = (__pyx_v_pos - 1);

            /* "nms/cpu_nms.pyx":151
 * 		    # if box score falls below threshold, discard the box by swapping with last box
 * 		    # update N
 *                     if boxes[pos, 4] < threshold:             # <<<<<<<<<<<<<<
 *                         boxes[pos,0] = boxes[N-1, 0]
 *                         boxes[pos,1] = boxes[N-1, 1]
 */
          }

          /* "nms/cpu_nms.pyx":130
 *             if iw > 0:
 *                 ih = (min(ty2, y2) - max(ty1, y1) + 1)
 *                 if ih > 0:             # <<<<<<<<<<<<<<
 *                     ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
 *                     ov = iw * ih / ua #iou between max box and detection box
 */
        }

        /* "nms/cpu_nms.pyx":128
 *             area = (x2 - x1 + 1) * (y2 - y1 + 1)
 *             iw = (min(tx2, x2) - max(tx1, x1) + 1)
 *             if iw > 0:             # <<<<<<<<<<<<<<
 *                 ih = (min(ty2, y2) - max(ty1, y1) + 1)
 *                 if ih > 0:
 */
      }

      /* "nms/cpu_nms.pyx":160
 *                         pos = pos - 1
 * 
 *             pos = pos + 1             # <<<<<<<<<<<<<<
 * 
 *     keep = [i for i in range(N)]
 */
      __pyx_v_pos = (__pyx_v_pos + 1);
    }

    /* "nms/cpu_nms.pyx":79
 *     cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov
 * 
 *     for i in range(N):             # <<<<<<<<<<<<<<
 *         maxscore = boxes[i, 4]
 *         maxpos = i
 */
  }
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;

  /* "nms/cpu_nms.pyx":162
 *             pos = pos + 1
 * 
 *     keep = [i for i in range(N)]             # <<<<<<<<<<<<<<
 *     return keep
 */
  __pyx_t_2 = PyList_New(0); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 162, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_5 = __Pyx_PyInt_From_unsigned_int(__pyx_v_N); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 162, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __pyx_t_43 = PyTuple_New(1); if (unlikely(!__pyx_t_43)) __PYX_ERR(0, 162, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_43);
  __Pyx_GIVEREF(__pyx_t_5);
  PyTuple_SET_ITEM(__pyx_t_43, 0, __pyx_t_5);
  __pyx_t_5 = 0;
  __pyx_t_5 = __Pyx_PyObject_Call(__pyx_builtin_range, __pyx_t_43, NULL); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 162, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_DECREF(__pyx_t_43); __pyx_t_43 = 0;
  if (likely(PyList_CheckExact(__pyx_t_5)) || PyTuple_CheckExact(__pyx_t_5)) {
    __pyx_t_43 = __pyx_t_5; __Pyx_INCREF(__pyx_t_43); __pyx_t_3 = 0;
    __pyx_t_4 = NULL;
  } else {
    __pyx_t_3 = -1; __pyx_t_43 = PyObject_GetIter(__pyx_t_5); if (unlikely(!__pyx_t_43)) __PYX_ERR(0, 162, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_43);
    __pyx_t_4 = Py_TYPE(__pyx_t_43)->tp_iternext; if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 162, __pyx_L1_error)
  }
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  for (;;) {
    if (likely(!__pyx_t_4)) {
      if (likely(PyList_CheckExact(__pyx_t_43))) {
        if (__pyx_t_3 >= PyList_GET_SIZE(__pyx_t_43)) break;
        #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
        __pyx_t_5 = PyList_GET_ITEM(__pyx_t_43, __pyx_t_3); __Pyx_INCREF(__pyx_t_5); __pyx_t_3++; if (unlikely(0 < 0)) __PYX_ERR(0, 162, __pyx_L1_error)
        #else
        __pyx_t_5 = PySequence_ITEM(__pyx_t_43, __pyx_t_3); __pyx_t_3++; if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 162, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_5);
        #endif
      } else {
        if (__pyx_t_3 >= PyTuple_GET_SIZE(__pyx_t_43)) break;
        #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
        __pyx_t_5 = PyTuple_GET_ITEM(__pyx_t_43, __pyx_t_3); __Pyx_INCREF(__pyx_t_5); __pyx_t_3++; if (unlikely(0 < 0)) __PYX_ERR(0, 162, __pyx_L1_error)
        #else
        __pyx_t_5 = PySequence_ITEM(__pyx_t_43, __pyx_t_3); __pyx_t_3++; if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 162, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_5);
        #endif
      }
    } else {
      __pyx_t_5 = __pyx_t_4(__pyx_t_43);
      if (unlikely(!__pyx_t_5)) {
        PyObject* exc_type = PyErr_Occurred();
        if (exc_type) {
          if (likely(exc_type == PyExc_StopIteration || PyErr_GivenExceptionMatches(exc_type, PyExc_StopIteration))) PyErr_Clear();
          else __PYX_ERR(0, 162, __pyx_L1_error)
        }
        break;
      }
      __Pyx_GOTREF(__pyx_t_5);
    }
    __Pyx_XDECREF_SET(__pyx_v_i, __pyx_t_5);
    __pyx_t_5 = 0;
    if (unlikely(__Pyx_ListComp_Append(__pyx_t_2, (PyObject*)__pyx_v_i))) __PYX_ERR(0, 162, __pyx_L1_error)
  }
  __Pyx_DECREF(__pyx_t_43); __pyx_t_43 = 0;
  __pyx_v_keep = ((PyObject*)__pyx_t_2);
  __pyx_t_2 = 0;

  /* "nms/cpu_nms.pyx":163
 * 
 *     keep = [i for i in range(N)]
 *     return keep             # <<<<<<<<<<<<<<
 */
  __Pyx_XDECREF(__pyx_r);
  __Pyx_INCREF(__pyx_v_keep);
  __pyx_r = __pyx_v_keep;
  goto __pyx_L0;

  /* "nms/cpu_nms.pyx":70
 *     return keep
 * 
 * def cpu_soft_nms(np.ndarray[float, ndim=2] boxes, float sigma=0.5, float Nt=0.3, float threshold=0.001, unsigned int method=0):             # <<<<<<<<<<<<<<
 *     cdef unsigned int N = boxes.shape[0]
 *     cdef float iw, ih, box_area
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_43);
  __Pyx_XDECREF(__pyx_t_44);
  __Pyx_XDECREF(__pyx_t_45);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_boxes.rcbuffer->pybuffer);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("nms.cpu_nms.cpu_soft_nms", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_boxes.rcbuffer->pybuffer);
  __pyx_L2:;
  __Pyx_XDECREF(__pyx_v_i);
  __Pyx_XDECREF(__pyx_v_s);
  __Pyx_XDECREF(__pyx_v_keep);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":197
 *         # experimental exception made for __getbuffer__ and __releasebuffer__
 *         # -- the details of this may change.
 *         def __getbuffer__(ndarray self, Py_buffer* info, int flags):             # <<<<<<<<<<<<<<
 *             # This implementation of getbuffer is geared towards Cython
 *             # requirements, and does not yet fullfill the PEP.
 */

/* Python wrapper */
static CYTHON_UNUSED int __pyx_pw_5numpy_7ndarray_1__getbuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info, int __pyx_v_flags); /*proto*/
static CYTHON_UNUSED int __pyx_pw_5numpy_7ndarray_1__getbuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info, int __pyx_v_flags) {
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__getbuffer__ (wrapper)", 0);
  __pyx_r = __pyx_pf_5numpy_7ndarray___getbuffer__(((PyArrayObject *)__pyx_v_self), ((Py_buffer *)__pyx_v_info), ((int)__pyx_v_flags));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static int __pyx_pf_5numpy_7ndarray___getbuffer__(PyArrayObject *__pyx_v_self, Py_buffer *__pyx_v_info, int __pyx_v_flags) {
  int __pyx_v_copy_shape;
  int __pyx_v_i;
  int __pyx_v_ndim;
  int __pyx_v_endian_detector;
  int __pyx_v_little_endian;
  int __pyx_v_t;
  char *__pyx_v_f;
  PyArray_Descr *__pyx_v_descr = 0;
  int __pyx_v_offset;
  int __pyx_v_hasfields;
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  int __pyx_t_2;
  PyObject *__pyx_t_3 = NULL;
  int __pyx_t_4;
  int __pyx_t_5;
  PyObject *__pyx_t_6 = NULL;
  char *__pyx_t_7;
  __Pyx_RefNannySetupContext("__getbuffer__", 0);
  if (__pyx_v_info != NULL) {
    __pyx_v_info->obj = Py_None; __Pyx_INCREF(Py_None);
    __Pyx_GIVEREF(__pyx_v_info->obj);
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":203
 *             # of flags
 * 
 *             if info == NULL: return             # <<<<<<<<<<<<<<
 * 
 *             cdef int copy_shape, i, ndim
 */
  __pyx_t_1 = ((__pyx_v_info == NULL) != 0);
  if (__pyx_t_1) {
    __pyx_r = 0;
    goto __pyx_L0;
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":206
 * 
 *             cdef int copy_shape, i, ndim
 *             cdef int endian_detector = 1             # <<<<<<<<<<<<<<
 *             cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)
 * 
 */
  __pyx_v_endian_detector = 1;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":207
 *             cdef int copy_shape, i, ndim
 *             cdef int endian_detector = 1
 *             cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)             # <<<<<<<<<<<<<<
 * 
 *             ndim = PyArray_NDIM(self)
 */
  __pyx_v_little_endian = ((((char *)(&__pyx_v_endian_detector))[0]) != 0);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":209
 *             cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)
 * 
 *             ndim = PyArray_NDIM(self)             # <<<<<<<<<<<<<<
 * 
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 */
  __pyx_v_ndim = PyArray_NDIM(__pyx_v_self);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":211
 *             ndim = PyArray_NDIM(self)
 * 
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):             # <<<<<<<<<<<<<<
 *                 copy_shape = 1
 *             else:
 */
  __pyx_t_1 = (((sizeof(npy_intp)) != (sizeof(Py_ssize_t))) != 0);
  if (__pyx_t_1) {

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":212
 * 
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 *                 copy_shape = 1             # <<<<<<<<<<<<<<
 *             else:
 *                 copy_shape = 0
 */
    __pyx_v_copy_shape = 1;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":211
 *             ndim = PyArray_NDIM(self)
 * 
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):             # <<<<<<<<<<<<<<
 *                 copy_shape = 1
 *             else:
 */
    goto __pyx_L4;
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":214
 *                 copy_shape = 1
 *             else:
 *                 copy_shape = 0             # <<<<<<<<<<<<<<
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 */
  /*else*/ {
    __pyx_v_copy_shape = 0;
  }
  __pyx_L4:;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":216
 *                 copy_shape = 0
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")
 */
  __pyx_t_2 = (((__pyx_v_flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS) != 0);
  if (__pyx_t_2) {
  } else {
    __pyx_t_1 = __pyx_t_2;
    goto __pyx_L6_bool_binop_done;
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":217
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):             # <<<<<<<<<<<<<<
 *                 raise ValueError(u"ndarray is not C contiguous")
 * 
 */
  __pyx_t_2 = ((!(PyArray_CHKFLAGS(__pyx_v_self, NPY_C_CONTIGUOUS) != 0)) != 0);
  __pyx_t_1 = __pyx_t_2;
  __pyx_L6_bool_binop_done:;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":216
 *                 copy_shape = 0
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")
 */
  if (__pyx_t_1) {

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":218
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")             # <<<<<<<<<<<<<<
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 */
    __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_tuple__12, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 218, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __Pyx_Raise(__pyx_t_3, 0, 0, 0);
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __PYX_ERR(1, 218, __pyx_L1_error)

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":216
 *                 copy_shape = 0
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")
 */
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":220
 *                 raise ValueError(u"ndarray is not C contiguous")
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 */
  __pyx_t_2 = (((__pyx_v_flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS) != 0);
  if (__pyx_t_2) {
  } else {
    __pyx_t_1 = __pyx_t_2;
    goto __pyx_L9_bool_binop_done;
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":221
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):             # <<<<<<<<<<<<<<
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 * 
 */
  __pyx_t_2 = ((!(PyArray_CHKFLAGS(__pyx_v_self, NPY_F_CONTIGUOUS) != 0)) != 0);
  __pyx_t_1 = __pyx_t_2;
  __pyx_L9_bool_binop_done:;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":220
 *                 raise ValueError(u"ndarray is not C contiguous")
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 */
  if (__pyx_t_1) {

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":222
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")             # <<<<<<<<<<<<<<
 * 
 *             info.buf = PyArray_DATA(self)
 */
    __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_tuple__13, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 222, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __Pyx_Raise(__pyx_t_3, 0, 0, 0);
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __PYX_ERR(1, 222, __pyx_L1_error)

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":220
 *                 raise ValueError(u"ndarray is not C contiguous")
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 */
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":224
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 * 
 *             info.buf = PyArray_DATA(self)             # <<<<<<<<<<<<<<
 *             info.ndim = ndim
 *             if copy_shape:
 */
  __pyx_v_info->buf = PyArray_DATA(__pyx_v_self);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":225
 * 
 *             info.buf = PyArray_DATA(self)
 *             info.ndim = ndim             # <<<<<<<<<<<<<<
 *             if copy_shape:
 *                 # Allocate new buffer for strides and shape info.
 */
  __pyx_v_info->ndim = __pyx_v_ndim;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":226
 *             info.buf = PyArray_DATA(self)
 *             info.ndim = ndim
 *             if copy_shape:             # <<<<<<<<<<<<<<
 *                 # Allocate new buffer for strides and shape info.
 *                 # This is allocated as one block, strides first.
 */
  __pyx_t_1 = (__pyx_v_copy_shape != 0);
  if (__pyx_t_1) {

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":229
 *                 # Allocate new buffer for strides and shape info.
 *                 # This is allocated as one block, strides first.
 *                 info.strides = <Py_ssize_t*>stdlib.malloc(sizeof(Py_ssize_t) * <size_t>ndim * 2)             # <<<<<<<<<<<<<<
 *                 info.shape = info.strides + ndim
 *                 for i in range(ndim):
 */
    __pyx_v_info->strides = ((Py_ssize_t *)malloc((((sizeof(Py_ssize_t)) * ((size_t)__pyx_v_ndim)) * 2)));

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":230
 *                 # This is allocated as one block, strides first.
 *                 info.strides = <Py_ssize_t*>stdlib.malloc(sizeof(Py_ssize_t) * <size_t>ndim * 2)
 *                 info.shape = info.strides + ndim             # <<<<<<<<<<<<<<
 *                 for i in range(ndim):
 *                     info.strides[i] = PyArray_STRIDES(self)[i]
 */
    __pyx_v_info->shape = (__pyx_v_info->strides + __pyx_v_ndim);

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":231
 *                 info.strides = <Py_ssize_t*>stdlib.malloc(sizeof(Py_ssize_t) * <size_t>ndim * 2)
 *                 info.shape = info.strides + ndim
 *                 for i in range(ndim):             # <<<<<<<<<<<<<<
 *                     info.strides[i] = PyArray_STRIDES(self)[i]
 *                     info.shape[i] = PyArray_DIMS(self)[i]
 */
    __pyx_t_4 = __pyx_v_ndim;
    for (__pyx_t_5 = 0; __pyx_t_5 < __pyx_t_4; __pyx_t_5+=1) {
      __pyx_v_i = __pyx_t_5;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":232
 *                 info.shape = info.strides + ndim
 *                 for i in range(ndim):
 *                     info.strides[i] = PyArray_STRIDES(self)[i]             # <<<<<<<<<<<<<<
 *                     info.shape[i] = PyArray_DIMS(self)[i]
 *             else:
 */
      (__pyx_v_info->strides[__pyx_v_i]) = (PyArray_STRIDES(__pyx_v_self)[__pyx_v_i]);

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":233
 *                 for i in range(ndim):
 *                     info.strides[i] = PyArray_STRIDES(self)[i]
 *                     info.shape[i] = PyArray_DIMS(self)[i]             # <<<<<<<<<<<<<<
 *             else:
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)
 */
      (__pyx_v_info->shape[__pyx_v_i]) = (PyArray_DIMS(__pyx_v_self)[__pyx_v_i]);
    }

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":226
 *             info.buf = PyArray_DATA(self)
 *             info.ndim = ndim
 *             if copy_shape:             # <<<<<<<<<<<<<<
 *                 # Allocate new buffer for strides and shape info.
 *                 # This is allocated as one block, strides first.
 */
    goto __pyx_L11;
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":235
 *                     info.shape[i] = PyArray_DIMS(self)[i]
 *             else:
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)             # <<<<<<<<<<<<<<
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)
 *             info.suboffsets = NULL
 */
  /*else*/ {
    __pyx_v_info->strides = ((Py_ssize_t *)PyArray_STRIDES(__pyx_v_self));

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":236
 *             else:
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)             # <<<<<<<<<<<<<<
 *             info.suboffsets = NULL
 *             info.itemsize = PyArray_ITEMSIZE(self)
 */
    __pyx_v_info->shape = ((Py_ssize_t *)PyArray_DIMS(__pyx_v_self));
  }
  __pyx_L11:;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":237
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)
 *             info.suboffsets = NULL             # <<<<<<<<<<<<<<
 *             info.itemsize = PyArray_ITEMSIZE(self)
 *             info.readonly = not PyArray_ISWRITEABLE(self)
 */
  __pyx_v_info->suboffsets = NULL;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":238
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)
 *             info.suboffsets = NULL
 *             info.itemsize = PyArray_ITEMSIZE(self)             # <<<<<<<<<<<<<<
 *             info.readonly = not PyArray_ISWRITEABLE(self)
 * 
 */
  __pyx_v_info->itemsize = PyArray_ITEMSIZE(__pyx_v_self);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":239
 *             info.suboffsets = NULL
 *             info.itemsize = PyArray_ITEMSIZE(self)
 *             info.readonly = not PyArray_ISWRITEABLE(self)             # <<<<<<<<<<<<<<
 * 
 *             cdef int t
 */
  __pyx_v_info->readonly = (!(PyArray_ISWRITEABLE(__pyx_v_self) != 0));

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":242
 * 
 *             cdef int t
 *             cdef char* f = NULL             # <<<<<<<<<<<<<<
 *             cdef dtype descr = self.descr
 *             cdef int offset
 */
  __pyx_v_f = NULL;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":243
 *             cdef int t
 *             cdef char* f = NULL
 *             cdef dtype descr = self.descr             # <<<<<<<<<<<<<<
 *             cdef int offset
 * 
 */
  __pyx_t_3 = ((PyObject *)__pyx_v_self->descr);
  __Pyx_INCREF(__pyx_t_3);
  __pyx_v_descr = ((PyArray_Descr *)__pyx_t_3);
  __pyx_t_3 = 0;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":246
 *             cdef int offset
 * 
 *             cdef bint hasfields = PyDataType_HASFIELDS(descr)             # <<<<<<<<<<<<<<
 * 
 *             if not hasfields and not copy_shape:
 */
  __pyx_v_hasfields = PyDataType_HASFIELDS(__pyx_v_descr);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":248
 *             cdef bint hasfields = PyDataType_HASFIELDS(descr)
 * 
 *             if not hasfields and not copy_shape:             # <<<<<<<<<<<<<<
 *                 # do not call releasebuffer
 *                 info.obj = None
 */
  __pyx_t_2 = ((!(__pyx_v_hasfields != 0)) != 0);
  if (__pyx_t_2) {
  } else {
    __pyx_t_1 = __pyx_t_2;
    goto __pyx_L15_bool_binop_done;
  }
  __pyx_t_2 = ((!(__pyx_v_copy_shape != 0)) != 0);
  __pyx_t_1 = __pyx_t_2;
  __pyx_L15_bool_binop_done:;
  if (__pyx_t_1) {

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":250
 *             if not hasfields and not copy_shape:
 *                 # do not call releasebuffer
 *                 info.obj = None             # <<<<<<<<<<<<<<
 *             else:
 *                 # need to call releasebuffer
 */
    __Pyx_INCREF(Py_None);
    __Pyx_GIVEREF(Py_None);
    __Pyx_GOTREF(__pyx_v_info->obj);
    __Pyx_DECREF(__pyx_v_info->obj);
    __pyx_v_info->obj = Py_None;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":248
 *             cdef bint hasfields = PyDataType_HASFIELDS(descr)
 * 
 *             if not hasfields and not copy_shape:             # <<<<<<<<<<<<<<
 *                 # do not call releasebuffer
 *                 info.obj = None
 */
    goto __pyx_L14;
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":253
 *             else:
 *                 # need to call releasebuffer
 *                 info.obj = self             # <<<<<<<<<<<<<<
 * 
 *             if not hasfields:
 */
  /*else*/ {
    __Pyx_INCREF(((PyObject *)__pyx_v_self));
    __Pyx_GIVEREF(((PyObject *)__pyx_v_self));
    __Pyx_GOTREF(__pyx_v_info->obj);
    __Pyx_DECREF(__pyx_v_info->obj);
    __pyx_v_info->obj = ((PyObject *)__pyx_v_self);
  }
  __pyx_L14:;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":255
 *                 info.obj = self
 * 
 *             if not hasfields:             # <<<<<<<<<<<<<<
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or
 */
  __pyx_t_1 = ((!(__pyx_v_hasfields != 0)) != 0);
  if (__pyx_t_1) {

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":256
 * 
 *             if not hasfields:
 *                 t = descr.type_num             # <<<<<<<<<<<<<<
 *                 if ((descr.byteorder == c'>' and little_endian) or
 *                     (descr.byteorder == c'<' and not little_endian)):
 */
    __pyx_t_4 = __pyx_v_descr->type_num;
    __pyx_v_t = __pyx_t_4;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":257
 *             if not hasfields:
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")
 */
    __pyx_t_2 = ((__pyx_v_descr->byteorder == '>') != 0);
    if (!__pyx_t_2) {
      goto __pyx_L20_next_or;
    } else {
    }
    __pyx_t_2 = (__pyx_v_little_endian != 0);
    if (!__pyx_t_2) {
    } else {
      __pyx_t_1 = __pyx_t_2;
      goto __pyx_L19_bool_binop_done;
    }
    __pyx_L20_next_or:;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":258
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or
 *                     (descr.byteorder == c'<' and not little_endian)):             # <<<<<<<<<<<<<<
 *                     raise ValueError(u"Non-native byte order not supported")
 *                 if   t == NPY_BYTE:        f = "b"
 */
    __pyx_t_2 = ((__pyx_v_descr->byteorder == '<') != 0);
    if (__pyx_t_2) {
    } else {
      __pyx_t_1 = __pyx_t_2;
      goto __pyx_L19_bool_binop_done;
    }
    __pyx_t_2 = ((!(__pyx_v_little_endian != 0)) != 0);
    __pyx_t_1 = __pyx_t_2;
    __pyx_L19_bool_binop_done:;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":257
 *             if not hasfields:
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")
 */
    if (__pyx_t_1) {

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":259
 *                 if ((descr.byteorder == c'>' and little_endian) or
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"
 */
      __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_tuple__14, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 259, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_Raise(__pyx_t_3, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __PYX_ERR(1, 259, __pyx_L1_error)

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":257
 *             if not hasfields:
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")
 */
    }

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":260
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")
 *                 if   t == NPY_BYTE:        f = "b"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_UBYTE:       f = "B"
 *                 elif t == NPY_SHORT:       f = "h"
 */
    switch (__pyx_v_t) {
      case NPY_BYTE:
      __pyx_v_f = ((char *)"b");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":261
 *                     raise ValueError(u"Non-native byte order not supported")
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_SHORT:       f = "h"
 *                 elif t == NPY_USHORT:      f = "H"
 */
      case NPY_UBYTE:
      __pyx_v_f = ((char *)"B");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":262
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"
 *                 elif t == NPY_SHORT:       f = "h"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_USHORT:      f = "H"
 *                 elif t == NPY_INT:         f = "i"
 */
      case NPY_SHORT:
      __pyx_v_f = ((char *)"h");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":263
 *                 elif t == NPY_UBYTE:       f = "B"
 *                 elif t == NPY_SHORT:       f = "h"
 *                 elif t == NPY_USHORT:      f = "H"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_INT:         f = "i"
 *                 elif t == NPY_UINT:        f = "I"
 */
      case NPY_USHORT:
      __pyx_v_f = ((char *)"H");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":264
 *                 elif t == NPY_SHORT:       f = "h"
 *                 elif t == NPY_USHORT:      f = "H"
 *                 elif t == NPY_INT:         f = "i"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_UINT:        f = "I"
 *                 elif t == NPY_LONG:        f = "l"
 */
      case NPY_INT:
      __pyx_v_f = ((char *)"i");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":265
 *                 elif t == NPY_USHORT:      f = "H"
 *                 elif t == NPY_INT:         f = "i"
 *                 elif t == NPY_UINT:        f = "I"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_LONG:        f = "l"
 *                 elif t == NPY_ULONG:       f = "L"
 */
      case NPY_UINT:
      __pyx_v_f = ((char *)"I");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":266
 *                 elif t == NPY_INT:         f = "i"
 *                 elif t == NPY_UINT:        f = "I"
 *                 elif t == NPY_LONG:        f = "l"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_ULONG:       f = "L"
 *                 elif t == NPY_LONGLONG:    f = "q"
 */
      case NPY_LONG:
      __pyx_v_f = ((char *)"l");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":267
 *                 elif t == NPY_UINT:        f = "I"
 *                 elif t == NPY_LONG:        f = "l"
 *                 elif t == NPY_ULONG:       f = "L"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_LONGLONG:    f = "q"
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 */
      case NPY_ULONG:
      __pyx_v_f = ((char *)"L");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":268
 *                 elif t == NPY_LONG:        f = "l"
 *                 elif t == NPY_ULONG:       f = "L"
 *                 elif t == NPY_LONGLONG:    f = "q"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 *                 elif t == NPY_FLOAT:       f = "f"
 */
      case NPY_LONGLONG:
      __pyx_v_f = ((char *)"q");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":269
 *                 elif t == NPY_ULONG:       f = "L"
 *                 elif t == NPY_LONGLONG:    f = "q"
 *                 elif t == NPY_ULONGLONG:   f = "Q"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_FLOAT:       f = "f"
 *                 elif t == NPY_DOUBLE:      f = "d"
 */
      case NPY_ULONGLONG:
      __pyx_v_f = ((char *)"Q");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":270
 *                 elif t == NPY_LONGLONG:    f = "q"
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 *                 elif t == NPY_FLOAT:       f = "f"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_DOUBLE:      f = "d"
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 */
      case NPY_FLOAT:
      __pyx_v_f = ((char *)"f");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":271
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 *                 elif t == NPY_FLOAT:       f = "f"
 *                 elif t == NPY_DOUBLE:      f = "d"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 */
      case NPY_DOUBLE:
      __pyx_v_f = ((char *)"d");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":272
 *                 elif t == NPY_FLOAT:       f = "f"
 *                 elif t == NPY_DOUBLE:      f = "d"
 *                 elif t == NPY_LONGDOUBLE:  f = "g"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 */
      case NPY_LONGDOUBLE:
      __pyx_v_f = ((char *)"g");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":273
 *                 elif t == NPY_DOUBLE:      f = "d"
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 *                 elif t == NPY_CFLOAT:      f = "Zf"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"
 */
      case NPY_CFLOAT:
      __pyx_v_f = ((char *)"Zf");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":274
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 *                 elif t == NPY_CDOUBLE:     f = "Zd"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"
 *                 elif t == NPY_OBJECT:      f = "O"
 */
      case NPY_CDOUBLE:
      __pyx_v_f = ((char *)"Zd");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":275
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_OBJECT:      f = "O"
 *                 else:
 */
      case NPY_CLONGDOUBLE:
      __pyx_v_f = ((char *)"Zg");
      break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":276
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"
 *                 elif t == NPY_OBJECT:      f = "O"             # <<<<<<<<<<<<<<
 *                 else:
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 */
      case NPY_OBJECT:
      __pyx_v_f = ((char *)"O");
      break;
      default:

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":278
 *                 elif t == NPY_OBJECT:      f = "O"
 *                 else:
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)             # <<<<<<<<<<<<<<
 *                 info.format = f
 *                 return
 */
      __pyx_t_3 = __Pyx_PyInt_From_int(__pyx_v_t); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 278, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_6 = PyUnicode_Format(__pyx_kp_u_unknown_dtype_code_in_numpy_pxd, __pyx_t_3); if (unlikely(!__pyx_t_6)) __PYX_ERR(1, 278, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 278, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_GIVEREF(__pyx_t_6);
      PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_6);
      __pyx_t_6 = 0;
      __pyx_t_6 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_t_3, NULL); if (unlikely(!__pyx_t_6)) __PYX_ERR(1, 278, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __Pyx_Raise(__pyx_t_6, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
      __PYX_ERR(1, 278, __pyx_L1_error)
      break;
    }

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":279
 *                 else:
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 *                 info.format = f             # <<<<<<<<<<<<<<
 *                 return
 *             else:
 */
    __pyx_v_info->format = __pyx_v_f;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":280
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 *                 info.format = f
 *                 return             # <<<<<<<<<<<<<<
 *             else:
 *                 info.format = <char*>stdlib.malloc(_buffer_format_string_len)
 */
    __pyx_r = 0;
    goto __pyx_L0;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":255
 *                 info.obj = self
 * 
 *             if not hasfields:             # <<<<<<<<<<<<<<
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or
 */
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":282
 *                 return
 *             else:
 *                 info.format = <char*>stdlib.malloc(_buffer_format_string_len)             # <<<<<<<<<<<<<<
 *                 info.format[0] = c'^' # Native data types, manual alignment
 *                 offset = 0
 */
  /*else*/ {
    __pyx_v_info->format = ((char *)malloc(0xFF));

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":283
 *             else:
 *                 info.format = <char*>stdlib.malloc(_buffer_format_string_len)
 *                 info.format[0] = c'^' # Native data types, manual alignment             # <<<<<<<<<<<<<<
 *                 offset = 0
 *                 f = _util_dtypestring(descr, info.format + 1,
 */
    (__pyx_v_info->format[0]) = '^';

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":284
 *                 info.format = <char*>stdlib.malloc(_buffer_format_string_len)
 *                 info.format[0] = c'^' # Native data types, manual alignment
 *                 offset = 0             # <<<<<<<<<<<<<<
 *                 f = _util_dtypestring(descr, info.format + 1,
 *                                       info.format + _buffer_format_string_len,
 */
    __pyx_v_offset = 0;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":285
 *                 info.format[0] = c'^' # Native data types, manual alignment
 *                 offset = 0
 *                 f = _util_dtypestring(descr, info.format + 1,             # <<<<<<<<<<<<<<
 *                                       info.format + _buffer_format_string_len,
 *                                       &offset)
 */
    __pyx_t_7 = __pyx_f_5numpy__util_dtypestring(__pyx_v_descr, (__pyx_v_info->format + 1), (__pyx_v_info->format + 0xFF), (&__pyx_v_offset)); if (unlikely(__pyx_t_7 == NULL)) __PYX_ERR(1, 285, __pyx_L1_error)
    __pyx_v_f = __pyx_t_7;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":288
 *                                       info.format + _buffer_format_string_len,
 *                                       &offset)
 *                 f[0] = c'\0' # Terminate format string             # <<<<<<<<<<<<<<
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):
 */
    (__pyx_v_f[0]) = '\x00';
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":197
 *         # experimental exception made for __getbuffer__ and __releasebuffer__
 *         # -- the details of this may change.
 *         def __getbuffer__(ndarray self, Py_buffer* info, int flags):             # <<<<<<<<<<<<<<
 *             # This implementation of getbuffer is geared towards Cython
 *             # requirements, and does not yet fullfill the PEP.
 */

  /* function exit code */
  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_AddTraceback("numpy.ndarray.__getbuffer__", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = -1;
  if (__pyx_v_info != NULL && __pyx_v_info->obj != NULL) {
    __Pyx_GOTREF(__pyx_v_info->obj);
    __Pyx_DECREF(__pyx_v_info->obj); __pyx_v_info->obj = NULL;
  }
  goto __pyx_L2;
  __pyx_L0:;
  if (__pyx_v_info != NULL && __pyx_v_info->obj == Py_None) {
    __Pyx_GOTREF(Py_None);
    __Pyx_DECREF(Py_None); __pyx_v_info->obj = NULL;
  }
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_descr);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":290
 *                 f[0] = c'\0' # Terminate format string
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):             # <<<<<<<<<<<<<<
 *             if PyArray_HASFIELDS(self):
 *                 stdlib.free(info.format)
 */

/* Python wrapper */
static CYTHON_UNUSED void __pyx_pw_5numpy_7ndarray_3__releasebuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info); /*proto*/
static CYTHON_UNUSED void __pyx_pw_5numpy_7ndarray_3__releasebuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info) {
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__releasebuffer__ (wrapper)", 0);
  __pyx_pf_5numpy_7ndarray_2__releasebuffer__(((PyArrayObject *)__pyx_v_self), ((Py_buffer *)__pyx_v_info));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
}

static void __pyx_pf_5numpy_7ndarray_2__releasebuffer__(PyArrayObject *__pyx_v_self, Py_buffer *__pyx_v_info) {
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  __Pyx_RefNannySetupContext("__releasebuffer__", 0);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":291
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):
 *             if PyArray_HASFIELDS(self):             # <<<<<<<<<<<<<<
 *                 stdlib.free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 */
  __pyx_t_1 = (PyArray_HASFIELDS(__pyx_v_self) != 0);
  if (__pyx_t_1) {

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":292
 *         def __releasebuffer__(ndarray self, Py_buffer* info):
 *             if PyArray_HASFIELDS(self):
 *                 stdlib.free(info.format)             # <<<<<<<<<<<<<<
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 *                 stdlib.free(info.strides)
 */
    free(__pyx_v_info->format);

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":291
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):
 *             if PyArray_HASFIELDS(self):             # <<<<<<<<<<<<<<
 *                 stdlib.free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 */
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":293
 *             if PyArray_HASFIELDS(self):
 *                 stdlib.free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):             # <<<<<<<<<<<<<<
 *                 stdlib.free(info.strides)
 *                 # info.shape was stored after info.strides in the same block
 */
  __pyx_t_1 = (((sizeof(npy_intp)) != (sizeof(Py_ssize_t))) != 0);
  if (__pyx_t_1) {

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":294
 *                 stdlib.free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 *                 stdlib.free(info.strides)             # <<<<<<<<<<<<<<
 *                 # info.shape was stored after info.strides in the same block
 * 
 */
    free(__pyx_v_info->strides);

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":293
 *             if PyArray_HASFIELDS(self):
 *                 stdlib.free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):             # <<<<<<<<<<<<<<
 *                 stdlib.free(info.strides)
 *                 # info.shape was stored after info.strides in the same block
 */
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":290
 *                 f[0] = c'\0' # Terminate format string
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):             # <<<<<<<<<<<<<<
 *             if PyArray_HASFIELDS(self):
 *                 stdlib.free(info.format)
 */

  /* function exit code */
  __Pyx_RefNannyFinishContext();
}

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":770
 * ctypedef npy_cdouble     complex_t
 * 
 * cdef inline object PyArray_MultiIterNew1(a):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(1, <void*>a)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew1(PyObject *__pyx_v_a) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew1", 0);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":771
 * 
 * cdef inline object PyArray_MultiIterNew1(a):
 *     return PyArray_MultiIterNew(1, <void*>a)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew2(a, b):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(1, ((void *)__pyx_v_a)); if (unlikely(!__pyx_t_1)) __PYX_ERR(1, 771, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":770
 * ctypedef npy_cdouble     complex_t
 * 
 * cdef inline object PyArray_MultiIterNew1(a):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(1, <void*>a)
 * 
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew1", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":773
 *     return PyArray_MultiIterNew(1, <void*>a)
 * 
 * cdef inline object PyArray_MultiIterNew2(a, b):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew2(PyObject *__pyx_v_a, PyObject *__pyx_v_b) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew2", 0);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":774
 * 
 * cdef inline object PyArray_MultiIterNew2(a, b):
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew3(a, b, c):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(2, ((void *)__pyx_v_a), ((void *)__pyx_v_b)); if (unlikely(!__pyx_t_1)) __PYX_ERR(1, 774, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":773
 *     return PyArray_MultiIterNew(1, <void*>a)
 * 
 * cdef inline object PyArray_MultiIterNew2(a, b):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)
 * 
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew2", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":776
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)
 * 
 * cdef inline object PyArray_MultiIterNew3(a, b, c):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew3(PyObject *__pyx_v_a, PyObject *__pyx_v_b, PyObject *__pyx_v_c) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew3", 0);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":777
 * 
 * cdef inline object PyArray_MultiIterNew3(a, b, c):
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew4(a, b, c, d):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(3, ((void *)__pyx_v_a), ((void *)__pyx_v_b), ((void *)__pyx_v_c)); if (unlikely(!__pyx_t_1)) __PYX_ERR(1, 777, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":776
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)
 * 
 * cdef inline object PyArray_MultiIterNew3(a, b, c):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)
 * 
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew3", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":779
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)
 * 
 * cdef inline object PyArray_MultiIterNew4(a, b, c, d):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew4(PyObject *__pyx_v_a, PyObject *__pyx_v_b, PyObject *__pyx_v_c, PyObject *__pyx_v_d) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew4", 0);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":780
 * 
 * cdef inline object PyArray_MultiIterNew4(a, b, c, d):
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(4, ((void *)__pyx_v_a), ((void *)__pyx_v_b), ((void *)__pyx_v_c), ((void *)__pyx_v_d)); if (unlikely(!__pyx_t_1)) __PYX_ERR(1, 780, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":779
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)
 * 
 * cdef inline object PyArray_MultiIterNew4(a, b, c, d):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)
 * 
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew4", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":782
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)
 * 
 * cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew5(PyObject *__pyx_v_a, PyObject *__pyx_v_b, PyObject *__pyx_v_c, PyObject *__pyx_v_d, PyObject *__pyx_v_e) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew5", 0);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":783
 * 
 * cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)             # <<<<<<<<<<<<<<
 * 
 * cdef inline char* _util_dtypestring(dtype descr, char* f, char* end, int* offset) except NULL:
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(5, ((void *)__pyx_v_a), ((void *)__pyx_v_b), ((void *)__pyx_v_c), ((void *)__pyx_v_d), ((void *)__pyx_v_e)); if (unlikely(!__pyx_t_1)) __PYX_ERR(1, 783, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":782
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)
 * 
 * cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)
 * 
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew5", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":785
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)
 * 
 * cdef inline char* _util_dtypestring(dtype descr, char* f, char* end, int* offset) except NULL:             # <<<<<<<<<<<<<<
 *     # Recursive utility function used in __getbuffer__ to get format
 *     # string. The new location in the format string is returned.
 */

static CYTHON_INLINE char *__pyx_f_5numpy__util_dtypestring(PyArray_Descr *__pyx_v_descr, char *__pyx_v_f, char *__pyx_v_end, int *__pyx_v_offset) {
  PyArray_Descr *__pyx_v_child = 0;
  int __pyx_v_endian_detector;
  int __pyx_v_little_endian;
  PyObject *__pyx_v_fields = 0;
  PyObject *__pyx_v_childname = NULL;
  PyObject *__pyx_v_new_offset = NULL;
  PyObject *__pyx_v_t = NULL;
  char *__pyx_r;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  Py_ssize_t __pyx_t_2;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  int __pyx_t_5;
  int __pyx_t_6;
  int __pyx_t_7;
  long __pyx_t_8;
  char *__pyx_t_9;
  __Pyx_RefNannySetupContext("_util_dtypestring", 0);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":790
 * 
 *     cdef dtype child
 *     cdef int endian_detector = 1             # <<<<<<<<<<<<<<
 *     cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)
 *     cdef tuple fields
 */
  __pyx_v_endian_detector = 1;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":791
 *     cdef dtype child
 *     cdef int endian_detector = 1
 *     cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)             # <<<<<<<<<<<<<<
 *     cdef tuple fields
 * 
 */
  __pyx_v_little_endian = ((((char *)(&__pyx_v_endian_detector))[0]) != 0);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":794
 *     cdef tuple fields
 * 
 *     for childname in descr.names:             # <<<<<<<<<<<<<<
 *         fields = descr.fields[childname]
 *         child, new_offset = fields
 */
  if (unlikely(__pyx_v_descr->names == Py_None)) {
    PyErr_SetString(PyExc_TypeError, "'NoneType' object is not iterable");
    __PYX_ERR(1, 794, __pyx_L1_error)
  }
  __pyx_t_1 = __pyx_v_descr->names; __Pyx_INCREF(__pyx_t_1); __pyx_t_2 = 0;
  for (;;) {
    if (__pyx_t_2 >= PyTuple_GET_SIZE(__pyx_t_1)) break;
    #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
    __pyx_t_3 = PyTuple_GET_ITEM(__pyx_t_1, __pyx_t_2); __Pyx_INCREF(__pyx_t_3); __pyx_t_2++; if (unlikely(0 < 0)) __PYX_ERR(1, 794, __pyx_L1_error)
    #else
    __pyx_t_3 = PySequence_ITEM(__pyx_t_1, __pyx_t_2); __pyx_t_2++; if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 794, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    #endif
    __Pyx_XDECREF_SET(__pyx_v_childname, __pyx_t_3);
    __pyx_t_3 = 0;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":795
 * 
 *     for childname in descr.names:
 *         fields = descr.fields[childname]             # <<<<<<<<<<<<<<
 *         child, new_offset = fields
 * 
 */
    if (unlikely(__pyx_v_descr->fields == Py_None)) {
      PyErr_SetString(PyExc_TypeError, "'NoneType' object is not subscriptable");
      __PYX_ERR(1, 795, __pyx_L1_error)
    }
    __pyx_t_3 = __Pyx_PyDict_GetItem(__pyx_v_descr->fields, __pyx_v_childname); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 795, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    if (!(likely(PyTuple_CheckExact(__pyx_t_3))||((__pyx_t_3) == Py_None)||(PyErr_Format(PyExc_TypeError, "Expected %.16s, got %.200s", "tuple", Py_TYPE(__pyx_t_3)->tp_name), 0))) __PYX_ERR(1, 795, __pyx_L1_error)
    __Pyx_XDECREF_SET(__pyx_v_fields, ((PyObject*)__pyx_t_3));
    __pyx_t_3 = 0;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":796
 *     for childname in descr.names:
 *         fields = descr.fields[childname]
 *         child, new_offset = fields             # <<<<<<<<<<<<<<
 * 
 *         if (end - f) - <int>(new_offset - offset[0]) < 15:
 */
    if (likely(__pyx_v_fields != Py_None)) {
      PyObject* sequence = __pyx_v_fields;
      #if !CYTHON_COMPILING_IN_PYPY
      Py_ssize_t size = Py_SIZE(sequence);
      #else
      Py_ssize_t size = PySequence_Size(sequence);
      #endif
      if (unlikely(size != 2)) {
        if (size > 2) __Pyx_RaiseTooManyValuesError(2);
        else if (size >= 0) __Pyx_RaiseNeedMoreValuesError(size);
        __PYX_ERR(1, 796, __pyx_L1_error)
      }
      #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
      __pyx_t_3 = PyTuple_GET_ITEM(sequence, 0); 
      __pyx_t_4 = PyTuple_GET_ITEM(sequence, 1); 
      __Pyx_INCREF(__pyx_t_3);
      __Pyx_INCREF(__pyx_t_4);
      #else
      __pyx_t_3 = PySequence_ITEM(sequence, 0); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 796, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PySequence_ITEM(sequence, 1); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 796, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      #endif
    } else {
      __Pyx_RaiseNoneNotIterableError(); __PYX_ERR(1, 796, __pyx_L1_error)
    }
    if (!(likely(((__pyx_t_3) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_3, __pyx_ptype_5numpy_dtype))))) __PYX_ERR(1, 796, __pyx_L1_error)
    __Pyx_XDECREF_SET(__pyx_v_child, ((PyArray_Descr *)__pyx_t_3));
    __pyx_t_3 = 0;
    __Pyx_XDECREF_SET(__pyx_v_new_offset, __pyx_t_4);
    __pyx_t_4 = 0;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":798
 *         child, new_offset = fields
 * 
 *         if (end - f) - <int>(new_offset - offset[0]) < 15:             # <<<<<<<<<<<<<<
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 */
    __pyx_t_4 = __Pyx_PyInt_From_int((__pyx_v_offset[0])); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 798, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_4);
    __pyx_t_3 = PyNumber_Subtract(__pyx_v_new_offset, __pyx_t_4); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 798, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    __pyx_t_5 = __Pyx_PyInt_As_int(__pyx_t_3); if (unlikely((__pyx_t_5 == (int)-1) && PyErr_Occurred())) __PYX_ERR(1, 798, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_6 = ((((__pyx_v_end - __pyx_v_f) - ((int)__pyx_t_5)) < 15) != 0);
    if (__pyx_t_6) {

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":799
 * 
 *         if (end - f) - <int>(new_offset - offset[0]) < 15:
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")             # <<<<<<<<<<<<<<
 * 
 *         if ((child.byteorder == c'>' and little_endian) or
 */
      __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_RuntimeError, __pyx_tuple__15, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 799, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_Raise(__pyx_t_3, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __PYX_ERR(1, 799, __pyx_L1_error)

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":798
 *         child, new_offset = fields
 * 
 *         if (end - f) - <int>(new_offset - offset[0]) < 15:             # <<<<<<<<<<<<<<
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 */
    }

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":801
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 *         if ((child.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *             (child.byteorder == c'<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")
 */
    __pyx_t_7 = ((__pyx_v_child->byteorder == '>') != 0);
    if (!__pyx_t_7) {
      goto __pyx_L8_next_or;
    } else {
    }
    __pyx_t_7 = (__pyx_v_little_endian != 0);
    if (!__pyx_t_7) {
    } else {
      __pyx_t_6 = __pyx_t_7;
      goto __pyx_L7_bool_binop_done;
    }
    __pyx_L8_next_or:;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":802
 * 
 *         if ((child.byteorder == c'>' and little_endian) or
 *             (child.byteorder == c'<' and not little_endian)):             # <<<<<<<<<<<<<<
 *             raise ValueError(u"Non-native byte order not supported")
 *             # One could encode it in the format string and have Cython
 */
    __pyx_t_7 = ((__pyx_v_child->byteorder == '<') != 0);
    if (__pyx_t_7) {
    } else {
      __pyx_t_6 = __pyx_t_7;
      goto __pyx_L7_bool_binop_done;
    }
    __pyx_t_7 = ((!(__pyx_v_little_endian != 0)) != 0);
    __pyx_t_6 = __pyx_t_7;
    __pyx_L7_bool_binop_done:;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":801
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 *         if ((child.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *             (child.byteorder == c'<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")
 */
    if (__pyx_t_6) {

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":803
 *         if ((child.byteorder == c'>' and little_endian) or
 *             (child.byteorder == c'<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *             # One could encode it in the format string and have Cython
 *             # complain instead, BUT: < and > in format strings also imply
 */
      __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_tuple__16, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 803, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_Raise(__pyx_t_3, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __PYX_ERR(1, 803, __pyx_L1_error)

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":801
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 *         if ((child.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *             (child.byteorder == c'<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")
 */
    }

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":813
 * 
 *         # Output padding bytes
 *         while offset[0] < new_offset:             # <<<<<<<<<<<<<<
 *             f[0] = 120 # "x"; pad byte
 *             f += 1
 */
    while (1) {
      __pyx_t_3 = __Pyx_PyInt_From_int((__pyx_v_offset[0])); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 813, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_t_3, __pyx_v_new_offset, Py_LT); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 813, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 813, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (!__pyx_t_6) break;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":814
 *         # Output padding bytes
 *         while offset[0] < new_offset:
 *             f[0] = 120 # "x"; pad byte             # <<<<<<<<<<<<<<
 *             f += 1
 *             offset[0] += 1
 */
      (__pyx_v_f[0]) = 0x78;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":815
 *         while offset[0] < new_offset:
 *             f[0] = 120 # "x"; pad byte
 *             f += 1             # <<<<<<<<<<<<<<
 *             offset[0] += 1
 * 
 */
      __pyx_v_f = (__pyx_v_f + 1);

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":816
 *             f[0] = 120 # "x"; pad byte
 *             f += 1
 *             offset[0] += 1             # <<<<<<<<<<<<<<
 * 
 *         offset[0] += child.itemsize
 */
      __pyx_t_8 = 0;
      (__pyx_v_offset[__pyx_t_8]) = ((__pyx_v_offset[__pyx_t_8]) + 1);
    }

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":818
 *             offset[0] += 1
 * 
 *         offset[0] += child.itemsize             # <<<<<<<<<<<<<<
 * 
 *         if not PyDataType_HASFIELDS(child):
 */
    __pyx_t_8 = 0;
    (__pyx_v_offset[__pyx_t_8]) = ((__pyx_v_offset[__pyx_t_8]) + __pyx_v_child->elsize);

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":820
 *         offset[0] += child.itemsize
 * 
 *         if not PyDataType_HASFIELDS(child):             # <<<<<<<<<<<<<<
 *             t = child.type_num
 *             if end - f < 5:
 */
    __pyx_t_6 = ((!(PyDataType_HASFIELDS(__pyx_v_child) != 0)) != 0);
    if (__pyx_t_6) {

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":821
 * 
 *         if not PyDataType_HASFIELDS(child):
 *             t = child.type_num             # <<<<<<<<<<<<<<
 *             if end - f < 5:
 *                 raise RuntimeError(u"Format string allocated too short.")
 */
      __pyx_t_4 = __Pyx_PyInt_From_int(__pyx_v_child->type_num); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 821, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __Pyx_XDECREF_SET(__pyx_v_t, __pyx_t_4);
      __pyx_t_4 = 0;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":822
 *         if not PyDataType_HASFIELDS(child):
 *             t = child.type_num
 *             if end - f < 5:             # <<<<<<<<<<<<<<
 *                 raise RuntimeError(u"Format string allocated too short.")
 * 
 */
      __pyx_t_6 = (((__pyx_v_end - __pyx_v_f) < 5) != 0);
      if (__pyx_t_6) {

        /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":823
 *             t = child.type_num
 *             if end - f < 5:
 *                 raise RuntimeError(u"Format string allocated too short.")             # <<<<<<<<<<<<<<
 * 
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 */
        __pyx_t_4 = __Pyx_PyObject_Call(__pyx_builtin_RuntimeError, __pyx_tuple__17, NULL); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 823, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_4);
        __Pyx_Raise(__pyx_t_4, 0, 0, 0);
        __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
        __PYX_ERR(1, 823, __pyx_L1_error)

        /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":822
 *         if not PyDataType_HASFIELDS(child):
 *             t = child.type_num
 *             if end - f < 5:             # <<<<<<<<<<<<<<
 *                 raise RuntimeError(u"Format string allocated too short.")
 * 
 */
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":826
 * 
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 *             if   t == NPY_BYTE:        f[0] =  98 #"b"             # <<<<<<<<<<<<<<
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_BYTE); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 826, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 826, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 826, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 98;
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":827
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 *             if   t == NPY_BYTE:        f[0] =  98 #"b"
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"             # <<<<<<<<<<<<<<
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_UBYTE); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 827, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 827, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 827, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 66;
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":828
 *             if   t == NPY_BYTE:        f[0] =  98 #"b"
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"             # <<<<<<<<<<<<<<
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_SHORT); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 828, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 828, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 828, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x68;
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":829
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"             # <<<<<<<<<<<<<<
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_USHORT); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 829, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 829, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 829, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 72;
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":830
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 *             elif t == NPY_INT:         f[0] = 105 #"i"             # <<<<<<<<<<<<<<
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_INT); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 830, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 830, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 830, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x69;
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":831
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 *             elif t == NPY_UINT:        f[0] =  73 #"I"             # <<<<<<<<<<<<<<
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_UINT); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 831, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 831, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 831, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 73;
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":832
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 *             elif t == NPY_LONG:        f[0] = 108 #"l"             # <<<<<<<<<<<<<<
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_LONG); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 832, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 832, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 832, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x6C;
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":833
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"             # <<<<<<<<<<<<<<
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_ULONG); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 833, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 833, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 833, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 76;
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":834
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"             # <<<<<<<<<<<<<<
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_LONGLONG); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 834, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 834, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 834, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x71;
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":835
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"             # <<<<<<<<<<<<<<
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_ULONGLONG); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 835, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 835, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 835, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 81;
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":836
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"             # <<<<<<<<<<<<<<
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_FLOAT); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 836, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 836, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 836, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x66;
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":837
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"             # <<<<<<<<<<<<<<
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_DOUBLE); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 837, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 837, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 837, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x64;
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":838
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"             # <<<<<<<<<<<<<<
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_LONGDOUBLE); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 838, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 838, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 838, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x67;
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":839
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf             # <<<<<<<<<<<<<<
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_CFLOAT); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 839, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 839, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 839, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 90;
        (__pyx_v_f[1]) = 0x66;
        __pyx_v_f = (__pyx_v_f + 1);
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":840
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd             # <<<<<<<<<<<<<<
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_CDOUBLE); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 840, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 840, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 840, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 90;
        (__pyx_v_f[1]) = 0x64;
        __pyx_v_f = (__pyx_v_f + 1);
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":841
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg             # <<<<<<<<<<<<<<
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"
 *             else:
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_CLONGDOUBLE); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 841, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 841, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 841, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 90;
        (__pyx_v_f[1]) = 0x67;
        __pyx_v_f = (__pyx_v_f + 1);
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":842
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"             # <<<<<<<<<<<<<<
 *             else:
 *                 raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_OBJECT); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 842, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 842, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 842, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 79;
        goto __pyx_L15;
      }

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":844
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"
 *             else:
 *                 raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)             # <<<<<<<<<<<<<<
 *             f += 1
 *         else:
 */
      /*else*/ {
        __pyx_t_3 = PyUnicode_Format(__pyx_kp_u_unknown_dtype_code_in_numpy_pxd, __pyx_v_t); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 844, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_3);
        __pyx_t_4 = PyTuple_New(1); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 844, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_4);
        __Pyx_GIVEREF(__pyx_t_3);
        PyTuple_SET_ITEM(__pyx_t_4, 0, __pyx_t_3);
        __pyx_t_3 = 0;
        __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_t_4, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 844, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_3);
        __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
        __Pyx_Raise(__pyx_t_3, 0, 0, 0);
        __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
        __PYX_ERR(1, 844, __pyx_L1_error)
      }
      __pyx_L15:;

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":845
 *             else:
 *                 raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 *             f += 1             # <<<<<<<<<<<<<<
 *         else:
 *             # Cython ignores struct boundary information ("T{...}"),
 */
      __pyx_v_f = (__pyx_v_f + 1);

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":820
 *         offset[0] += child.itemsize
 * 
 *         if not PyDataType_HASFIELDS(child):             # <<<<<<<<<<<<<<
 *             t = child.type_num
 *             if end - f < 5:
 */
      goto __pyx_L13;
    }

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":849
 *             # Cython ignores struct boundary information ("T{...}"),
 *             # so don't output it
 *             f = _util_dtypestring(child, f, end, offset)             # <<<<<<<<<<<<<<
 *     return f
 * 
 */
    /*else*/ {
      __pyx_t_9 = __pyx_f_5numpy__util_dtypestring(__pyx_v_child, __pyx_v_f, __pyx_v_end, __pyx_v_offset); if (unlikely(__pyx_t_9 == NULL)) __PYX_ERR(1, 849, __pyx_L1_error)
      __pyx_v_f = __pyx_t_9;
    }
    __pyx_L13:;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":794
 *     cdef tuple fields
 * 
 *     for childname in descr.names:             # <<<<<<<<<<<<<<
 *         fields = descr.fields[childname]
 *         child, new_offset = fields
 */
  }
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":850
 *             # so don't output it
 *             f = _util_dtypestring(child, f, end, offset)
 *     return f             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = __pyx_v_f;
  goto __pyx_L0;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":785
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)
 * 
 * cdef inline char* _util_dtypestring(dtype descr, char* f, char* end, int* offset) except NULL:             # <<<<<<<<<<<<<<
 *     # Recursive utility function used in __getbuffer__ to get format
 *     # string. The new location in the format string is returned.
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_AddTraceback("numpy._util_dtypestring", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XDECREF((PyObject *)__pyx_v_child);
  __Pyx_XDECREF(__pyx_v_fields);
  __Pyx_XDECREF(__pyx_v_childname);
  __Pyx_XDECREF(__pyx_v_new_offset);
  __Pyx_XDECREF(__pyx_v_t);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":966
 * 
 * 
 * cdef inline void set_array_base(ndarray arr, object base):             # <<<<<<<<<<<<<<
 *      cdef PyObject* baseptr
 *      if base is None:
 */

static CYTHON_INLINE void __pyx_f_5numpy_set_array_base(PyArrayObject *__pyx_v_arr, PyObject *__pyx_v_base) {
  PyObject *__pyx_v_baseptr;
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  int __pyx_t_2;
  __Pyx_RefNannySetupContext("set_array_base", 0);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":968
 * cdef inline void set_array_base(ndarray arr, object base):
 *      cdef PyObject* baseptr
 *      if base is None:             # <<<<<<<<<<<<<<
 *          baseptr = NULL
 *      else:
 */
  __pyx_t_1 = (__pyx_v_base == Py_None);
  __pyx_t_2 = (__pyx_t_1 != 0);
  if (__pyx_t_2) {

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":969
 *      cdef PyObject* baseptr
 *      if base is None:
 *          baseptr = NULL             # <<<<<<<<<<<<<<
 *      else:
 *          Py_INCREF(base) # important to do this before decref below!
 */
    __pyx_v_baseptr = NULL;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":968
 * cdef inline void set_array_base(ndarray arr, object base):
 *      cdef PyObject* baseptr
 *      if base is None:             # <<<<<<<<<<<<<<
 *          baseptr = NULL
 *      else:
 */
    goto __pyx_L3;
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":971
 *          baseptr = NULL
 *      else:
 *          Py_INCREF(base) # important to do this before decref below!             # <<<<<<<<<<<<<<
 *          baseptr = <PyObject*>base
 *      Py_XDECREF(arr.base)
 */
  /*else*/ {
    Py_INCREF(__pyx_v_base);

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":972
 *      else:
 *          Py_INCREF(base) # important to do this before decref below!
 *          baseptr = <PyObject*>base             # <<<<<<<<<<<<<<
 *      Py_XDECREF(arr.base)
 *      arr.base = baseptr
 */
    __pyx_v_baseptr = ((PyObject *)__pyx_v_base);
  }
  __pyx_L3:;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":973
 *          Py_INCREF(base) # important to do this before decref below!
 *          baseptr = <PyObject*>base
 *      Py_XDECREF(arr.base)             # <<<<<<<<<<<<<<
 *      arr.base = baseptr
 * 
 */
  Py_XDECREF(__pyx_v_arr->base);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":974
 *          baseptr = <PyObject*>base
 *      Py_XDECREF(arr.base)
 *      arr.base = baseptr             # <<<<<<<<<<<<<<
 * 
 * cdef inline object get_array_base(ndarray arr):
 */
  __pyx_v_arr->base = __pyx_v_baseptr;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":966
 * 
 * 
 * cdef inline void set_array_base(ndarray arr, object base):             # <<<<<<<<<<<<<<
 *      cdef PyObject* baseptr
 *      if base is None:
 */

  /* function exit code */
  __Pyx_RefNannyFinishContext();
}

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":976
 *      arr.base = baseptr
 * 
 * cdef inline object get_array_base(ndarray arr):             # <<<<<<<<<<<<<<
 *     if arr.base is NULL:
 *         return None
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_get_array_base(PyArrayObject *__pyx_v_arr) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  __Pyx_RefNannySetupContext("get_array_base", 0);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":977
 * 
 * cdef inline object get_array_base(ndarray arr):
 *     if arr.base is NULL:             # <<<<<<<<<<<<<<
 *         return None
 *     else:
 */
  __pyx_t_1 = ((__pyx_v_arr->base == NULL) != 0);
  if (__pyx_t_1) {

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":978
 * cdef inline object get_array_base(ndarray arr):
 *     if arr.base is NULL:
 *         return None             # <<<<<<<<<<<<<<
 *     else:
 *         return <object>arr.base
 */
    __Pyx_XDECREF(__pyx_r);
    __Pyx_INCREF(Py_None);
    __pyx_r = Py_None;
    goto __pyx_L0;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":977
 * 
 * cdef inline object get_array_base(ndarray arr):
 *     if arr.base is NULL:             # <<<<<<<<<<<<<<
 *         return None
 *     else:
 */
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":980
 *         return None
 *     else:
 *         return <object>arr.base             # <<<<<<<<<<<<<<
 * 
 * 
 */
  /*else*/ {
    __Pyx_XDECREF(__pyx_r);
    __Pyx_INCREF(((PyObject *)__pyx_v_arr->base));
    __pyx_r = ((PyObject *)__pyx_v_arr->base);
    goto __pyx_L0;
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":976
 *      arr.base = baseptr
 * 
 * cdef inline object get_array_base(ndarray arr):             # <<<<<<<<<<<<<<
 *     if arr.base is NULL:
 *         return None
 */

  /* function exit code */
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":985
 * # Versions of the import_* functions which are more suitable for
 * # Cython code.
 * cdef inline int import_array() except -1:             # <<<<<<<<<<<<<<
 *     try:
 *         _import_array()
 */

static CYTHON_INLINE int __pyx_f_5numpy_import_array(void) {
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  int __pyx_t_4;
  PyObject *__pyx_t_5 = NULL;
  PyObject *__pyx_t_6 = NULL;
  PyObject *__pyx_t_7 = NULL;
  PyObject *__pyx_t_8 = NULL;
  __Pyx_RefNannySetupContext("import_array", 0);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":986
 * # Cython code.
 * cdef inline int import_array() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_array()
 *     except Exception:
 */
  {
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ExceptionSave(&__pyx_t_1, &__pyx_t_2, &__pyx_t_3);
    __Pyx_XGOTREF(__pyx_t_1);
    __Pyx_XGOTREF(__pyx_t_2);
    __Pyx_XGOTREF(__pyx_t_3);
    /*try:*/ {

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":987
 * cdef inline int import_array() except -1:
 *     try:
 *         _import_array()             # <<<<<<<<<<<<<<
 *     except Exception:
 *         raise ImportError("numpy.core.multiarray failed to import")
 */
      __pyx_t_4 = _import_array(); if (unlikely(__pyx_t_4 == -1)) __PYX_ERR(1, 987, __pyx_L3_error)

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":986
 * # Cython code.
 * cdef inline int import_array() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_array()
 *     except Exception:
 */
    }
    __Pyx_XDECREF(__pyx_t_1); __pyx_t_1 = 0;
    __Pyx_XDECREF(__pyx_t_2); __pyx_t_2 = 0;
    __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
    goto __pyx_L10_try_end;
    __pyx_L3_error:;
    __Pyx_PyThreadState_assign

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":988
 *     try:
 *         _import_array()
 *     except Exception:             # <<<<<<<<<<<<<<
 *         raise ImportError("numpy.core.multiarray failed to import")
 * 
 */
    __pyx_t_4 = __Pyx_PyErr_ExceptionMatches(((PyObject *)(&((PyTypeObject*)PyExc_Exception)[0])));
    if (__pyx_t_4) {
      __Pyx_AddTraceback("numpy.import_array", __pyx_clineno, __pyx_lineno, __pyx_filename);
      if (__Pyx_GetException(&__pyx_t_5, &__pyx_t_6, &__pyx_t_7) < 0) __PYX_ERR(1, 988, __pyx_L5_except_error)
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_GOTREF(__pyx_t_7);

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":989
 *         _import_array()
 *     except Exception:
 *         raise ImportError("numpy.core.multiarray failed to import")             # <<<<<<<<<<<<<<
 * 
 * cdef inline int import_umath() except -1:
 */
      __pyx_t_8 = __Pyx_PyObject_Call(__pyx_builtin_ImportError, __pyx_tuple__18, NULL); if (unlikely(!__pyx_t_8)) __PYX_ERR(1, 989, __pyx_L5_except_error)
      __Pyx_GOTREF(__pyx_t_8);
      __Pyx_Raise(__pyx_t_8, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
      __PYX_ERR(1, 989, __pyx_L5_except_error)
    }
    goto __pyx_L5_except_error;
    __pyx_L5_except_error:;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":986
 * # Cython code.
 * cdef inline int import_array() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_array()
 *     except Exception:
 */
    __Pyx_PyThreadState_assign
    __Pyx_XGIVEREF(__pyx_t_1);
    __Pyx_XGIVEREF(__pyx_t_2);
    __Pyx_XGIVEREF(__pyx_t_3);
    __Pyx_ExceptionReset(__pyx_t_1, __pyx_t_2, __pyx_t_3);
    goto __pyx_L1_error;
    __pyx_L10_try_end:;
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":985
 * # Versions of the import_* functions which are more suitable for
 * # Cython code.
 * cdef inline int import_array() except -1:             # <<<<<<<<<<<<<<
 *     try:
 *         _import_array()
 */

  /* function exit code */
  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_XDECREF(__pyx_t_7);
  __Pyx_XDECREF(__pyx_t_8);
  __Pyx_AddTraceback("numpy.import_array", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = -1;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":991
 *         raise ImportError("numpy.core.multiarray failed to import")
 * 
 * cdef inline int import_umath() except -1:             # <<<<<<<<<<<<<<
 *     try:
 *         _import_umath()
 */

static CYTHON_INLINE int __pyx_f_5numpy_import_umath(void) {
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  int __pyx_t_4;
  PyObject *__pyx_t_5 = NULL;
  PyObject *__pyx_t_6 = NULL;
  PyObject *__pyx_t_7 = NULL;
  PyObject *__pyx_t_8 = NULL;
  __Pyx_RefNannySetupContext("import_umath", 0);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":992
 * 
 * cdef inline int import_umath() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_umath()
 *     except Exception:
 */
  {
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ExceptionSave(&__pyx_t_1, &__pyx_t_2, &__pyx_t_3);
    __Pyx_XGOTREF(__pyx_t_1);
    __Pyx_XGOTREF(__pyx_t_2);
    __Pyx_XGOTREF(__pyx_t_3);
    /*try:*/ {

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":993
 * cdef inline int import_umath() except -1:
 *     try:
 *         _import_umath()             # <<<<<<<<<<<<<<
 *     except Exception:
 *         raise ImportError("numpy.core.umath failed to import")
 */
      __pyx_t_4 = _import_umath(); if (unlikely(__pyx_t_4 == -1)) __PYX_ERR(1, 993, __pyx_L3_error)

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":992
 * 
 * cdef inline int import_umath() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_umath()
 *     except Exception:
 */
    }
    __Pyx_XDECREF(__pyx_t_1); __pyx_t_1 = 0;
    __Pyx_XDECREF(__pyx_t_2); __pyx_t_2 = 0;
    __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
    goto __pyx_L10_try_end;
    __pyx_L3_error:;
    __Pyx_PyThreadState_assign

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":994
 *     try:
 *         _import_umath()
 *     except Exception:             # <<<<<<<<<<<<<<
 *         raise ImportError("numpy.core.umath failed to import")
 * 
 */
    __pyx_t_4 = __Pyx_PyErr_ExceptionMatches(((PyObject *)(&((PyTypeObject*)PyExc_Exception)[0])));
    if (__pyx_t_4) {
      __Pyx_AddTraceback("numpy.import_umath", __pyx_clineno, __pyx_lineno, __pyx_filename);
      if (__Pyx_GetException(&__pyx_t_5, &__pyx_t_6, &__pyx_t_7) < 0) __PYX_ERR(1, 994, __pyx_L5_except_error)
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_GOTREF(__pyx_t_7);

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":995
 *         _import_umath()
 *     except Exception:
 *         raise ImportError("numpy.core.umath failed to import")             # <<<<<<<<<<<<<<
 * 
 * cdef inline int import_ufunc() except -1:
 */
      __pyx_t_8 = __Pyx_PyObject_Call(__pyx_builtin_ImportError, __pyx_tuple__19, NULL); if (unlikely(!__pyx_t_8)) __PYX_ERR(1, 995, __pyx_L5_except_error)
      __Pyx_GOTREF(__pyx_t_8);
      __Pyx_Raise(__pyx_t_8, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
      __PYX_ERR(1, 995, __pyx_L5_except_error)
    }
    goto __pyx_L5_except_error;
    __pyx_L5_except_error:;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":992
 * 
 * cdef inline int import_umath() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_umath()
 *     except Exception:
 */
    __Pyx_PyThreadState_assign
    __Pyx_XGIVEREF(__pyx_t_1);
    __Pyx_XGIVEREF(__pyx_t_2);
    __Pyx_XGIVEREF(__pyx_t_3);
    __Pyx_ExceptionReset(__pyx_t_1, __pyx_t_2, __pyx_t_3);
    goto __pyx_L1_error;
    __pyx_L10_try_end:;
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":991
 *         raise ImportError("numpy.core.multiarray failed to import")
 * 
 * cdef inline int import_umath() except -1:             # <<<<<<<<<<<<<<
 *     try:
 *         _import_umath()
 */

  /* function exit code */
  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_XDECREF(__pyx_t_7);
  __Pyx_XDECREF(__pyx_t_8);
  __Pyx_AddTraceback("numpy.import_umath", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = -1;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":997
 *         raise ImportError("numpy.core.umath failed to import")
 * 
 * cdef inline int import_ufunc() except -1:             # <<<<<<<<<<<<<<
 *     try:
 *         _import_umath()
 */

static CYTHON_INLINE int __pyx_f_5numpy_import_ufunc(void) {
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  int __pyx_t_4;
  PyObject *__pyx_t_5 = NULL;
  PyObject *__pyx_t_6 = NULL;
  PyObject *__pyx_t_7 = NULL;
  PyObject *__pyx_t_8 = NULL;
  __Pyx_RefNannySetupContext("import_ufunc", 0);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":998
 * 
 * cdef inline int import_ufunc() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_umath()
 *     except Exception:
 */
  {
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ExceptionSave(&__pyx_t_1, &__pyx_t_2, &__pyx_t_3);
    __Pyx_XGOTREF(__pyx_t_1);
    __Pyx_XGOTREF(__pyx_t_2);
    __Pyx_XGOTREF(__pyx_t_3);
    /*try:*/ {

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":999
 * cdef inline int import_ufunc() except -1:
 *     try:
 *         _import_umath()             # <<<<<<<<<<<<<<
 *     except Exception:
 *         raise ImportError("numpy.core.umath failed to import")
 */
      __pyx_t_4 = _import_umath(); if (unlikely(__pyx_t_4 == -1)) __PYX_ERR(1, 999, __pyx_L3_error)

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":998
 * 
 * cdef inline int import_ufunc() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_umath()
 *     except Exception:
 */
    }
    __Pyx_XDECREF(__pyx_t_1); __pyx_t_1 = 0;
    __Pyx_XDECREF(__pyx_t_2); __pyx_t_2 = 0;
    __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
    goto __pyx_L10_try_end;
    __pyx_L3_error:;
    __Pyx_PyThreadState_assign

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1000
 *     try:
 *         _import_umath()
 *     except Exception:             # <<<<<<<<<<<<<<
 *         raise ImportError("numpy.core.umath failed to import")
 */
    __pyx_t_4 = __Pyx_PyErr_ExceptionMatches(((PyObject *)(&((PyTypeObject*)PyExc_Exception)[0])));
    if (__pyx_t_4) {
      __Pyx_AddTraceback("numpy.import_ufunc", __pyx_clineno, __pyx_lineno, __pyx_filename);
      if (__Pyx_GetException(&__pyx_t_5, &__pyx_t_6, &__pyx_t_7) < 0) __PYX_ERR(1, 1000, __pyx_L5_except_error)
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_GOTREF(__pyx_t_7);

      /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1001
 *         _import_umath()
 *     except Exception:
 *         raise ImportError("numpy.core.umath failed to import")             # <<<<<<<<<<<<<<
 */
      __pyx_t_8 = __Pyx_PyObject_Call(__pyx_builtin_ImportError, __pyx_tuple__20, NULL); if (unlikely(!__pyx_t_8)) __PYX_ERR(1, 1001, __pyx_L5_except_error)
      __Pyx_GOTREF(__pyx_t_8);
      __Pyx_Raise(__pyx_t_8, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
      __PYX_ERR(1, 1001, __pyx_L5_except_error)
    }
    goto __pyx_L5_except_error;
    __pyx_L5_except_error:;

    /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":998
 * 
 * cdef inline int import_ufunc() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_umath()
 *     except Exception:
 */
    __Pyx_PyThreadState_assign
    __Pyx_XGIVEREF(__pyx_t_1);
    __Pyx_XGIVEREF(__pyx_t_2);
    __Pyx_XGIVEREF(__pyx_t_3);
    __Pyx_ExceptionReset(__pyx_t_1, __pyx_t_2, __pyx_t_3);
    goto __pyx_L1_error;
    __pyx_L10_try_end:;
  }

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":997
 *         raise ImportError("numpy.core.umath failed to import")
 * 
 * cdef inline int import_ufunc() except -1:             # <<<<<<<<<<<<<<
 *     try:
 *         _import_umath()
 */

  /* function exit code */
  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_XDECREF(__pyx_t_7);
  __Pyx_XDECREF(__pyx_t_8);
  __Pyx_AddTraceback("numpy.import_ufunc", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = -1;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyMethodDef __pyx_methods[] = {
  {0, 0, 0, 0}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef __pyx_moduledef = {
  #if PY_VERSION_HEX < 0x03020000
    { PyObject_HEAD_INIT(NULL) NULL, 0, NULL },
  #else
    PyModuleDef_HEAD_INIT,
  #endif
    "cpu_nms",
    0, /* m_doc */
    -1, /* m_size */
    __pyx_methods /* m_methods */,
    NULL, /* m_reload */
    NULL, /* m_traverse */
    NULL, /* m_clear */
    NULL /* m_free */
};
#endif

static __Pyx_StringTabEntry __pyx_string_tab[] = {
  {&__pyx_kp_u_Format_string_allocated_too_shor, __pyx_k_Format_string_allocated_too_shor, sizeof(__pyx_k_Format_string_allocated_too_shor), 0, 1, 0, 0},
  {&__pyx_kp_u_Format_string_allocated_too_shor_2, __pyx_k_Format_string_allocated_too_shor_2, sizeof(__pyx_k_Format_string_allocated_too_shor_2), 0, 1, 0, 0},
  {&__pyx_n_s_ImportError, __pyx_k_ImportError, sizeof(__pyx_k_ImportError), 0, 0, 1, 1},
  {&__pyx_n_s_N, __pyx_k_N, sizeof(__pyx_k_N), 0, 0, 1, 1},
  {&__pyx_kp_u_Non_native_byte_order_not_suppor, __pyx_k_Non_native_byte_order_not_suppor, sizeof(__pyx_k_Non_native_byte_order_not_suppor), 0, 1, 0, 0},
  {&__pyx_n_s_Nt, __pyx_k_Nt, sizeof(__pyx_k_Nt), 0, 0, 1, 1},
  {&__pyx_n_s_RuntimeError, __pyx_k_RuntimeError, sizeof(__pyx_k_RuntimeError), 0, 0, 1, 1},
  {&__pyx_n_s_ValueError, __pyx_k_ValueError, sizeof(__pyx_k_ValueError), 0, 0, 1, 1},
  {&__pyx_n_s_area, __pyx_k_area, sizeof(__pyx_k_area), 0, 0, 1, 1},
  {&__pyx_n_s_areas, __pyx_k_areas, sizeof(__pyx_k_areas), 0, 0, 1, 1},
  {&__pyx_n_s_argsort, __pyx_k_argsort, sizeof(__pyx_k_argsort), 0, 0, 1, 1},
  {&__pyx_n_s_box_area, __pyx_k_box_area, sizeof(__pyx_k_box_area), 0, 0, 1, 1},
  {&__pyx_n_s_boxes, __pyx_k_boxes, sizeof(__pyx_k_boxes), 0, 0, 1, 1},
  {&__pyx_n_s_cpu_nms, __pyx_k_cpu_nms, sizeof(__pyx_k_cpu_nms), 0, 0, 1, 1},
  {&__pyx_n_s_cpu_soft_nms, __pyx_k_cpu_soft_nms, sizeof(__pyx_k_cpu_soft_nms), 0, 0, 1, 1},
  {&__pyx_n_s_dets, __pyx_k_dets, sizeof(__pyx_k_dets), 0, 0, 1, 1},
  {&__pyx_n_s_dtype, __pyx_k_dtype, sizeof(__pyx_k_dtype), 0, 0, 1, 1},
  {&__pyx_n_s_exp, __pyx_k_exp, sizeof(__pyx_k_exp), 0, 0, 1, 1},
  {&__pyx_n_s_h, __pyx_k_h, sizeof(__pyx_k_h), 0, 0, 1, 1},
  {&__pyx_kp_s_home_messi_RFBNet_utils_nms_cpu, __pyx_k_home_messi_RFBNet_utils_nms_cpu, sizeof(__pyx_k_home_messi_RFBNet_utils_nms_cpu), 0, 0, 1, 0},
  {&__pyx_n_s_i, __pyx_k_i, sizeof(__pyx_k_i), 0, 0, 1, 1},
  {&__pyx_n_s_i_2, __pyx_k_i_2, sizeof(__pyx_k_i_2), 0, 0, 1, 1},
  {&__pyx_n_s_iarea, __pyx_k_iarea, sizeof(__pyx_k_iarea), 0, 0, 1, 1},
  {&__pyx_n_s_ih, __pyx_k_ih, sizeof(__pyx_k_ih), 0, 0, 1, 1},
  {&__pyx_n_s_import, __pyx_k_import, sizeof(__pyx_k_import), 0, 0, 1, 1},
  {&__pyx_n_s_int, __pyx_k_int, sizeof(__pyx_k_int), 0, 0, 1, 1},
  {&__pyx_n_s_inter, __pyx_k_inter, sizeof(__pyx_k_inter), 0, 0, 1, 1},
  {&__pyx_n_s_iw, __pyx_k_iw, sizeof(__pyx_k_iw), 0, 0, 1, 1},
  {&__pyx_n_s_ix1, __pyx_k_ix1, sizeof(__pyx_k_ix1), 0, 0, 1, 1},
  {&__pyx_n_s_ix2, __pyx_k_ix2, sizeof(__pyx_k_ix2), 0, 0, 1, 1},
  {&__pyx_n_s_iy1, __pyx_k_iy1, sizeof(__pyx_k_iy1), 0, 0, 1, 1},
  {&__pyx_n_s_iy2, __pyx_k_iy2, sizeof(__pyx_k_iy2), 0, 0, 1, 1},
  {&__pyx_n_s_j, __pyx_k_j, sizeof(__pyx_k_j), 0, 0, 1, 1},
  {&__pyx_n_s_j_2, __pyx_k_j_2, sizeof(__pyx_k_j_2), 0, 0, 1, 1},
  {&__pyx_n_s_keep, __pyx_k_keep, sizeof(__pyx_k_keep), 0, 0, 1, 1},
  {&__pyx_n_s_main, __pyx_k_main, sizeof(__pyx_k_main), 0, 0, 1, 1},
  {&__pyx_n_s_maxpos, __pyx_k_maxpos, sizeof(__pyx_k_maxpos), 0, 0, 1, 1},
  {&__pyx_n_s_maxscore, __pyx_k_maxscore, sizeof(__pyx_k_maxscore), 0, 0, 1, 1},
  {&__pyx_n_s_method, __pyx_k_method, sizeof(__pyx_k_method), 0, 0, 1, 1},
  {&__pyx_kp_u_ndarray_is_not_C_contiguous, __pyx_k_ndarray_is_not_C_contiguous, sizeof(__pyx_k_ndarray_is_not_C_contiguous), 0, 1, 0, 0},
  {&__pyx_kp_u_ndarray_is_not_Fortran_contiguou, __pyx_k_ndarray_is_not_Fortran_contiguou, sizeof(__pyx_k_ndarray_is_not_Fortran_contiguou), 0, 1, 0, 0},
  {&__pyx_n_s_ndets, __pyx_k_ndets, sizeof(__pyx_k_ndets), 0, 0, 1, 1},
  {&__pyx_n_s_nms_cpu_nms, __pyx_k_nms_cpu_nms, sizeof(__pyx_k_nms_cpu_nms), 0, 0, 1, 1},
  {&__pyx_n_s_np, __pyx_k_np, sizeof(__pyx_k_np), 0, 0, 1, 1},
  {&__pyx_n_s_numpy, __pyx_k_numpy, sizeof(__pyx_k_numpy), 0, 0, 1, 1},
  {&__pyx_kp_s_numpy_core_multiarray_failed_to, __pyx_k_numpy_core_multiarray_failed_to, sizeof(__pyx_k_numpy_core_multiarray_failed_to), 0, 0, 1, 0},
  {&__pyx_kp_s_numpy_core_umath_failed_to_impor, __pyx_k_numpy_core_umath_failed_to_impor, sizeof(__pyx_k_numpy_core_umath_failed_to_impor), 0, 0, 1, 0},
  {&__pyx_n_s_order, __pyx_k_order, sizeof(__pyx_k_order), 0, 0, 1, 1},
  {&__pyx_n_s_ov, __pyx_k_ov, sizeof(__pyx_k_ov), 0, 0, 1, 1},
  {&__pyx_n_s_ovr, __pyx_k_ovr, sizeof(__pyx_k_ovr), 0, 0, 1, 1},
  {&__pyx_n_s_pos, __pyx_k_pos, sizeof(__pyx_k_pos), 0, 0, 1, 1},
  {&__pyx_n_s_range, __pyx_k_range, sizeof(__pyx_k_range), 0, 0, 1, 1},
  {&__pyx_n_s_s, __pyx_k_s, sizeof(__pyx_k_s), 0, 0, 1, 1},
  {&__pyx_n_s_scores, __pyx_k_scores, sizeof(__pyx_k_scores), 0, 0, 1, 1},
  {&__pyx_n_s_sigma, __pyx_k_sigma, sizeof(__pyx_k_sigma), 0, 0, 1, 1},
  {&__pyx_n_s_suppressed, __pyx_k_suppressed, sizeof(__pyx_k_suppressed), 0, 0, 1, 1},
  {&__pyx_n_s_test, __pyx_k_test, sizeof(__pyx_k_test), 0, 0, 1, 1},
  {&__pyx_n_s_thresh, __pyx_k_thresh, sizeof(__pyx_k_thresh), 0, 0, 1, 1},
  {&__pyx_n_s_threshold, __pyx_k_threshold, sizeof(__pyx_k_threshold), 0, 0, 1, 1},
  {&__pyx_n_s_ts, __pyx_k_ts, sizeof(__pyx_k_ts), 0, 0, 1, 1},
  {&__pyx_n_s_tx1, __pyx_k_tx1, sizeof(__pyx_k_tx1), 0, 0, 1, 1},
  {&__pyx_n_s_tx2, __pyx_k_tx2, sizeof(__pyx_k_tx2), 0, 0, 1, 1},
  {&__pyx_n_s_ty1, __pyx_k_ty1, sizeof(__pyx_k_ty1), 0, 0, 1, 1},
  {&__pyx_n_s_ty2, __pyx_k_ty2, sizeof(__pyx_k_ty2), 0, 0, 1, 1},
  {&__pyx_n_s_ua, __pyx_k_ua, sizeof(__pyx_k_ua), 0, 0, 1, 1},
  {&__pyx_kp_u_unknown_dtype_code_in_numpy_pxd, __pyx_k_unknown_dtype_code_in_numpy_pxd, sizeof(__pyx_k_unknown_dtype_code_in_numpy_pxd), 0, 1, 0, 0},
  {&__pyx_n_s_w, __pyx_k_w, sizeof(__pyx_k_w), 0, 0, 1, 1},
  {&__pyx_n_s_weight, __pyx_k_weight, sizeof(__pyx_k_weight), 0, 0, 1, 1},
  {&__pyx_n_s_x1, __pyx_k_x1, sizeof(__pyx_k_x1), 0, 0, 1, 1},
  {&__pyx_n_s_x2, __pyx_k_x2, sizeof(__pyx_k_x2), 0, 0, 1, 1},
  {&__pyx_n_s_xx1, __pyx_k_xx1, sizeof(__pyx_k_xx1), 0, 0, 1, 1},
  {&__pyx_n_s_xx2, __pyx_k_xx2, sizeof(__pyx_k_xx2), 0, 0, 1, 1},
  {&__pyx_n_s_y1, __pyx_k_y1, sizeof(__pyx_k_y1), 0, 0, 1, 1},
  {&__pyx_n_s_y2, __pyx_k_y2, sizeof(__pyx_k_y2), 0, 0, 1, 1},
  {&__pyx_n_s_yy1, __pyx_k_yy1, sizeof(__pyx_k_yy1), 0, 0, 1, 1},
  {&__pyx_n_s_yy2, __pyx_k_yy2, sizeof(__pyx_k_yy2), 0, 0, 1, 1},
  {&__pyx_n_s_zeros, __pyx_k_zeros, sizeof(__pyx_k_zeros), 0, 0, 1, 1},
  {0, 0, 0, 0, 0, 0, 0}
};
static int __Pyx_InitCachedBuiltins(void) {
  __pyx_builtin_range = __Pyx_GetBuiltinName(__pyx_n_s_range); if (!__pyx_builtin_range) __PYX_ERR(0, 43, __pyx_L1_error)
  __pyx_builtin_ValueError = __Pyx_GetBuiltinName(__pyx_n_s_ValueError); if (!__pyx_builtin_ValueError) __PYX_ERR(1, 218, __pyx_L1_error)
  __pyx_builtin_RuntimeError = __Pyx_GetBuiltinName(__pyx_n_s_RuntimeError); if (!__pyx_builtin_RuntimeError) __PYX_ERR(1, 799, __pyx_L1_error)
  __pyx_builtin_ImportError = __Pyx_GetBuiltinName(__pyx_n_s_ImportError); if (!__pyx_builtin_ImportError) __PYX_ERR(1, 989, __pyx_L1_error)
  return 0;
  __pyx_L1_error:;
  return -1;
}

static int __Pyx_InitCachedConstants(void) {
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__Pyx_InitCachedConstants", 0);

  /* "nms/cpu_nms.pyx":18
 * 
 * def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
 *     cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
 *     cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
 */
  __pyx_slice_ = PySlice_New(Py_None, Py_None, Py_None); if (unlikely(!__pyx_slice_)) __PYX_ERR(0, 18, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_slice_);
  __Pyx_GIVEREF(__pyx_slice_);
  __pyx_tuple__2 = PyTuple_Pack(2, __pyx_slice_, __pyx_int_0); if (unlikely(!__pyx_tuple__2)) __PYX_ERR(0, 18, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__2);
  __Pyx_GIVEREF(__pyx_tuple__2);

  /* "nms/cpu_nms.pyx":19
 * def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
 *     cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
 *     cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
 *     cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
 */
  __pyx_slice__3 = PySlice_New(Py_None, Py_None, Py_None); if (unlikely(!__pyx_slice__3)) __PYX_ERR(0, 19, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_slice__3);
  __Pyx_GIVEREF(__pyx_slice__3);
  __pyx_tuple__4 = PyTuple_Pack(2, __pyx_slice__3, __pyx_int_1); if (unlikely(!__pyx_tuple__4)) __PYX_ERR(0, 19, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__4);
  __Pyx_GIVEREF(__pyx_tuple__4);

  /* "nms/cpu_nms.pyx":20
 *     cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
 *     cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
 *     cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
 *     cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]
 */
  __pyx_slice__5 = PySlice_New(Py_None, Py_None, Py_None); if (unlikely(!__pyx_slice__5)) __PYX_ERR(0, 20, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_slice__5);
  __Pyx_GIVEREF(__pyx_slice__5);
  __pyx_tuple__6 = PyTuple_Pack(2, __pyx_slice__5, __pyx_int_2); if (unlikely(!__pyx_tuple__6)) __PYX_ERR(0, 20, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__6);
  __Pyx_GIVEREF(__pyx_tuple__6);

  /* "nms/cpu_nms.pyx":21
 *     cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
 *     cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
 *     cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]
 * 
 */
  __pyx_slice__7 = PySlice_New(Py_None, Py_None, Py_None); if (unlikely(!__pyx_slice__7)) __PYX_ERR(0, 21, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_slice__7);
  __Pyx_GIVEREF(__pyx_slice__7);
  __pyx_tuple__8 = PyTuple_Pack(2, __pyx_slice__7, __pyx_int_3); if (unlikely(!__pyx_tuple__8)) __PYX_ERR(0, 21, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__8);
  __Pyx_GIVEREF(__pyx_tuple__8);

  /* "nms/cpu_nms.pyx":22
 *     cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
 *     cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
 *     cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
 */
  __pyx_slice__9 = PySlice_New(Py_None, Py_None, Py_None); if (unlikely(!__pyx_slice__9)) __PYX_ERR(0, 22, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_slice__9);
  __Pyx_GIVEREF(__pyx_slice__9);
  __pyx_tuple__10 = PyTuple_Pack(2, __pyx_slice__9, __pyx_int_4); if (unlikely(!__pyx_tuple__10)) __PYX_ERR(0, 22, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__10);
  __Pyx_GIVEREF(__pyx_tuple__10);

  /* "nms/cpu_nms.pyx":25
 * 
 *     cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
 *     cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]             # <<<<<<<<<<<<<<
 * 
 *     cdef int ndets = dets.shape[0]
 */
  __pyx_slice__11 = PySlice_New(Py_None, Py_None, __pyx_int_neg_1); if (unlikely(!__pyx_slice__11)) __PYX_ERR(0, 25, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_slice__11);
  __Pyx_GIVEREF(__pyx_slice__11);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":218
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")             # <<<<<<<<<<<<<<
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 */
  __pyx_tuple__12 = PyTuple_Pack(1, __pyx_kp_u_ndarray_is_not_C_contiguous); if (unlikely(!__pyx_tuple__12)) __PYX_ERR(1, 218, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__12);
  __Pyx_GIVEREF(__pyx_tuple__12);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":222
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")             # <<<<<<<<<<<<<<
 * 
 *             info.buf = PyArray_DATA(self)
 */
  __pyx_tuple__13 = PyTuple_Pack(1, __pyx_kp_u_ndarray_is_not_Fortran_contiguou); if (unlikely(!__pyx_tuple__13)) __PYX_ERR(1, 222, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__13);
  __Pyx_GIVEREF(__pyx_tuple__13);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":259
 *                 if ((descr.byteorder == c'>' and little_endian) or
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"
 */
  __pyx_tuple__14 = PyTuple_Pack(1, __pyx_kp_u_Non_native_byte_order_not_suppor); if (unlikely(!__pyx_tuple__14)) __PYX_ERR(1, 259, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__14);
  __Pyx_GIVEREF(__pyx_tuple__14);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":799
 * 
 *         if (end - f) - <int>(new_offset - offset[0]) < 15:
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")             # <<<<<<<<<<<<<<
 * 
 *         if ((child.byteorder == c'>' and little_endian) or
 */
  __pyx_tuple__15 = PyTuple_Pack(1, __pyx_kp_u_Format_string_allocated_too_shor); if (unlikely(!__pyx_tuple__15)) __PYX_ERR(1, 799, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__15);
  __Pyx_GIVEREF(__pyx_tuple__15);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":803
 *         if ((child.byteorder == c'>' and little_endian) or
 *             (child.byteorder == c'<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *             # One could encode it in the format string and have Cython
 *             # complain instead, BUT: < and > in format strings also imply
 */
  __pyx_tuple__16 = PyTuple_Pack(1, __pyx_kp_u_Non_native_byte_order_not_suppor); if (unlikely(!__pyx_tuple__16)) __PYX_ERR(1, 803, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__16);
  __Pyx_GIVEREF(__pyx_tuple__16);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":823
 *             t = child.type_num
 *             if end - f < 5:
 *                 raise RuntimeError(u"Format string allocated too short.")             # <<<<<<<<<<<<<<
 * 
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 */
  __pyx_tuple__17 = PyTuple_Pack(1, __pyx_kp_u_Format_string_allocated_too_shor_2); if (unlikely(!__pyx_tuple__17)) __PYX_ERR(1, 823, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__17);
  __Pyx_GIVEREF(__pyx_tuple__17);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":989
 *         _import_array()
 *     except Exception:
 *         raise ImportError("numpy.core.multiarray failed to import")             # <<<<<<<<<<<<<<
 * 
 * cdef inline int import_umath() except -1:
 */
  __pyx_tuple__18 = PyTuple_Pack(1, __pyx_kp_s_numpy_core_multiarray_failed_to); if (unlikely(!__pyx_tuple__18)) __PYX_ERR(1, 989, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__18);
  __Pyx_GIVEREF(__pyx_tuple__18);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":995
 *         _import_umath()
 *     except Exception:
 *         raise ImportError("numpy.core.umath failed to import")             # <<<<<<<<<<<<<<
 * 
 * cdef inline int import_ufunc() except -1:
 */
  __pyx_tuple__19 = PyTuple_Pack(1, __pyx_kp_s_numpy_core_umath_failed_to_impor); if (unlikely(!__pyx_tuple__19)) __PYX_ERR(1, 995, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__19);
  __Pyx_GIVEREF(__pyx_tuple__19);

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1001
 *         _import_umath()
 *     except Exception:
 *         raise ImportError("numpy.core.umath failed to import")             # <<<<<<<<<<<<<<
 */
  __pyx_tuple__20 = PyTuple_Pack(1, __pyx_kp_s_numpy_core_umath_failed_to_impor); if (unlikely(!__pyx_tuple__20)) __PYX_ERR(1, 1001, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__20);
  __Pyx_GIVEREF(__pyx_tuple__20);

  /* "nms/cpu_nms.pyx":17
 *     return a if a <= b else b
 * 
 * def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
 *     cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
 */
  __pyx_tuple__21 = PyTuple_Pack(29, __pyx_n_s_dets, __pyx_n_s_thresh, __pyx_n_s_x1, __pyx_n_s_y1, __pyx_n_s_x2, __pyx_n_s_y2, __pyx_n_s_scores, __pyx_n_s_areas, __pyx_n_s_order, __pyx_n_s_ndets, __pyx_n_s_suppressed, __pyx_n_s_i, __pyx_n_s_j, __pyx_n_s_i_2, __pyx_n_s_j_2, __pyx_n_s_ix1, __pyx_n_s_iy1, __pyx_n_s_ix2, __pyx_n_s_iy2, __pyx_n_s_iarea, __pyx_n_s_xx1, __pyx_n_s_yy1, __pyx_n_s_xx2, __pyx_n_s_yy2, __pyx_n_s_w, __pyx_n_s_h, __pyx_n_s_inter, __pyx_n_s_ovr, __pyx_n_s_keep); if (unlikely(!__pyx_tuple__21)) __PYX_ERR(0, 17, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__21);
  __Pyx_GIVEREF(__pyx_tuple__21);
  __pyx_codeobj__22 = (PyObject*)__Pyx_PyCode_New(2, 0, 29, 0, 0, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__21, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_home_messi_RFBNet_utils_nms_cpu, __pyx_n_s_cpu_nms, 17, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__22)) __PYX_ERR(0, 17, __pyx_L1_error)

  /* "nms/cpu_nms.pyx":70
 *     return keep
 * 
 * def cpu_soft_nms(np.ndarray[float, ndim=2] boxes, float sigma=0.5, float Nt=0.3, float threshold=0.001, unsigned int method=0):             # <<<<<<<<<<<<<<
 *     cdef unsigned int N = boxes.shape[0]
 *     cdef float iw, ih, box_area
 */
  __pyx_tuple__23 = PyTuple_Pack(28, __pyx_n_s_boxes, __pyx_n_s_sigma, __pyx_n_s_Nt, __pyx_n_s_threshold, __pyx_n_s_method, __pyx_n_s_N, __pyx_n_s_iw, __pyx_n_s_ih, __pyx_n_s_box_area, __pyx_n_s_ua, __pyx_n_s_pos, __pyx_n_s_maxscore, __pyx_n_s_maxpos, __pyx_n_s_x1, __pyx_n_s_x2, __pyx_n_s_y1, __pyx_n_s_y2, __pyx_n_s_tx1, __pyx_n_s_tx2, __pyx_n_s_ty1, __pyx_n_s_ty2, __pyx_n_s_ts, __pyx_n_s_area, __pyx_n_s_weight, __pyx_n_s_ov, __pyx_n_s_i_2, __pyx_n_s_s, __pyx_n_s_keep); if (unlikely(!__pyx_tuple__23)) __PYX_ERR(0, 70, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__23);
  __Pyx_GIVEREF(__pyx_tuple__23);
  __pyx_codeobj__24 = (PyObject*)__Pyx_PyCode_New(5, 0, 28, 0, 0, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__23, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_home_messi_RFBNet_utils_nms_cpu, __pyx_n_s_cpu_soft_nms, 70, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__24)) __PYX_ERR(0, 70, __pyx_L1_error)
  __Pyx_RefNannyFinishContext();
  return 0;
  __pyx_L1_error:;
  __Pyx_RefNannyFinishContext();
  return -1;
}

static int __Pyx_InitGlobals(void) {
  if (__Pyx_InitStrings(__pyx_string_tab) < 0) __PYX_ERR(0, 1, __pyx_L1_error);
  __pyx_int_0 = PyInt_FromLong(0); if (unlikely(!__pyx_int_0)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_int_1 = PyInt_FromLong(1); if (unlikely(!__pyx_int_1)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_int_2 = PyInt_FromLong(2); if (unlikely(!__pyx_int_2)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_int_3 = PyInt_FromLong(3); if (unlikely(!__pyx_int_3)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_int_4 = PyInt_FromLong(4); if (unlikely(!__pyx_int_4)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_int_neg_1 = PyInt_FromLong(-1); if (unlikely(!__pyx_int_neg_1)) __PYX_ERR(0, 1, __pyx_L1_error)
  return 0;
  __pyx_L1_error:;
  return -1;
}

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initcpu_nms(void); /*proto*/
PyMODINIT_FUNC initcpu_nms(void)
#else
PyMODINIT_FUNC PyInit_cpu_nms(void); /*proto*/
PyMODINIT_FUNC PyInit_cpu_nms(void)
#endif
{
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannyDeclarations
  #if CYTHON_REFNANNY
  __Pyx_RefNanny = __Pyx_RefNannyImportAPI("refnanny");
  if (!__Pyx_RefNanny) {
      PyErr_Clear();
      __Pyx_RefNanny = __Pyx_RefNannyImportAPI("Cython.Runtime.refnanny");
      if (!__Pyx_RefNanny)
          Py_FatalError("failed to import 'refnanny' module");
  }
  #endif
  __Pyx_RefNannySetupContext("PyMODINIT_FUNC PyInit_cpu_nms(void)", 0);
  if (__Pyx_check_binary_version() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_empty_tuple = PyTuple_New(0); if (unlikely(!__pyx_empty_tuple)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_empty_bytes = PyBytes_FromStringAndSize("", 0); if (unlikely(!__pyx_empty_bytes)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_empty_unicode = PyUnicode_FromStringAndSize("", 0); if (unlikely(!__pyx_empty_unicode)) __PYX_ERR(0, 1, __pyx_L1_error)
  #ifdef __Pyx_CyFunction_USED
  if (__pyx_CyFunction_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  #ifdef __Pyx_FusedFunction_USED
  if (__pyx_FusedFunction_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  #ifdef __Pyx_Coroutine_USED
  if (__pyx_Coroutine_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  #ifdef __Pyx_Generator_USED
  if (__pyx_Generator_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  #ifdef __Pyx_StopAsyncIteration_USED
  if (__pyx_StopAsyncIteration_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  /*--- Library function declarations ---*/
  /*--- Threads initialization code ---*/
  #if defined(__PYX_FORCE_INIT_THREADS) && __PYX_FORCE_INIT_THREADS
  #ifdef WITH_THREAD /* Python build with threading support? */
  PyEval_InitThreads();
  #endif
  #endif
  /*--- Module creation code ---*/
  #if PY_MAJOR_VERSION < 3
  __pyx_m = Py_InitModule4("cpu_nms", __pyx_methods, 0, 0, PYTHON_API_VERSION); Py_XINCREF(__pyx_m);
  #else
  __pyx_m = PyModule_Create(&__pyx_moduledef);
  #endif
  if (unlikely(!__pyx_m)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_d = PyModule_GetDict(__pyx_m); if (unlikely(!__pyx_d)) __PYX_ERR(0, 1, __pyx_L1_error)
  Py_INCREF(__pyx_d);
  __pyx_b = PyImport_AddModule(__Pyx_BUILTIN_MODULE_NAME); if (unlikely(!__pyx_b)) __PYX_ERR(0, 1, __pyx_L1_error)
  #if CYTHON_COMPILING_IN_PYPY
  Py_INCREF(__pyx_b);
  #endif
  if (PyObject_SetAttrString(__pyx_m, "__builtins__", __pyx_b) < 0) __PYX_ERR(0, 1, __pyx_L1_error);
  /*--- Initialize various global constants etc. ---*/
  if (__Pyx_InitGlobals() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #if PY_MAJOR_VERSION < 3 && (__PYX_DEFAULT_STRING_ENCODING_IS_ASCII || __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT)
  if (__Pyx_init_sys_getdefaultencoding_params() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  if (__pyx_module_is_main_nms__cpu_nms) {
    if (PyObject_SetAttrString(__pyx_m, "__name__", __pyx_n_s_main) < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  }
  #if PY_MAJOR_VERSION >= 3
  {
    PyObject *modules = PyImport_GetModuleDict(); if (unlikely(!modules)) __PYX_ERR(0, 1, __pyx_L1_error)
    if (!PyDict_GetItemString(modules, "nms.cpu_nms")) {
      if (unlikely(PyDict_SetItemString(modules, "nms.cpu_nms", __pyx_m) < 0)) __PYX_ERR(0, 1, __pyx_L1_error)
    }
  }
  #endif
  /*--- Builtin init code ---*/
  if (__Pyx_InitCachedBuiltins() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  /*--- Constants init code ---*/
  if (__Pyx_InitCachedConstants() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  /*--- Global init code ---*/
  /*--- Variable export code ---*/
  /*--- Function export code ---*/
  /*--- Type init code ---*/
  /*--- Type import code ---*/
  __pyx_ptype_7cpython_4type_type = __Pyx_ImportType(__Pyx_BUILTIN_MODULE_NAME, "type", 
  #if CYTHON_COMPILING_IN_PYPY
  sizeof(PyTypeObject),
  #else
  sizeof(PyHeapTypeObject),
  #endif
  0); if (unlikely(!__pyx_ptype_7cpython_4type_type)) __PYX_ERR(2, 9, __pyx_L1_error)
  __pyx_ptype_5numpy_dtype = __Pyx_ImportType("numpy", "dtype", sizeof(PyArray_Descr), 0); if (unlikely(!__pyx_ptype_5numpy_dtype)) __PYX_ERR(1, 155, __pyx_L1_error)
  __pyx_ptype_5numpy_flatiter = __Pyx_ImportType("numpy", "flatiter", sizeof(PyArrayIterObject), 0); if (unlikely(!__pyx_ptype_5numpy_flatiter)) __PYX_ERR(1, 168, __pyx_L1_error)
  __pyx_ptype_5numpy_broadcast = __Pyx_ImportType("numpy", "broadcast", sizeof(PyArrayMultiIterObject), 0); if (unlikely(!__pyx_ptype_5numpy_broadcast)) __PYX_ERR(1, 172, __pyx_L1_error)
  __pyx_ptype_5numpy_ndarray = __Pyx_ImportType("numpy", "ndarray", sizeof(PyArrayObject), 0); if (unlikely(!__pyx_ptype_5numpy_ndarray)) __PYX_ERR(1, 181, __pyx_L1_error)
  __pyx_ptype_5numpy_ufunc = __Pyx_ImportType("numpy", "ufunc", sizeof(PyUFuncObject), 0); if (unlikely(!__pyx_ptype_5numpy_ufunc)) __PYX_ERR(1, 861, __pyx_L1_error)
  /*--- Variable import code ---*/
  /*--- Function import code ---*/
  /*--- Execution code ---*/
  #if defined(__Pyx_Generator_USED) || defined(__Pyx_Coroutine_USED)
  if (__Pyx_patch_abc() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif

  /* "nms/cpu_nms.pyx":8
 * # --------------------------------------------------------
 * 
 * import numpy as np             # <<<<<<<<<<<<<<
 * cimport numpy as np
 * 
 */
  __pyx_t_1 = __Pyx_Import(__pyx_n_s_numpy, 0, -1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 8, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_np, __pyx_t_1) < 0) __PYX_ERR(0, 8, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "nms/cpu_nms.pyx":17
 *     return a if a <= b else b
 * 
 * def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
 *     cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
 */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_3nms_7cpu_nms_1cpu_nms, NULL, __pyx_n_s_nms_cpu_nms); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 17, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_cpu_nms, __pyx_t_1) < 0) __PYX_ERR(0, 17, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "nms/cpu_nms.pyx":70
 *     return keep
 * 
 * def cpu_soft_nms(np.ndarray[float, ndim=2] boxes, float sigma=0.5, float Nt=0.3, float threshold=0.001, unsigned int method=0):             # <<<<<<<<<<<<<<
 *     cdef unsigned int N = boxes.shape[0]
 *     cdef float iw, ih, box_area
 */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_3nms_7cpu_nms_3cpu_soft_nms, NULL, __pyx_n_s_nms_cpu_nms); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 70, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_cpu_soft_nms, __pyx_t_1) < 0) __PYX_ERR(0, 70, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "nms/cpu_nms.pyx":1
 * # --------------------------------------------------------             # <<<<<<<<<<<<<<
 * # Fast R-CNN
 * # Copyright (c) 2015 Microsoft
 */
  __pyx_t_1 = PyDict_New(); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 1, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_test, __pyx_t_1) < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "../../.pyenv/versions/anaconda3/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":997
 *         raise ImportError("numpy.core.umath failed to import")
 * 
 * cdef inline int import_ufunc() except -1:             # <<<<<<<<<<<<<<
 *     try:
 *         _import_umath()
 */

  /*--- Wrapped vars code ---*/

  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  if (__pyx_m) {
    if (__pyx_d) {
      __Pyx_AddTraceback("init nms.cpu_nms", __pyx_clineno, __pyx_lineno, __pyx_filename);
    }
    Py_DECREF(__pyx_m); __pyx_m = 0;
  } else if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_ImportError, "init nms.cpu_nms");
  }
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  #if PY_MAJOR_VERSION < 3
  return;
  #else
  return __pyx_m;
  #endif
}

/* --- Runtime support code --- */
/* Refnanny */
#if CYTHON_REFNANNY
static __Pyx_RefNannyAPIStruct *__Pyx_RefNannyImportAPI(const char *modname) {
    PyObject *m = NULL, *p = NULL;
    void *r = NULL;
    m = PyImport_ImportModule((char *)modname);
    if (!m) goto end;
    p = PyObject_GetAttrString(m, (char *)"RefNannyAPI");
    if (!p) goto end;
    r = PyLong_AsVoidPtr(p);
end:
    Py_XDECREF(p);
    Py_XDECREF(m);
    return (__Pyx_RefNannyAPIStruct *)r;
}
#endif

/* GetBuiltinName */
static PyObject *__Pyx_GetBuiltinName(PyObject *name) {
    PyObject* result = __Pyx_PyObject_GetAttrStr(__pyx_b, name);
    if (unlikely(!result)) {
        PyErr_Format(PyExc_NameError,
#if PY_MAJOR_VERSION >= 3
            "name '%U' is not defined", name);
#else
            "name '%.200s' is not defined", PyString_AS_STRING(name));
#endif
    }
    return result;
}

/* RaiseArgTupleInvalid */
static void __Pyx_RaiseArgtupleInvalid(
    const char* func_name,
    int exact,
    Py_ssize_t num_min,
    Py_ssize_t num_max,
    Py_ssize_t num_found)
{
    Py_ssize_t num_expected;
    const char *more_or_less;
    if (num_found < num_min) {
        num_expected = num_min;
        more_or_less = "at least";
    } else {
        num_expected = num_max;
        more_or_less = "at most";
    }
    if (exact) {
        more_or_less = "exactly";
    }
    PyErr_Format(PyExc_TypeError,
                 "%.200s() takes %.8s %" CYTHON_FORMAT_SSIZE_T "d positional argument%.1s (%" CYTHON_FORMAT_SSIZE_T "d given)",
                 func_name, more_or_less, num_expected,
                 (num_expected == 1) ? "" : "s", num_found);
}

/* RaiseDoubleKeywords */
static void __Pyx_RaiseDoubleKeywordsError(
    const char* func_name,
    PyObject* kw_name)
{
    PyErr_Format(PyExc_TypeError,
        #if PY_MAJOR_VERSION >= 3
        "%s() got multiple values for keyword argument '%U'", func_name, kw_name);
        #else
        "%s() got multiple values for keyword argument '%s'", func_name,
        PyString_AsString(kw_name));
        #endif
}

/* ParseKeywords */
static int __Pyx_ParseOptionalKeywords(
    PyObject *kwds,
    PyObject **argnames[],
    PyObject *kwds2,
    PyObject *values[],
    Py_ssize_t num_pos_args,
    const char* function_name)
{
    PyObject *key = 0, *value = 0;
    Py_ssize_t pos = 0;
    PyObject*** name;
    PyObject*** first_kw_arg = argnames + num_pos_args;
    while (PyDict_Next(kwds, &pos, &key, &value)) {
        name = first_kw_arg;
        while (*name && (**name != key)) name++;
        if (*name) {
            values[name-argnames] = value;
            continue;
        }
        name = first_kw_arg;
        #if PY_MAJOR_VERSION < 3
        if (likely(PyString_CheckExact(key)) || likely(PyString_Check(key))) {
            while (*name) {
                if ((CYTHON_COMPILING_IN_PYPY || PyString_GET_SIZE(**name) == PyString_GET_SIZE(key))
                        && _PyString_Eq(**name, key)) {
                    values[name-argnames] = value;
                    break;
                }
                name++;
            }
            if (*name) continue;
            else {
                PyObject*** argname = argnames;
                while (argname != first_kw_arg) {
                    if ((**argname == key) || (
                            (CYTHON_COMPILING_IN_PYPY || PyString_GET_SIZE(**argname) == PyString_GET_SIZE(key))
                             && _PyString_Eq(**argname, key))) {
                        goto arg_passed_twice;
                    }
                    argname++;
                }
            }
        } else
        #endif
        if (likely(PyUnicode_Check(key))) {
            while (*name) {
                int cmp = (**name == key) ? 0 :
                #if !CYTHON_COMPILING_IN_PYPY && PY_MAJOR_VERSION >= 3
                    (PyUnicode_GET_SIZE(**name) != PyUnicode_GET_SIZE(key)) ? 1 :
                #endif
                    PyUnicode_Compare(**name, key);
                if (cmp < 0 && unlikely(PyErr_Occurred())) goto bad;
                if (cmp == 0) {
                    values[name-argnames] = value;
                    break;
                }
                name++;
            }
            if (*name) continue;
            else {
                PyObject*** argname = argnames;
                while (argname != first_kw_arg) {
                    int cmp = (**argname == key) ? 0 :
                    #if !CYTHON_COMPILING_IN_PYPY && PY_MAJOR_VERSION >= 3
                        (PyUnicode_GET_SIZE(**argname) != PyUnicode_GET_SIZE(key)) ? 1 :
                    #endif
                        PyUnicode_Compare(**argname, key);
                    if (cmp < 0 && unlikely(PyErr_Occurred())) goto bad;
                    if (cmp == 0) goto arg_passed_twice;
                    argname++;
                }
            }
        } else
            goto invalid_keyword_type;
        if (kwds2) {
            if (unlikely(PyDict_SetItem(kwds2, key, value))) goto bad;
        } else {
            goto invalid_keyword;
        }
    }
    return 0;
arg_passed_twice:
    __Pyx_RaiseDoubleKeywordsError(function_name, key);
    goto bad;
invalid_keyword_type:
    PyErr_Format(PyExc_TypeError,
        "%.200s() keywords must be strings", function_name);
    goto bad;
invalid_keyword:
    PyErr_Format(PyExc_TypeError,
    #if PY_MAJOR_VERSION < 3
        "%.200s() got an unexpected keyword argument '%.200s'",
        function_name, PyString_AsString(key));
    #else
        "%s() got an unexpected keyword argument '%U'",
        function_name, key);
    #endif
bad:
    return -1;
}

/* ArgTypeTest */
static void __Pyx_RaiseArgumentTypeInvalid(const char* name, PyObject *obj, PyTypeObject *type) {
    PyErr_Format(PyExc_TypeError,
        "Argument '%.200s' has incorrect type (expected %.200s, got %.200s)",
        name, type->tp_name, Py_TYPE(obj)->tp_name);
}
static CYTHON_INLINE int __Pyx_ArgTypeTest(PyObject *obj, PyTypeObject *type, int none_allowed,
    const char *name, int exact)
{
    if (unlikely(!type)) {
        PyErr_SetString(PyExc_SystemError, "Missing type object");
        return 0;
    }
    if (none_allowed && obj == Py_None) return 1;
    else if (exact) {
        if (likely(Py_TYPE(obj) == type)) return 1;
        #if PY_MAJOR_VERSION == 2
        else if ((type == &PyBaseString_Type) && likely(__Pyx_PyBaseString_CheckExact(obj))) return 1;
        #endif
    }
    else {
        if (likely(PyObject_TypeCheck(obj, type))) return 1;
    }
    __Pyx_RaiseArgumentTypeInvalid(name, obj, type);
    return 0;
}

/* BufferFormatCheck */
static CYTHON_INLINE int __Pyx_IsLittleEndian(void) {
  unsigned int n = 1;
  return *(unsigned char*)(&n) != 0;
}
static void __Pyx_BufFmt_Init(__Pyx_BufFmt_Context* ctx,
                              __Pyx_BufFmt_StackElem* stack,
                              __Pyx_TypeInfo* type) {
  stack[0].field = &ctx->root;
  stack[0].parent_offset = 0;
  ctx->root.type = type;
  ctx->root.name = "buffer dtype";
  ctx->root.offset = 0;
  ctx->head = stack;
  ctx->head->field = &ctx->root;
  ctx->fmt_offset = 0;
  ctx->head->parent_offset = 0;
  ctx->new_packmode = '@';
  ctx->enc_packmode = '@';
  ctx->new_count = 1;
  ctx->enc_count = 0;
  ctx->enc_type = 0;
  ctx->is_complex = 0;
  ctx->is_valid_array = 0;
  ctx->struct_alignment = 0;
  while (type->typegroup == 'S') {
    ++ctx->head;
    ctx->head->field = type->fields;
    ctx->head->parent_offset = 0;
    type = type->fields->type;
  }
}
static int __Pyx_BufFmt_ParseNumber(const char** ts) {
    int count;
    const char* t = *ts;
    if (*t < '0' || *t > '9') {
      return -1;
    } else {
        count = *t++ - '0';
        while (*t >= '0' && *t < '9') {
            count *= 10;
            count += *t++ - '0';
        }
    }
    *ts = t;
    return count;
}
static int __Pyx_BufFmt_ExpectNumber(const char **ts) {
    int number = __Pyx_BufFmt_ParseNumber(ts);
    if (number == -1)
        PyErr_Format(PyExc_ValueError,\
                     "Does not understand character buffer dtype format string ('%c')", **ts);
    return number;
}
static void __Pyx_BufFmt_RaiseUnexpectedChar(char ch) {
  PyErr_Format(PyExc_ValueError,
               "Unexpected format string character: '%c'", ch);
}
static const char* __Pyx_BufFmt_DescribeTypeChar(char ch, int is_complex) {
  switch (ch) {
    case 'c': return "'char'";
    case 'b': return "'signed char'";
    case 'B': return "'unsigned char'";
    case 'h': return "'short'";
    case 'H': return "'unsigned short'";
    case 'i': return "'int'";
    case 'I': return "'unsigned int'";
    case 'l': return "'long'";
    case 'L': return "'unsigned long'";
    case 'q': return "'long long'";
    case 'Q': return "'unsigned long long'";
    case 'f': return (is_complex ? "'complex float'" : "'float'");
    case 'd': return (is_complex ? "'complex double'" : "'double'");
    case 'g': return (is_complex ? "'complex long double'" : "'long double'");
    case 'T': return "a struct";
    case 'O': return "Python object";
    case 'P': return "a pointer";
    case 's': case 'p': return "a string";
    case 0: return "end";
    default: return "unparseable format string";
  }
}
static size_t __Pyx_BufFmt_TypeCharToStandardSize(char ch, int is_complex) {
  switch (ch) {
    case '?': case 'c': case 'b': case 'B': case 's': case 'p': return 1;
    case 'h': case 'H': return 2;
    case 'i': case 'I': case 'l': case 'L': return 4;
    case 'q': case 'Q': return 8;
    case 'f': return (is_complex ? 8 : 4);
    case 'd': return (is_complex ? 16 : 8);
    case 'g': {
      PyErr_SetString(PyExc_ValueError, "Python does not define a standard format string size for long double ('g')..");
      return 0;
    }
    case 'O': case 'P': return sizeof(void*);
    default:
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
}
static size_t __Pyx_BufFmt_TypeCharToNativeSize(char ch, int is_complex) {
  switch (ch) {
    case 'c': case 'b': case 'B': case 's': case 'p': return 1;
    case 'h': case 'H': return sizeof(short);
    case 'i': case 'I': return sizeof(int);
    case 'l': case 'L': return sizeof(long);
    #ifdef HAVE_LONG_LONG
    case 'q': case 'Q': return sizeof(PY_LONG_LONG);
    #endif
    case 'f': return sizeof(float) * (is_complex ? 2 : 1);
    case 'd': return sizeof(double) * (is_complex ? 2 : 1);
    case 'g': return sizeof(long double) * (is_complex ? 2 : 1);
    case 'O': case 'P': return sizeof(void*);
    default: {
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
  }
}
typedef struct { char c; short x; } __Pyx_st_short;
typedef struct { char c; int x; } __Pyx_st_int;
typedef struct { char c; long x; } __Pyx_st_long;
typedef struct { char c; float x; } __Pyx_st_float;
typedef struct { char c; double x; } __Pyx_st_double;
typedef struct { char c; long double x; } __Pyx_st_longdouble;
typedef struct { char c; void *x; } __Pyx_st_void_p;
#ifdef HAVE_LONG_LONG
typedef struct { char c; PY_LONG_LONG x; } __Pyx_st_longlong;
#endif
static size_t __Pyx_BufFmt_TypeCharToAlignment(char ch, CYTHON_UNUSED int is_complex) {
  switch (ch) {
    case '?': case 'c': case 'b': case 'B': case 's': case 'p': return 1;
    case 'h': case 'H': return sizeof(__Pyx_st_short) - sizeof(short);
    case 'i': case 'I': return sizeof(__Pyx_st_int) - sizeof(int);
    case 'l': case 'L': return sizeof(__Pyx_st_long) - sizeof(long);
#ifdef HAVE_LONG_LONG
    case 'q': case 'Q': return sizeof(__Pyx_st_longlong) - sizeof(PY_LONG_LONG);
#endif
    case 'f': return sizeof(__Pyx_st_float) - sizeof(float);
    case 'd': return sizeof(__Pyx_st_double) - sizeof(double);
    case 'g': return sizeof(__Pyx_st_longdouble) - sizeof(long double);
    case 'P': case 'O': return sizeof(__Pyx_st_void_p) - sizeof(void*);
    default:
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
}
/* These are for computing the padding at the end of the struct to align
   on the first member of the struct. This will probably the same as above,
   but we don't have any guarantees.
 */
typedef struct { short x; char c; } __Pyx_pad_short;
typedef struct { int x; char c; } __Pyx_pad_int;
typedef struct { long x; char c; } __Pyx_pad_long;
typedef struct { float x; char c; } __Pyx_pad_float;
typedef struct { double x; char c; } __Pyx_pad_double;
typedef struct { long double x; char c; } __Pyx_pad_longdouble;
typedef struct { void *x; char c; } __Pyx_pad_void_p;
#ifdef HAVE_LONG_LONG
typedef struct { PY_LONG_LONG x; char c; } __Pyx_pad_longlong;
#endif
static size_t __Pyx_BufFmt_TypeCharToPadding(char ch, CYTHON_UNUSED int is_complex) {
  switch (ch) {
    case '?': case 'c': case 'b': case 'B': case 's': case 'p': return 1;
    case 'h': case 'H': return sizeof(__Pyx_pad_short) - sizeof(short);
    case 'i': case 'I': return sizeof(__Pyx_pad_int) - sizeof(int);
    case 'l': case 'L': return sizeof(__Pyx_pad_long) - sizeof(long);
#ifdef HAVE_LONG_LONG
    case 'q': case 'Q': return sizeof(__Pyx_pad_longlong) - sizeof(PY_LONG_LONG);
#endif
    case 'f': return sizeof(__Pyx_pad_float) - sizeof(float);
    case 'd': return sizeof(__Pyx_pad_double) - sizeof(double);
    case 'g': return sizeof(__Pyx_pad_longdouble) - sizeof(long double);
    case 'P': case 'O': return sizeof(__Pyx_pad_void_p) - sizeof(void*);
    default:
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
}
static char __Pyx_BufFmt_TypeCharToGroup(char ch, int is_complex) {
  switch (ch) {
    case 'c':
        return 'H';
    case 'b': case 'h': case 'i':
    case 'l': case 'q': case 's': case 'p':
        return 'I';
    case 'B': case 'H': case 'I': case 'L': case 'Q':
        return 'U';
    case 'f': case 'd': case 'g':
        return (is_complex ? 'C' : 'R');
    case 'O':
        return 'O';
    case 'P':
        return 'P';
    default: {
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
  }
}
static void __Pyx_BufFmt_RaiseExpected(__Pyx_BufFmt_Context* ctx) {
  if (ctx->head == NULL || ctx->head->field == &ctx->root) {
    const char* expected;
    const char* quote;
    if (ctx->head == NULL) {
      expected = "end";
      quote = "";
    } else {
      expected = ctx->head->field->type->name;
      quote = "'";
    }
    PyErr_Format(PyExc_ValueError,
                 "Buffer dtype mismatch, expected %s%s%s but got %s",
                 quote, expected, quote,
                 __Pyx_BufFmt_DescribeTypeChar(ctx->enc_type, ctx->is_complex));
  } else {
    __Pyx_StructField* field = ctx->head->field;
    __Pyx_StructField* parent = (ctx->head - 1)->field;
    PyErr_Format(PyExc_ValueError,
                 "Buffer dtype mismatch, expected '%s' but got %s in '%s.%s'",
                 field->type->name, __Pyx_BufFmt_DescribeTypeChar(ctx->enc_type, ctx->is_complex),
                 parent->type->name, field->name);
  }
}
static int __Pyx_BufFmt_ProcessTypeChunk(__Pyx_BufFmt_Context* ctx) {
  char group;
  size_t size, offset, arraysize = 1;
  if (ctx->enc_type == 0) return 0;
  if (ctx->head->field->type->arraysize[0]) {
    int i, ndim = 0;
    if (ctx->enc_type == 's' || ctx->enc_type == 'p') {
        ctx->is_valid_array = ctx->head->field->type->ndim == 1;
        ndim = 1;
        if (ctx->enc_count != ctx->head->field->type->arraysize[0]) {
            PyErr_Format(PyExc_ValueError,
                         "Expected a dimension of size %zu, got %zu",
                         ctx->head->field->type->arraysize[0], ctx->enc_count);
            return -1;
        }
    }
    if (!ctx->is_valid_array) {
      PyErr_Format(PyExc_ValueError, "Expected %d dimensions, got %d",
                   ctx->head->field->type->ndim, ndim);
      return -1;
    }
    for (i = 0; i < ctx->head->field->type->ndim; i++) {
      arraysize *= ctx->head->field->type->arraysize[i];
    }
    ctx->is_valid_array = 0;
    ctx->enc_count = 1;
  }
  group = __Pyx_BufFmt_TypeCharToGroup(ctx->enc_type, ctx->is_complex);
  do {
    __Pyx_StructField* field = ctx->head->field;
    __Pyx_TypeInfo* type = field->type;
    if (ctx->enc_packmode == '@' || ctx->enc_packmode == '^') {
      size = __Pyx_BufFmt_TypeCharToNativeSize(ctx->enc_type, ctx->is_complex);
    } else {
      size = __Pyx_BufFmt_TypeCharToStandardSize(ctx->enc_type, ctx->is_complex);
    }
    if (ctx->enc_packmode == '@') {
      size_t align_at = __Pyx_BufFmt_TypeCharToAlignment(ctx->enc_type, ctx->is_complex);
      size_t align_mod_offset;
      if (align_at == 0) return -1;
      align_mod_offset = ctx->fmt_offset % align_at;
      if (align_mod_offset > 0) ctx->fmt_offset += align_at - align_mod_offset;
      if (ctx->struct_alignment == 0)
          ctx->struct_alignment = __Pyx_BufFmt_TypeCharToPadding(ctx->enc_type,
                                                                 ctx->is_complex);
    }
    if (type->size != size || type->typegroup != group) {
      if (type->typegroup == 'C' && type->fields != NULL) {
        size_t parent_offset = ctx->head->parent_offset + field->offset;
        ++ctx->head;
        ctx->head->field = type->fields;
        ctx->head->parent_offset = parent_offset;
        continue;
      }
      if ((type->typegroup == 'H' || group == 'H') && type->size == size) {
      } else {
          __Pyx_BufFmt_RaiseExpected(ctx);
          return -1;
      }
    }
    offset = ctx->head->parent_offset + field->offset;
    if (ctx->fmt_offset != offset) {
      PyErr_Format(PyExc_ValueError,
                   "Buffer dtype mismatch; next field is at offset %" CYTHON_FORMAT_SSIZE_T "d but %" CYTHON_FORMAT_SSIZE_T "d expected",
                   (Py_ssize_t)ctx->fmt_offset, (Py_ssize_t)offset);
      return -1;
    }
    ctx->fmt_offset += size;
    if (arraysize)
      ctx->fmt_offset += (arraysize - 1) * size;
    --ctx->enc_count;
    while (1) {
      if (field == &ctx->root) {
        ctx->head = NULL;
        if (ctx->enc_count != 0) {
          __Pyx_BufFmt_RaiseExpected(ctx);
          return -1;
        }
        break;
      }
      ctx->head->field = ++field;
      if (field->type == NULL) {
        --ctx->head;
        field = ctx->head->field;
        continue;
      } else if (field->type->typegroup == 'S') {
        size_t parent_offset = ctx->head->parent_offset + field->offset;
        if (field->type->fields->type == NULL) continue;
        field = field->type->fields;
        ++ctx->head;
        ctx->head->field = field;
        ctx->head->parent_offset = parent_offset;
        break;
      } else {
        break;
      }
    }
  } while (ctx->enc_count);
  ctx->enc_type = 0;
  ctx->is_complex = 0;
  return 0;
}
static CYTHON_INLINE PyObject *
__pyx_buffmt_parse_array(__Pyx_BufFmt_Context* ctx, const char** tsp)
{
    const char *ts = *tsp;
    int i = 0, number;
    int ndim = ctx->head->field->type->ndim;
;
    ++ts;
    if (ctx->new_count != 1) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot handle repeated arrays in format string");
        return NULL;
    }
    if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
    while (*ts && *ts != ')') {
        switch (*ts) {
            case ' ': case '\f': case '\r': case '\n': case '\t': case '\v':  continue;
            default:  break;
        }
        number = __Pyx_BufFmt_ExpectNumber(&ts);
        if (number == -1) return NULL;
        if (i < ndim && (size_t) number != ctx->head->field->type->arraysize[i])
            return PyErr_Format(PyExc_ValueError,
                        "Expected a dimension of size %zu, got %d",
                        ctx->head->field->type->arraysize[i], number);
        if (*ts != ',' && *ts != ')')
            return PyErr_Format(PyExc_ValueError,
                                "Expected a comma in format string, got '%c'", *ts);
        if (*ts == ',') ts++;
        i++;
    }
    if (i != ndim)
        return PyErr_Format(PyExc_ValueError, "Expected %d dimension(s), got %d",
                            ctx->head->field->type->ndim, i);
    if (!*ts) {
        PyErr_SetString(PyExc_ValueError,
                        "Unexpected end of format string, expected ')'");
        return NULL;
    }
    ctx->is_valid_array = 1;
    ctx->new_count = 1;
    *tsp = ++ts;
    return Py_None;
}
static const char* __Pyx_BufFmt_CheckString(__Pyx_BufFmt_Context* ctx, const char* ts) {
  int got_Z = 0;
  while (1) {
    switch(*ts) {
      case 0:
        if (ctx->enc_type != 0 && ctx->head == NULL) {
          __Pyx_BufFmt_RaiseExpected(ctx);
          return NULL;
        }
        if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
        if (ctx->head != NULL) {
          __Pyx_BufFmt_RaiseExpected(ctx);
          return NULL;
        }
        return ts;
      case ' ':
      case '\r':
      case '\n':
        ++ts;
        break;
      case '<':
        if (!__Pyx_IsLittleEndian()) {
          PyErr_SetString(PyExc_ValueError, "Little-endian buffer not supported on big-endian compiler");
          return NULL;
        }
        ctx->new_packmode = '=';
        ++ts;
        break;
      case '>':
      case '!':
        if (__Pyx_IsLittleEndian()) {
          PyErr_SetString(PyExc_ValueError, "Big-endian buffer not supported on little-endian compiler");
          return NULL;
        }
        ctx->new_packmode = '=';
        ++ts;
        break;
      case '=':
      case '@':
      case '^':
        ctx->new_packmode = *ts++;
        break;
      case 'T':
        {
          const char* ts_after_sub;
          size_t i, struct_count = ctx->new_count;
          size_t struct_alignment = ctx->struct_alignment;
          ctx->new_count = 1;
          ++ts;
          if (*ts != '{') {
            PyErr_SetString(PyExc_ValueError, "Buffer acquisition: Expected '{' after 'T'");
            return NULL;
          }
          if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
          ctx->enc_type = 0;
          ctx->enc_count = 0;
          ctx->struct_alignment = 0;
          ++ts;
          ts_after_sub = ts;
          for (i = 0; i != struct_count; ++i) {
            ts_after_sub = __Pyx_BufFmt_CheckString(ctx, ts);
            if (!ts_after_sub) return NULL;
          }
          ts = ts_after_sub;
          if (struct_alignment) ctx->struct_alignment = struct_alignment;
        }
        break;
      case '}':
        {
          size_t alignment = ctx->struct_alignment;
          ++ts;
          if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
          ctx->enc_type = 0;
          if (alignment && ctx->fmt_offset % alignment) {
            ctx->fmt_offset += alignment - (ctx->fmt_offset % alignment);
          }
        }
        return ts;
      case 'x':
        if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
        ctx->fmt_offset += ctx->new_count;
        ctx->new_count = 1;
        ctx->enc_count = 0;
        ctx->enc_type = 0;
        ctx->enc_packmode = ctx->new_packmode;
        ++ts;
        break;
      case 'Z':
        got_Z = 1;
        ++ts;
        if (*ts != 'f' && *ts != 'd' && *ts != 'g') {
          __Pyx_BufFmt_RaiseUnexpectedChar('Z');
          return NULL;
        }
      case 'c': case 'b': case 'B': case 'h': case 'H': case 'i': case 'I':
      case 'l': case 'L': case 'q': case 'Q':
      case 'f': case 'd': case 'g':
      case 'O': case 'p':
        if (ctx->enc_type == *ts && got_Z == ctx->is_complex &&
            ctx->enc_packmode == ctx->new_packmode) {
          ctx->enc_count += ctx->new_count;
          ctx->new_count = 1;
          got_Z = 0;
          ++ts;
          break;
        }
      case 's':
        if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
        ctx->enc_count = ctx->new_count;
        ctx->enc_packmode = ctx->new_packmode;
        ctx->enc_type = *ts;
        ctx->is_complex = got_Z;
        ++ts;
        ctx->new_count = 1;
        got_Z = 0;
        break;
      case ':':
        ++ts;
        while(*ts != ':') ++ts;
        ++ts;
        break;
      case '(':
        if (!__pyx_buffmt_parse_array(ctx, &ts)) return NULL;
        break;
      default:
        {
          int number = __Pyx_BufFmt_ExpectNumber(&ts);
          if (number == -1) return NULL;
          ctx->new_count = (size_t)number;
        }
    }
  }
}
static CYTHON_INLINE void __Pyx_ZeroBuffer(Py_buffer* buf) {
  buf->buf = NULL;
  buf->obj = NULL;
  buf->strides = __Pyx_zeros;
  buf->shape = __Pyx_zeros;
  buf->suboffsets = __Pyx_minusones;
}
static CYTHON_INLINE int __Pyx_GetBufferAndValidate(
        Py_buffer* buf, PyObject* obj,  __Pyx_TypeInfo* dtype, int flags,
        int nd, int cast, __Pyx_BufFmt_StackElem* stack)
{
  if (obj == Py_None || obj == NULL) {
    __Pyx_ZeroBuffer(buf);
    return 0;
  }
  buf->buf = NULL;
  if (__Pyx_GetBuffer(obj, buf, flags) == -1) goto fail;
  if (buf->ndim != nd) {
    PyErr_Format(PyExc_ValueError,
                 "Buffer has wrong number of dimensions (expected %d, got %d)",
                 nd, buf->ndim);
    goto fail;
  }
  if (!cast) {
    __Pyx_BufFmt_Context ctx;
    __Pyx_BufFmt_Init(&ctx, stack, dtype);
    if (!__Pyx_BufFmt_CheckString(&ctx, buf->format)) goto fail;
  }
  if ((unsigned)buf->itemsize != dtype->size) {
    PyErr_Format(PyExc_ValueError,
      "Item size of buffer (%" CYTHON_FORMAT_SSIZE_T "d byte%s) does not match size of '%s' (%" CYTHON_FORMAT_SSIZE_T "d byte%s)",
      buf->itemsize, (buf->itemsize > 1) ? "s" : "",
      dtype->name, (Py_ssize_t)dtype->size, (dtype->size > 1) ? "s" : "");
    goto fail;
  }
  if (buf->suboffsets == NULL) buf->suboffsets = __Pyx_minusones;
  return 0;
fail:;
  __Pyx_ZeroBuffer(buf);
  return -1;
}
static CYTHON_INLINE void __Pyx_SafeReleaseBuffer(Py_buffer* info) {
  if (info->buf == NULL) return;
  if (info->suboffsets == __Pyx_minusones) info->suboffsets = NULL;
  __Pyx_ReleaseBuffer(info);
}

/* ExtTypeTest */
  static CYTHON_INLINE int __Pyx_TypeTest(PyObject *obj, PyTypeObject *type) {
    if (unlikely(!type)) {
        PyErr_SetString(PyExc_SystemError, "Missing type object");
        return 0;
    }
    if (likely(PyObject_TypeCheck(obj, type)))
        return 1;
    PyErr_Format(PyExc_TypeError, "Cannot convert %.200s to %.200s",
                 Py_TYPE(obj)->tp_name, type->tp_name);
    return 0;
}

/* PyIntBinop */
  #if !CYTHON_COMPILING_IN_PYPY
static PyObject* __Pyx_PyInt_AddObjC(PyObject *op1, PyObject *op2, CYTHON_UNUSED long intval, CYTHON_UNUSED int inplace) {
    #if PY_MAJOR_VERSION < 3
    if (likely(PyInt_CheckExact(op1))) {
        const long b = intval;
        long x;
        long a = PyInt_AS_LONG(op1);
            x = (long)((unsigned long)a + b);
            if (likely((x^a) >= 0 || (x^b) >= 0))
                return PyInt_FromLong(x);
            return PyLong_Type.tp_as_number->nb_add(op1, op2);
    }
    #endif
    #if CYTHON_USE_PYLONG_INTERNALS
    if (likely(PyLong_CheckExact(op1))) {
        const long b = intval;
        long a, x;
#ifdef HAVE_LONG_LONG
        const PY_LONG_LONG llb = intval;
        PY_LONG_LONG lla, llx;
#endif
        const digit* digits = ((PyLongObject*)op1)->ob_digit;
        const Py_ssize_t size = Py_SIZE(op1);
        if (likely(__Pyx_sst_abs(size) <= 1)) {
            a = likely(size) ? digits[0] : 0;
            if (size == -1) a = -a;
        } else {
            switch (size) {
                case -2:
                    if (8 * sizeof(long) - 1 > 2 * PyLong_SHIFT) {
                        a = -(long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
#ifdef HAVE_LONG_LONG
                    } else if (8 * sizeof(PY_LONG_LONG) - 1 > 2 * PyLong_SHIFT) {
                        lla = -(PY_LONG_LONG) (((((unsigned PY_LONG_LONG)digits[1]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[0]));
                        goto long_long;
#endif
                    }
                case 2:
                    if (8 * sizeof(long) - 1 > 2 * PyLong_SHIFT) {
                        a = (long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
#ifdef HAVE_LONG_LONG
                    } else if (8 * sizeof(PY_LONG_LONG) - 1 > 2 * PyLong_SHIFT) {
                        lla = (PY_LONG_LONG) (((((unsigned PY_LONG_LONG)digits[1]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[0]));
                        goto long_long;
#endif
                    }
                case -3:
                    if (8 * sizeof(long) - 1 > 3 * PyLong_SHIFT) {
                        a = -(long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
#ifdef HAVE_LONG_LONG
                    } else if (8 * sizeof(PY_LONG_LONG) - 1 > 3 * PyLong_SHIFT) {
                        lla = -(PY_LONG_LONG) (((((((unsigned PY_LONG_LONG)digits[2]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[1]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[0]));
                        goto long_long;
#endif
                    }
                case 3:
                    if (8 * sizeof(long) - 1 > 3 * PyLong_SHIFT) {
                        a = (long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
#ifdef HAVE_LONG_LONG
                    } else if (8 * sizeof(PY_LONG_LONG) - 1 > 3 * PyLong_SHIFT) {
                        lla = (PY_LONG_LONG) (((((((unsigned PY_LONG_LONG)digits[2]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[1]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[0]));
                        goto long_long;
#endif
                    }
                case -4:
                    if (8 * sizeof(long) - 1 > 4 * PyLong_SHIFT) {
                        a = -(long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
#ifdef HAVE_LONG_LONG
                    } else if (8 * sizeof(PY_LONG_LONG) - 1 > 4 * PyLong_SHIFT) {
                        lla = -(PY_LONG_LONG) (((((((((unsigned PY_LONG_LONG)digits[3]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[2]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[1]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[0]));
                        goto long_long;
#endif
                    }
                case 4:
                    if (8 * sizeof(long) - 1 > 4 * PyLong_SHIFT) {
                        a = (long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
#ifdef HAVE_LONG_LONG
                    } else if (8 * sizeof(PY_LONG_LONG) - 1 > 4 * PyLong_SHIFT) {
                        lla = (PY_LONG_LONG) (((((((((unsigned PY_LONG_LONG)digits[3]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[2]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[1]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[0]));
                        goto long_long;
#endif
                    }
                default: return PyLong_Type.tp_as_number->nb_add(op1, op2);
            }
        }
                x = a + b;
            return PyLong_FromLong(x);
#ifdef HAVE_LONG_LONG
        long_long:
                llx = lla + llb;
            return PyLong_FromLongLong(llx);
#endif
        
        
    }
    #endif
    if (PyFloat_CheckExact(op1)) {
        const long b = intval;
        double a = PyFloat_AS_DOUBLE(op1);
            double result;
            PyFPE_START_PROTECT("add", return NULL)
            result = ((double)a) + (double)b;
            PyFPE_END_PROTECT(result)
            return PyFloat_FromDouble(result);
    }
    return (inplace ? PyNumber_InPlaceAdd : PyNumber_Add)(op1, op2);
}
#endif

/* PyCFunctionFastCall */
  #if CYTHON_FAST_PYCCALL
static CYTHON_INLINE PyObject * __Pyx_PyCFunction_FastCall(PyObject *func_obj, PyObject **args, Py_ssize_t nargs) {
    PyCFunctionObject *func = (PyCFunctionObject*)func_obj;
    PyCFunction meth = PyCFunction_GET_FUNCTION(func);
    PyObject *self = PyCFunction_GET_SELF(func);
    assert(PyCFunction_Check(func));
    assert(METH_FASTCALL == (PyCFunction_GET_FLAGS(func) & ~(METH_CLASS | METH_STATIC | METH_COEXIST)));
    assert(nargs >= 0);
    assert(nargs == 0 || args != NULL);
    /* _PyCFunction_FastCallDict() must not be called with an exception set,
       because it may clear it (directly or indirectly) and so the
       caller loses its exception */
    assert(!PyErr_Occurred());
    return (*((__Pyx_PyCFunctionFast)meth)) (self, args, nargs, NULL);
}
#endif  // CYTHON_FAST_PYCCALL

/* PyFunctionFastCall */
  #if CYTHON_FAST_PYCALL
#include "frameobject.h"
static PyObject* __Pyx_PyFunction_FastCallNoKw(PyCodeObject *co, PyObject **args, Py_ssize_t na,
                                               PyObject *globals) {
    PyFrameObject *f;
    PyThreadState *tstate = PyThreadState_GET();
    PyObject **fastlocals;
    Py_ssize_t i;
    PyObject *result;
    assert(globals != NULL);
    /* XXX Perhaps we should create a specialized
       PyFrame_New() that doesn't take locals, but does
       take builtins without sanity checking them.
       */
    assert(tstate != NULL);
    f = PyFrame_New(tstate, co, globals, NULL);
    if (f == NULL) {
        return NULL;
    }
    fastlocals = f->f_localsplus;
    for (i = 0; i < na; i++) {
        Py_INCREF(*args);
        fastlocals[i] = *args++;
    }
    result = PyEval_EvalFrameEx(f,0);
    ++tstate->recursion_depth;
    Py_DECREF(f);
    --tstate->recursion_depth;
    return result;
}
#if 1 || PY_VERSION_HEX < 0x030600B1
static PyObject *__Pyx_PyFunction_FastCallDict(PyObject *func, PyObject **args, int nargs, PyObject *kwargs) {
    PyCodeObject *co = (PyCodeObject *)PyFunction_GET_CODE(func);
    PyObject *globals = PyFunction_GET_GLOBALS(func);
    PyObject *argdefs = PyFunction_GET_DEFAULTS(func);
    PyObject *closure;
#if PY_MAJOR_VERSION >= 3
    PyObject *kwdefs;
#endif
    PyObject *kwtuple, **k;
    PyObject **d;
    Py_ssize_t nd;
    Py_ssize_t nk;
    PyObject *result;
    assert(kwargs == NULL || PyDict_Check(kwargs));
    nk = kwargs ? PyDict_Size(kwargs) : 0;
    if (Py_EnterRecursiveCall((char*)" while calling a Python object")) {
        return NULL;
    }
    if (
#if PY_MAJOR_VERSION >= 3
            co->co_kwonlyargcount == 0 &&
#endif
            likely(kwargs == NULL || nk == 0) &&
            co->co_flags == (CO_OPTIMIZED | CO_NEWLOCALS | CO_NOFREE)) {
        if (argdefs == NULL && co->co_argcount == nargs) {
            result = __Pyx_PyFunction_FastCallNoKw(co, args, nargs, globals);
            goto done;
        }
        else if (nargs == 0 && argdefs != NULL
                 && co->co_argcount == Py_SIZE(argdefs)) {
            /* function called with no arguments, but all parameters have
               a default value: use default values as arguments .*/
            args = &PyTuple_GET_ITEM(argdefs, 0);
            result =__Pyx_PyFunction_FastCallNoKw(co, args, Py_SIZE(argdefs), globals);
            goto done;
        }
    }
    if (kwargs != NULL) {
        Py_ssize_t pos, i;
        kwtuple = PyTuple_New(2 * nk);
        if (kwtuple == NULL) {
            result = NULL;
            goto done;
        }
        k = &PyTuple_GET_ITEM(kwtuple, 0);
        pos = i = 0;
        while (PyDict_Next(kwargs, &pos, &k[i], &k[i+1])) {
            Py_INCREF(k[i]);
            Py_INCREF(k[i+1]);
            i += 2;
        }
        nk = i / 2;
    }
    else {
        kwtuple = NULL;
        k = NULL;
    }
    closure = PyFunction_GET_CLOSURE(func);
#if PY_MAJOR_VERSION >= 3
    kwdefs = PyFunction_GET_KW_DEFAULTS(func);
#endif
    if (argdefs != NULL) {
        d = &PyTuple_GET_ITEM(argdefs, 0);
        nd = Py_SIZE(argdefs);
    }
    else {
        d = NULL;
        nd = 0;
    }
#if PY_MAJOR_VERSION >= 3
    result = PyEval_EvalCodeEx((PyObject*)co, globals, (PyObject *)NULL,
                               args, nargs,
                               k, (int)nk,
                               d, (int)nd, kwdefs, closure);
#else
    result = PyEval_EvalCodeEx(co, globals, (PyObject *)NULL,
                               args, nargs,
                               k, (int)nk,
                               d, (int)nd, closure);
#endif
    Py_XDECREF(kwtuple);
done:
    Py_LeaveRecursiveCall();
    return result;
}
#endif  // CPython < 3.6
#endif  // CYTHON_FAST_PYCALL

/* PyObjectCall */
  #if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_Call(PyObject *func, PyObject *arg, PyObject *kw) {
    PyObject *result;
    ternaryfunc call = func->ob_type->tp_call;
    if (unlikely(!call))
        return PyObject_Call(func, arg, kw);
    if (unlikely(Py_EnterRecursiveCall((char*)" while calling a Python object")))
        return NULL;
    result = (*call)(func, arg, kw);
    Py_LeaveRecursiveCall();
    if (unlikely(!result) && unlikely(!PyErr_Occurred())) {
        PyErr_SetString(
            PyExc_SystemError,
            "NULL result without error in PyObject_Call");
    }
    return result;
}
#endif

/* PyObjectCallMethO */
  #if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_CallMethO(PyObject *func, PyObject *arg) {
    PyObject *self, *result;
    PyCFunction cfunc;
    cfunc = PyCFunction_GET_FUNCTION(func);
    self = PyCFunction_GET_SELF(func);
    if (unlikely(Py_EnterRecursiveCall((char*)" while calling a Python object")))
        return NULL;
    result = cfunc(self, arg);
    Py_LeaveRecursiveCall();
    if (unlikely(!result) && unlikely(!PyErr_Occurred())) {
        PyErr_SetString(
            PyExc_SystemError,
            "NULL result without error in PyObject_Call");
    }
    return result;
}
#endif

/* PyObjectCallOneArg */
  #if CYTHON_COMPILING_IN_CPYTHON
static PyObject* __Pyx__PyObject_CallOneArg(PyObject *func, PyObject *arg) {
    PyObject *result;
    PyObject *args = PyTuple_New(1);
    if (unlikely(!args)) return NULL;
    Py_INCREF(arg);
    PyTuple_SET_ITEM(args, 0, arg);
    result = __Pyx_PyObject_Call(func, args, NULL);
    Py_DECREF(args);
    return result;
}
static CYTHON_INLINE PyObject* __Pyx_PyObject_CallOneArg(PyObject *func, PyObject *arg) {
#if CYTHON_FAST_PYCALL
    if (PyFunction_Check(func)) {
        return __Pyx_PyFunction_FastCall(func, &arg, 1);
    }
#endif
#ifdef __Pyx_CyFunction_USED
    if (likely(PyCFunction_Check(func) || PyObject_TypeCheck(func, __pyx_CyFunctionType))) {
#else
    if (likely(PyCFunction_Check(func))) {
#endif
        if (likely(PyCFunction_GET_FLAGS(func) & METH_O)) {
            return __Pyx_PyObject_CallMethO(func, arg);
#if CYTHON_FAST_PYCCALL
        } else if (PyCFunction_GET_FLAGS(func) & METH_FASTCALL) {
            return __Pyx_PyCFunction_FastCall(func, &arg, 1);
#endif
        }
    }
    return __Pyx__PyObject_CallOneArg(func, arg);
}
#else
static CYTHON_INLINE PyObject* __Pyx_PyObject_CallOneArg(PyObject *func, PyObject *arg) {
    PyObject *result;
    PyObject *args = PyTuple_Pack(1, arg);
    if (unlikely(!args)) return NULL;
    result = __Pyx_PyObject_Call(func, args, NULL);
    Py_DECREF(args);
    return result;
}
#endif

/* PyObjectCallNoArg */
    #if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_CallNoArg(PyObject *func) {
#if CYTHON_FAST_PYCALL
    if (PyFunction_Check(func)) {
        return __Pyx_PyFunction_FastCall(func, NULL, 0);
    }
#endif
#ifdef __Pyx_CyFunction_USED
    if (likely(PyCFunction_Check(func) || PyObject_TypeCheck(func, __pyx_CyFunctionType))) {
#else
    if (likely(PyCFunction_Check(func))) {
#endif
        if (likely(PyCFunction_GET_FLAGS(func) & METH_NOARGS)) {
            return __Pyx_PyObject_CallMethO(func, NULL);
        }
    }
    return __Pyx_PyObject_Call(func, __pyx_empty_tuple, NULL);
}
#endif

/* GetModuleGlobalName */
      static CYTHON_INLINE PyObject *__Pyx_GetModuleGlobalName(PyObject *name) {
    PyObject *result;
#if !CYTHON_AVOID_BORROWED_REFS
    result = PyDict_GetItem(__pyx_d, name);
    if (likely(result)) {
        Py_INCREF(result);
    } else {
#else
    result = PyObject_GetItem(__pyx_d, name);
    if (!result) {
        PyErr_Clear();
#endif
        result = __Pyx_GetBuiltinName(name);
    }
    return result;
}

/* BufferIndexError */
        static void __Pyx_RaiseBufferIndexError(int axis) {
  PyErr_Format(PyExc_IndexError,
     "Out of bounds on buffer access (axis %d)", axis);
}

/* PyErrFetchRestore */
        #if CYTHON_FAST_THREAD_STATE
static CYTHON_INLINE void __Pyx_ErrRestoreInState(PyThreadState *tstate, PyObject *type, PyObject *value, PyObject *tb) {
    PyObject *tmp_type, *tmp_value, *tmp_tb;
    tmp_type = tstate->curexc_type;
    tmp_value = tstate->curexc_value;
    tmp_tb = tstate->curexc_traceback;
    tstate->curexc_type = type;
    tstate->curexc_value = value;
    tstate->curexc_traceback = tb;
    Py_XDECREF(tmp_type);
    Py_XDECREF(tmp_value);
    Py_XDECREF(tmp_tb);
}
static CYTHON_INLINE void __Pyx_ErrFetchInState(PyThreadState *tstate, PyObject **type, PyObject **value, PyObject **tb) {
    *type = tstate->curexc_type;
    *value = tstate->curexc_value;
    *tb = tstate->curexc_traceback;
    tstate->curexc_type = 0;
    tstate->curexc_value = 0;
    tstate->curexc_traceback = 0;
}
#endif

/* RaiseException */
        #if PY_MAJOR_VERSION < 3
static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb,
                        CYTHON_UNUSED PyObject *cause) {
    __Pyx_PyThreadState_declare
    Py_XINCREF(type);
    if (!value || value == Py_None)
        value = NULL;
    else
        Py_INCREF(value);
    if (!tb || tb == Py_None)
        tb = NULL;
    else {
        Py_INCREF(tb);
        if (!PyTraceBack_Check(tb)) {
            PyErr_SetString(PyExc_TypeError,
                "raise: arg 3 must be a traceback or None");
            goto raise_error;
        }
    }
    if (PyType_Check(type)) {
#if CYTHON_COMPILING_IN_PYPY
        if (!value) {
            Py_INCREF(Py_None);
            value = Py_None;
        }
#endif
        PyErr_NormalizeException(&type, &value, &tb);
    } else {
        if (value) {
            PyErr_SetString(PyExc_TypeError,
                "instance exception may not have a separate value");
            goto raise_error;
        }
        value = type;
        type = (PyObject*) Py_TYPE(type);
        Py_INCREF(type);
        if (!PyType_IsSubtype((PyTypeObject *)type, (PyTypeObject *)PyExc_BaseException)) {
            PyErr_SetString(PyExc_TypeError,
                "raise: exception class must be a subclass of BaseException");
            goto raise_error;
        }
    }
    __Pyx_PyThreadState_assign
    __Pyx_ErrRestore(type, value, tb);
    return;
raise_error:
    Py_XDECREF(value);
    Py_XDECREF(type);
    Py_XDECREF(tb);
    return;
}
#else
static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb, PyObject *cause) {
    PyObject* owned_instance = NULL;
    if (tb == Py_None) {
        tb = 0;
    } else if (tb && !PyTraceBack_Check(tb)) {
        PyErr_SetString(PyExc_TypeError,
            "raise: arg 3 must be a traceback or None");
        goto bad;
    }
    if (value == Py_None)
        value = 0;
    if (PyExceptionInstance_Check(type)) {
        if (value) {
            PyErr_SetString(PyExc_TypeError,
                "instance exception may not have a separate value");
            goto bad;
        }
        value = type;
        type = (PyObject*) Py_TYPE(value);
    } else if (PyExceptionClass_Check(type)) {
        PyObject *instance_class = NULL;
        if (value && PyExceptionInstance_Check(value)) {
            instance_class = (PyObject*) Py_TYPE(value);
            if (instance_class != type) {
                int is_subclass = PyObject_IsSubclass(instance_class, type);
                if (!is_subclass) {
                    instance_class = NULL;
                } else if (unlikely(is_subclass == -1)) {
                    goto bad;
                } else {
                    type = instance_class;
                }
            }
        }
        if (!instance_class) {
            PyObject *args;
            if (!value)
                args = PyTuple_New(0);
            else if (PyTuple_Check(value)) {
                Py_INCREF(value);
                args = value;
            } else
                args = PyTuple_Pack(1, value);
            if (!args)
                goto bad;
            owned_instance = PyObject_Call(type, args, NULL);
            Py_DECREF(args);
            if (!owned_instance)
                goto bad;
            value = owned_instance;
            if (!PyExceptionInstance_Check(value)) {
                PyErr_Format(PyExc_TypeError,
                             "calling %R should have returned an instance of "
                             "BaseException, not %R",
                             type, Py_TYPE(value));
                goto bad;
            }
        }
    } else {
        PyErr_SetString(PyExc_TypeError,
            "raise: exception class must be a subclass of BaseException");
        goto bad;
    }
#if PY_VERSION_HEX >= 0x03030000
    if (cause) {
#else
    if (cause && cause != Py_None) {
#endif
        PyObject *fixed_cause;
        if (cause == Py_None) {
            fixed_cause = NULL;
        } else if (PyExceptionClass_Check(cause)) {
            fixed_cause = PyObject_CallObject(cause, NULL);
            if (fixed_cause == NULL)
                goto bad;
        } else if (PyExceptionInstance_Check(cause)) {
            fixed_cause = cause;
            Py_INCREF(fixed_cause);
        } else {
            PyErr_SetString(PyExc_TypeError,
                            "exception causes must derive from "
                            "BaseException");
            goto bad;
        }
        PyException_SetCause(value, fixed_cause);
    }
    PyErr_SetObject(type, value);
    if (tb) {
#if CYTHON_COMPILING_IN_PYPY
        PyObject *tmp_type, *tmp_value, *tmp_tb;
        PyErr_Fetch(&tmp_type, &tmp_value, &tmp_tb);
        Py_INCREF(tb);
        PyErr_Restore(tmp_type, tmp_value, tb);
        Py_XDECREF(tmp_tb);
#else
        PyThreadState *tstate = PyThreadState_GET();
        PyObject* tmp_tb = tstate->curexc_traceback;
        if (tb != tmp_tb) {
            Py_INCREF(tb);
            tstate->curexc_traceback = tb;
            Py_XDECREF(tmp_tb);
        }
#endif
    }
bad:
    Py_XDECREF(owned_instance);
    return;
}
#endif

/* RaiseTooManyValuesToUnpack */
          static CYTHON_INLINE void __Pyx_RaiseTooManyValuesError(Py_ssize_t expected) {
    PyErr_Format(PyExc_ValueError,
                 "too many values to unpack (expected %" CYTHON_FORMAT_SSIZE_T "d)", expected);
}

/* RaiseNeedMoreValuesToUnpack */
          static CYTHON_INLINE void __Pyx_RaiseNeedMoreValuesError(Py_ssize_t index) {
    PyErr_Format(PyExc_ValueError,
                 "need more than %" CYTHON_FORMAT_SSIZE_T "d value%.1s to unpack",
                 index, (index == 1) ? "" : "s");
}

/* RaiseNoneIterError */
          static CYTHON_INLINE void __Pyx_RaiseNoneNotIterableError(void) {
    PyErr_SetString(PyExc_TypeError, "'NoneType' object is not iterable");
}

/* SaveResetException */
          #if CYTHON_FAST_THREAD_STATE
static CYTHON_INLINE void __Pyx__ExceptionSave(PyThreadState *tstate, PyObject **type, PyObject **value, PyObject **tb) {
    *type = tstate->exc_type;
    *value = tstate->exc_value;
    *tb = tstate->exc_traceback;
    Py_XINCREF(*type);
    Py_XINCREF(*value);
    Py_XINCREF(*tb);
}
static CYTHON_INLINE void __Pyx__ExceptionReset(PyThreadState *tstate, PyObject *type, PyObject *value, PyObject *tb) {
    PyObject *tmp_type, *tmp_value, *tmp_tb;
    tmp_type = tstate->exc_type;
    tmp_value = tstate->exc_value;
    tmp_tb = tstate->exc_traceback;
    tstate->exc_type = type;
    tstate->exc_value = value;
    tstate->exc_traceback = tb;
    Py_XDECREF(tmp_type);
    Py_XDECREF(tmp_value);
    Py_XDECREF(tmp_tb);
}
#endif

/* PyErrExceptionMatches */
          #if CYTHON_FAST_THREAD_STATE
static CYTHON_INLINE int __Pyx_PyErr_ExceptionMatchesInState(PyThreadState* tstate, PyObject* err) {
    PyObject *exc_type = tstate->curexc_type;
    if (exc_type == err) return 1;
    if (unlikely(!exc_type)) return 0;
    return PyErr_GivenExceptionMatches(exc_type, err);
}
#endif

/* GetException */
          #if CYTHON_FAST_THREAD_STATE
static int __Pyx__GetException(PyThreadState *tstate, PyObject **type, PyObject **value, PyObject **tb) {
#else
static int __Pyx_GetException(PyObject **type, PyObject **value, PyObject **tb) {
#endif
    PyObject *local_type, *local_value, *local_tb;
#if CYTHON_FAST_THREAD_STATE
    PyObject *tmp_type, *tmp_value, *tmp_tb;
    local_type = tstate->curexc_type;
    local_value = tstate->curexc_value;
    local_tb = tstate->curexc_traceback;
    tstate->curexc_type = 0;
    tstate->curexc_value = 0;
    tstate->curexc_traceback = 0;
#else
    PyErr_Fetch(&local_type, &local_value, &local_tb);
#endif
    PyErr_NormalizeException(&local_type, &local_value, &local_tb);
#if CYTHON_FAST_THREAD_STATE
    if (unlikely(tstate->curexc_type))
#else
    if (unlikely(PyErr_Occurred()))
#endif
        goto bad;
    #if PY_MAJOR_VERSION >= 3
    if (local_tb) {
        if (unlikely(PyException_SetTraceback(local_value, local_tb) < 0))
            goto bad;
    }
    #endif
    Py_XINCREF(local_tb);
    Py_XINCREF(local_type);
    Py_XINCREF(local_value);
    *type = local_type;
    *value = local_value;
    *tb = local_tb;
#if CYTHON_FAST_THREAD_STATE
    tmp_type = tstate->exc_type;
    tmp_value = tstate->exc_value;
    tmp_tb = tstate->exc_traceback;
    tstate->exc_type = local_type;
    tstate->exc_value = local_value;
    tstate->exc_traceback = local_tb;
    Py_XDECREF(tmp_type);
    Py_XDECREF(tmp_value);
    Py_XDECREF(tmp_tb);
#else
    PyErr_SetExcInfo(local_type, local_value, local_tb);
#endif
    return 0;
bad:
    *type = 0;
    *value = 0;
    *tb = 0;
    Py_XDECREF(local_type);
    Py_XDECREF(local_value);
    Py_XDECREF(local_tb);
    return -1;
}

/* Import */
            static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list, int level) {
    PyObject *empty_list = 0;
    PyObject *module = 0;
    PyObject *global_dict = 0;
    PyObject *empty_dict = 0;
    PyObject *list;
    #if PY_VERSION_HEX < 0x03030000
    PyObject *py_import;
    py_import = __Pyx_PyObject_GetAttrStr(__pyx_b, __pyx_n_s_import);
    if (!py_import)
        goto bad;
    #endif
    if (from_list)
        list = from_list;
    else {
        empty_list = PyList_New(0);
        if (!empty_list)
            goto bad;
        list = empty_list;
    }
    global_dict = PyModule_GetDict(__pyx_m);
    if (!global_dict)
        goto bad;
    empty_dict = PyDict_New();
    if (!empty_dict)
        goto bad;
    {
        #if PY_MAJOR_VERSION >= 3
        if (level == -1) {
            if (strchr(__Pyx_MODULE_NAME, '.')) {
                #if PY_VERSION_HEX < 0x03030000
                PyObject *py_level = PyInt_FromLong(1);
                if (!py_level)
                    goto bad;
                module = PyObject_CallFunctionObjArgs(py_import,
                    name, global_dict, empty_dict, list, py_level, NULL);
                Py_DECREF(py_level);
                #else
                module = PyImport_ImportModuleLevelObject(
                    name, global_dict, empty_dict, list, 1);
                #endif
                if (!module) {
                    if (!PyErr_ExceptionMatches(PyExc_ImportError))
                        goto bad;
                    PyErr_Clear();
                }
            }
            level = 0;
        }
        #endif
        if (!module) {
            #if PY_VERSION_HEX < 0x03030000
            PyObject *py_level = PyInt_FromLong(level);
            if (!py_level)
                goto bad;
            module = PyObject_CallFunctionObjArgs(py_import,
                name, global_dict, empty_dict, list, py_level, NULL);
            Py_DECREF(py_level);
            #else
            module = PyImport_ImportModuleLevelObject(
                name, global_dict, empty_dict, list, level);
            #endif
        }
    }
bad:
    #if PY_VERSION_HEX < 0x03030000
    Py_XDECREF(py_import);
    #endif
    Py_XDECREF(empty_list);
    Py_XDECREF(empty_dict);
    return module;
}

/* CodeObjectCache */
            static int __pyx_bisect_code_objects(__Pyx_CodeObjectCacheEntry* entries, int count, int code_line) {
    int start = 0, mid = 0, end = count - 1;
    if (end >= 0 && code_line > entries[end].code_line) {
        return count;
    }
    while (start < end) {
        mid = start + (end - start) / 2;
        if (code_line < entries[mid].code_line) {
            end = mid;
        } else if (code_line > entries[mid].code_line) {
             start = mid + 1;
        } else {
            return mid;
        }
    }
    if (code_line <= entries[mid].code_line) {
        return mid;
    } else {
        return mid + 1;
    }
}
static PyCodeObject *__pyx_find_code_object(int code_line) {
    PyCodeObject* code_object;
    int pos;
    if (unlikely(!code_line) || unlikely(!__pyx_code_cache.entries)) {
        return NULL;
    }
    pos = __pyx_bisect_code_objects(__pyx_code_cache.entries, __pyx_code_cache.count, code_line);
    if (unlikely(pos >= __pyx_code_cache.count) || unlikely(__pyx_code_cache.entries[pos].code_line != code_line)) {
        return NULL;
    }
    code_object = __pyx_code_cache.entries[pos].code_object;
    Py_INCREF(code_object);
    return code_object;
}
static void __pyx_insert_code_object(int code_line, PyCodeObject* code_object) {
    int pos, i;
    __Pyx_CodeObjectCacheEntry* entries = __pyx_code_cache.entries;
    if (unlikely(!code_line)) {
        return;
    }
    if (unlikely(!entries)) {
        entries = (__Pyx_CodeObjectCacheEntry*)PyMem_Malloc(64*sizeof(__Pyx_CodeObjectCacheEntry));
        if (likely(entries)) {
            __pyx_code_cache.entries = entries;
            __pyx_code_cache.max_count = 64;
            __pyx_code_cache.count = 1;
            entries[0].code_line = code_line;
            entries[0].code_object = code_object;
            Py_INCREF(code_object);
        }
        return;
    }
    pos = __pyx_bisect_code_objects(__pyx_code_cache.entries, __pyx_code_cache.count, code_line);
    if ((pos < __pyx_code_cache.count) && unlikely(__pyx_code_cache.entries[pos].code_line == code_line)) {
        PyCodeObject* tmp = entries[pos].code_object;
        entries[pos].code_object = code_object;
        Py_DECREF(tmp);
        return;
    }
    if (__pyx_code_cache.count == __pyx_code_cache.max_count) {
        int new_max = __pyx_code_cache.max_count + 64;
        entries = (__Pyx_CodeObjectCacheEntry*)PyMem_Realloc(
            __pyx_code_cache.entries, (size_t)new_max*sizeof(__Pyx_CodeObjectCacheEntry));
        if (unlikely(!entries)) {
            return;
        }
        __pyx_code_cache.entries = entries;
        __pyx_code_cache.max_count = new_max;
    }
    for (i=__pyx_code_cache.count; i>pos; i--) {
        entries[i] = entries[i-1];
    }
    entries[pos].code_line = code_line;
    entries[pos].code_object = code_object;
    __pyx_code_cache.count++;
    Py_INCREF(code_object);
}

/* AddTraceback */
            #include "compile.h"
#include "frameobject.h"
#include "traceback.h"
static PyCodeObject* __Pyx_CreateCodeObjectForTraceback(
            const char *funcname, int c_line,
            int py_line, const char *filename) {
    PyCodeObject *py_code = 0;
    PyObject *py_srcfile = 0;
    PyObject *py_funcname = 0;
    #if PY_MAJOR_VERSION < 3
    py_srcfile = PyString_FromString(filename);
    #else
    py_srcfile = PyUnicode_FromString(filename);
    #endif
    if (!py_srcfile) goto bad;
    if (c_line) {
        #if PY_MAJOR_VERSION < 3
        py_funcname = PyString_FromFormat( "%s (%s:%d)", funcname, __pyx_cfilenm, c_line);
        #else
        py_funcname = PyUnicode_FromFormat( "%s (%s:%d)", funcname, __pyx_cfilenm, c_line);
        #endif
    }
    else {
        #if PY_MAJOR_VERSION < 3
        py_funcname = PyString_FromString(funcname);
        #else
        py_funcname = PyUnicode_FromString(funcname);
        #endif
    }
    if (!py_funcname) goto bad;
    py_code = __Pyx_PyCode_New(
        0,
        0,
        0,
        0,
        0,
        __pyx_empty_bytes, /*PyObject *code,*/
        __pyx_empty_tuple, /*PyObject *consts,*/
        __pyx_empty_tuple, /*PyObject *names,*/
        __pyx_empty_tuple, /*PyObject *varnames,*/
        __pyx_empty_tuple, /*PyObject *freevars,*/
        __pyx_empty_tuple, /*PyObject *cellvars,*/
        py_srcfile,   /*PyObject *filename,*/
        py_funcname,  /*PyObject *name,*/
        py_line,
        __pyx_empty_bytes  /*PyObject *lnotab*/
    );
    Py_DECREF(py_srcfile);
    Py_DECREF(py_funcname);
    return py_code;
bad:
    Py_XDECREF(py_srcfile);
    Py_XDECREF(py_funcname);
    return NULL;
}
static void __Pyx_AddTraceback(const char *funcname, int c_line,
                               int py_line, const char *filename) {
    PyCodeObject *py_code = 0;
    PyFrameObject *py_frame = 0;
    py_code = __pyx_find_code_object(c_line ? c_line : py_line);
    if (!py_code) {
        py_code = __Pyx_CreateCodeObjectForTraceback(
            funcname, c_line, py_line, filename);
        if (!py_code) goto bad;
        __pyx_insert_code_object(c_line ? c_line : py_line, py_code);
    }
    py_frame = PyFrame_New(
        PyThreadState_GET(), /*PyThreadState *tstate,*/
        py_code,             /*PyCodeObject *code,*/
        __pyx_d,      /*PyObject *globals,*/
        0                    /*PyObject *locals*/
    );
    if (!py_frame) goto bad;
    __Pyx_PyFrame_SetLineNumber(py_frame, py_line);
    PyTraceBack_Here(py_frame);
bad:
    Py_XDECREF(py_code);
    Py_XDECREF(py_frame);
}

#if PY_MAJOR_VERSION < 3
static int __Pyx_GetBuffer(PyObject *obj, Py_buffer *view, int flags) {
    if (PyObject_CheckBuffer(obj)) return PyObject_GetBuffer(obj, view, flags);
        if (PyObject_TypeCheck(obj, __pyx_ptype_5numpy_ndarray)) return __pyx_pw_5numpy_7ndarray_1__getbuffer__(obj, view, flags);
    PyErr_Format(PyExc_TypeError, "'%.200s' does not have the buffer interface", Py_TYPE(obj)->tp_name);
    return -1;
}
static void __Pyx_ReleaseBuffer(Py_buffer *view) {
    PyObject *obj = view->obj;
    if (!obj) return;
    if (PyObject_CheckBuffer(obj)) {
        PyBuffer_Release(view);
        return;
    }
        if (PyObject_TypeCheck(obj, __pyx_ptype_5numpy_ndarray)) { __pyx_pw_5numpy_7ndarray_3__releasebuffer__(obj, view); return; }
    Py_DECREF(obj);
    view->obj = NULL;
}
#endif


            /* CIntFromPyVerify */
            #define __PYX_VERIFY_RETURN_INT(target_type, func_type, func_value)\
    __PYX__VERIFY_RETURN_INT(target_type, func_type, func_value, 0)
#define __PYX_VERIFY_RETURN_INT_EXC(target_type, func_type, func_value)\
    __PYX__VERIFY_RETURN_INT(target_type, func_type, func_value, 1)
#define __PYX__VERIFY_RETURN_INT(target_type, func_type, func_value, exc)\
    {\
        func_type value = func_value;\
        if (sizeof(target_type) < sizeof(func_type)) {\
            if (unlikely(value != (func_type) (target_type) value)) {\
                func_type zero = 0;\
                if (exc && unlikely(value == (func_type)-1 && PyErr_Occurred()))\
                    return (target_type) -1;\
                if (is_unsigned && unlikely(value < zero))\
                    goto raise_neg_overflow;\
                else\
                    goto raise_overflow;\
            }\
        }\
        return (target_type) value;\
    }

/* CIntToPy */
            static CYTHON_INLINE PyObject* __Pyx_PyInt_From_int(int value) {
    const int neg_one = (int) -1, const_zero = (int) 0;
    const int is_unsigned = neg_one > const_zero;
    if (is_unsigned) {
        if (sizeof(int) < sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(int) <= sizeof(unsigned long)) {
            return PyLong_FromUnsignedLong((unsigned long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(int) <= sizeof(unsigned PY_LONG_LONG)) {
            return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG) value);
#endif
        }
    } else {
        if (sizeof(int) <= sizeof(long)) {
            return PyInt_FromLong((long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(int) <= sizeof(PY_LONG_LONG)) {
            return PyLong_FromLongLong((PY_LONG_LONG) value);
#endif
        }
    }
    {
        int one = 1; int little = (int)*(unsigned char *)&one;
        unsigned char *bytes = (unsigned char *)&value;
        return _PyLong_FromByteArray(bytes, sizeof(int),
                                     little, !is_unsigned);
    }
}

/* CIntToPy */
            static CYTHON_INLINE PyObject* __Pyx_PyInt_From_long(long value) {
    const long neg_one = (long) -1, const_zero = (long) 0;
    const int is_unsigned = neg_one > const_zero;
    if (is_unsigned) {
        if (sizeof(long) < sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(long) <= sizeof(unsigned long)) {
            return PyLong_FromUnsignedLong((unsigned long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(long) <= sizeof(unsigned PY_LONG_LONG)) {
            return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG) value);
#endif
        }
    } else {
        if (sizeof(long) <= sizeof(long)) {
            return PyInt_FromLong((long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(long) <= sizeof(PY_LONG_LONG)) {
            return PyLong_FromLongLong((PY_LONG_LONG) value);
#endif
        }
    }
    {
        int one = 1; int little = (int)*(unsigned char *)&one;
        unsigned char *bytes = (unsigned char *)&value;
        return _PyLong_FromByteArray(bytes, sizeof(long),
                                     little, !is_unsigned);
    }
}

/* CIntToPy */
            static CYTHON_INLINE PyObject* __Pyx_PyInt_From_unsigned_int(unsigned int value) {
    const unsigned int neg_one = (unsigned int) -1, const_zero = (unsigned int) 0;
    const int is_unsigned = neg_one > const_zero;
    if (is_unsigned) {
        if (sizeof(unsigned int) < sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(unsigned int) <= sizeof(unsigned long)) {
            return PyLong_FromUnsignedLong((unsigned long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(unsigned int) <= sizeof(unsigned PY_LONG_LONG)) {
            return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG) value);
#endif
        }
    } else {
        if (sizeof(unsigned int) <= sizeof(long)) {
            return PyInt_FromLong((long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(unsigned int) <= sizeof(PY_LONG_LONG)) {
            return PyLong_FromLongLong((PY_LONG_LONG) value);
#endif
        }
    }
    {
        int one = 1; int little = (int)*(unsigned char *)&one;
        unsigned char *bytes = (unsigned char *)&value;
        return _PyLong_FromByteArray(bytes, sizeof(unsigned int),
                                     little, !is_unsigned);
    }
}

/* Declarations */
            #if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float x, float y) {
      return ::std::complex< float >(x, y);
    }
  #else
    static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float x, float y) {
      return x + y*(__pyx_t_float_complex)_Complex_I;
    }
  #endif
#else
    static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float x, float y) {
      __pyx_t_float_complex z;
      z.real = x;
      z.imag = y;
      return z;
    }
#endif

/* Arithmetic */
            #if CYTHON_CCOMPLEX
#else
    static CYTHON_INLINE int __Pyx_c_eq_float(__pyx_t_float_complex a, __pyx_t_float_complex b) {
       return (a.real == b.real) && (a.imag == b.imag);
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_sum_float(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        z.real = a.real + b.real;
        z.imag = a.imag + b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_diff_float(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        z.real = a.real - b.real;
        z.imag = a.imag - b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_prod_float(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        z.real = a.real * b.real - a.imag * b.imag;
        z.imag = a.real * b.imag + a.imag * b.real;
        return z;
    }
    #if 1
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_quot_float(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        if (b.imag == 0) {
            return __pyx_t_float_complex_from_parts(a.real / b.real, a.imag / b.real);
        } else if (fabsf(b.real) >= fabsf(b.imag)) {
            if (b.real == 0 && b.imag == 0) {
                return __pyx_t_float_complex_from_parts(a.real / b.real, a.imag / b.imag);
            } else {
                float r = b.imag / b.real;
                float s = 1.0 / (b.real + b.imag * r);
                return __pyx_t_float_complex_from_parts(
                    (a.real + a.imag * r) * s, (a.imag - a.real * r) * s);
            }
        } else {
            float r = b.real / b.imag;
            float s = 1.0 / (b.imag + b.real * r);
            return __pyx_t_float_complex_from_parts(
                (a.real * r + a.imag) * s, (a.imag * r - a.real) * s);
        }
    }
    #else
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_quot_float(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        if (b.imag == 0) {
            return __pyx_t_float_complex_from_parts(a.real / b.real, a.imag / b.real);
        } else {
            float denom = b.real * b.real + b.imag * b.imag;
            return __pyx_t_float_complex_from_parts(
                (a.real * b.real + a.imag * b.imag) / denom,
                (a.imag * b.real - a.real * b.imag) / denom);
        }
    }
    #endif
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_neg_float(__pyx_t_float_complex a) {
        __pyx_t_float_complex z;
        z.real = -a.real;
        z.imag = -a.imag;
        return z;
    }
    static CYTHON_INLINE int __Pyx_c_is_zero_float(__pyx_t_float_complex a) {
       return (a.real == 0) && (a.imag == 0);
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_conj_float(__pyx_t_float_complex a) {
        __pyx_t_float_complex z;
        z.real =  a.real;
        z.imag = -a.imag;
        return z;
    }
    #if 1
        static CYTHON_INLINE float __Pyx_c_abs_float(__pyx_t_float_complex z) {
          #if !defined(HAVE_HYPOT) || defined(_MSC_VER)
            return sqrtf(z.real*z.real + z.imag*z.imag);
          #else
            return hypotf(z.real, z.imag);
          #endif
        }
        static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_pow_float(__pyx_t_float_complex a, __pyx_t_float_complex b) {
            __pyx_t_float_complex z;
            float r, lnr, theta, z_r, z_theta;
            if (b.imag == 0 && b.real == (int)b.real) {
                if (b.real < 0) {
                    float denom = a.real * a.real + a.imag * a.imag;
                    a.real = a.real / denom;
                    a.imag = -a.imag / denom;
                    b.real = -b.real;
                }
                switch ((int)b.real) {
                    case 0:
                        z.real = 1;
                        z.imag = 0;
                        return z;
                    case 1:
                        return a;
                    case 2:
                        z = __Pyx_c_prod_float(a, a);
                        return __Pyx_c_prod_float(a, a);
                    case 3:
                        z = __Pyx_c_prod_float(a, a);
                        return __Pyx_c_prod_float(z, a);
                    case 4:
                        z = __Pyx_c_prod_float(a, a);
                        return __Pyx_c_prod_float(z, z);
                }
            }
            if (a.imag == 0) {
                if (a.real == 0) {
                    return a;
                } else if (b.imag == 0) {
                    z.real = powf(a.real, b.real);
                    z.imag = 0;
                    return z;
                } else if (a.real > 0) {
                    r = a.real;
                    theta = 0;
                } else {
                    r = -a.real;
                    theta = atan2f(0, -1);
                }
            } else {
                r = __Pyx_c_abs_float(a);
                theta = atan2f(a.imag, a.real);
            }
            lnr = logf(r);
            z_r = expf(lnr * b.real - theta * b.imag);
            z_theta = theta * b.real + lnr * b.imag;
            z.real = z_r * cosf(z_theta);
            z.imag = z_r * sinf(z_theta);
            return z;
        }
    #endif
#endif

/* Declarations */
            #if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double x, double y) {
      return ::std::complex< double >(x, y);
    }
  #else
    static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double x, double y) {
      return x + y*(__pyx_t_double_complex)_Complex_I;
    }
  #endif
#else
    static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double x, double y) {
      __pyx_t_double_complex z;
      z.real = x;
      z.imag = y;
      return z;
    }
#endif

/* Arithmetic */
            #if CYTHON_CCOMPLEX
#else
    static CYTHON_INLINE int __Pyx_c_eq_double(__pyx_t_double_complex a, __pyx_t_double_complex b) {
       return (a.real == b.real) && (a.imag == b.imag);
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_sum_double(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        z.real = a.real + b.real;
        z.imag = a.imag + b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_diff_double(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        z.real = a.real - b.real;
        z.imag = a.imag - b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_prod_double(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        z.real = a.real * b.real - a.imag * b.imag;
        z.imag = a.real * b.imag + a.imag * b.real;
        return z;
    }
    #if 1
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_quot_double(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        if (b.imag == 0) {
            return __pyx_t_double_complex_from_parts(a.real / b.real, a.imag / b.real);
        } else if (fabs(b.real) >= fabs(b.imag)) {
            if (b.real == 0 && b.imag == 0) {
                return __pyx_t_double_complex_from_parts(a.real / b.real, a.imag / b.imag);
            } else {
                double r = b.imag / b.real;
                double s = 1.0 / (b.real + b.imag * r);
                return __pyx_t_double_complex_from_parts(
                    (a.real + a.imag * r) * s, (a.imag - a.real * r) * s);
            }
        } else {
            double r = b.real / b.imag;
            double s = 1.0 / (b.imag + b.real * r);
            return __pyx_t_double_complex_from_parts(
                (a.real * r + a.imag) * s, (a.imag * r - a.real) * s);
        }
    }
    #else
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_quot_double(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        if (b.imag == 0) {
            return __pyx_t_double_complex_from_parts(a.real / b.real, a.imag / b.real);
        } else {
            double denom = b.real * b.real + b.imag * b.imag;
            return __pyx_t_double_complex_from_parts(
                (a.real * b.real + a.imag * b.imag) / denom,
                (a.imag * b.real - a.real * b.imag) / denom);
        }
    }
    #endif
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_neg_double(__pyx_t_double_complex a) {
        __pyx_t_double_complex z;
        z.real = -a.real;
        z.imag = -a.imag;
        return z;
    }
    static CYTHON_INLINE int __Pyx_c_is_zero_double(__pyx_t_double_complex a) {
       return (a.real == 0) && (a.imag == 0);
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_conj_double(__pyx_t_double_complex a) {
        __pyx_t_double_complex z;
        z.real =  a.real;
        z.imag = -a.imag;
        return z;
    }
    #if 1
        static CYTHON_INLINE double __Pyx_c_abs_double(__pyx_t_double_complex z) {
          #if !defined(HAVE_HYPOT) || defined(_MSC_VER)
            return sqrt(z.real*z.real + z.imag*z.imag);
          #else
            return hypot(z.real, z.imag);
          #endif
        }
        static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_pow_double(__pyx_t_double_complex a, __pyx_t_double_complex b) {
            __pyx_t_double_complex z;
            double r, lnr, theta, z_r, z_theta;
            if (b.imag == 0 && b.real == (int)b.real) {
                if (b.real < 0) {
                    double denom = a.real * a.real + a.imag * a.imag;
                    a.real = a.real / denom;
                    a.imag = -a.imag / denom;
                    b.real = -b.real;
                }
                switch ((int)b.real) {
                    case 0:
                        z.real = 1;
                        z.imag = 0;
                        return z;
                    case 1:
                        return a;
                    case 2:
                        z = __Pyx_c_prod_double(a, a);
                        return __Pyx_c_prod_double(a, a);
                    case 3:
                        z = __Pyx_c_prod_double(a, a);
                        return __Pyx_c_prod_double(z, a);
                    case 4:
                        z = __Pyx_c_prod_double(a, a);
                        return __Pyx_c_prod_double(z, z);
                }
            }
            if (a.imag == 0) {
                if (a.real == 0) {
                    return a;
                } else if (b.imag == 0) {
                    z.real = pow(a.real, b.real);
                    z.imag = 0;
                    return z;
                } else if (a.real > 0) {
                    r = a.real;
                    theta = 0;
                } else {
                    r = -a.real;
                    theta = atan2(0, -1);
                }
            } else {
                r = __Pyx_c_abs_double(a);
                theta = atan2(a.imag, a.real);
            }
            lnr = log(r);
            z_r = exp(lnr * b.real - theta * b.imag);
            z_theta = theta * b.real + lnr * b.imag;
            z.real = z_r * cos(z_theta);
            z.imag = z_r * sin(z_theta);
            return z;
        }
    #endif
#endif

/* CIntToPy */
            static CYTHON_INLINE PyObject* __Pyx_PyInt_From_enum__NPY_TYPES(enum NPY_TYPES value) {
    const enum NPY_TYPES neg_one = (enum NPY_TYPES) -1, const_zero = (enum NPY_TYPES) 0;
    const int is_unsigned = neg_one > const_zero;
    if (is_unsigned) {
        if (sizeof(enum NPY_TYPES) < sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(enum NPY_TYPES) <= sizeof(unsigned long)) {
            return PyLong_FromUnsignedLong((unsigned long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(enum NPY_TYPES) <= sizeof(unsigned PY_LONG_LONG)) {
            return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG) value);
#endif
        }
    } else {
        if (sizeof(enum NPY_TYPES) <= sizeof(long)) {
            return PyInt_FromLong((long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(enum NPY_TYPES) <= sizeof(PY_LONG_LONG)) {
            return PyLong_FromLongLong((PY_LONG_LONG) value);
#endif
        }
    }
    {
        int one = 1; int little = (int)*(unsigned char *)&one;
        unsigned char *bytes = (unsigned char *)&value;
        return _PyLong_FromByteArray(bytes, sizeof(enum NPY_TYPES),
                                     little, !is_unsigned);
    }
}

/* CIntFromPy */
            static CYTHON_INLINE unsigned int __Pyx_PyInt_As_unsigned_int(PyObject *x) {
    const unsigned int neg_one = (unsigned int) -1, const_zero = (unsigned int) 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_MAJOR_VERSION < 3
    if (likely(PyInt_Check(x))) {
        if (sizeof(unsigned int) < sizeof(long)) {
            __PYX_VERIFY_RETURN_INT(unsigned int, long, PyInt_AS_LONG(x))
        } else {
            long val = PyInt_AS_LONG(x);
            if (is_unsigned && unlikely(val < 0)) {
                goto raise_neg_overflow;
            }
            return (unsigned int) val;
        }
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (unsigned int) 0;
                case  1: __PYX_VERIFY_RETURN_INT(unsigned int, digit, digits[0])
                case 2:
                    if (8 * sizeof(unsigned int) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(unsigned int, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(unsigned int) >= 2 * PyLong_SHIFT) {
                            return (unsigned int) (((((unsigned int)digits[1]) << PyLong_SHIFT) | (unsigned int)digits[0]));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(unsigned int) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(unsigned int, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(unsigned int) >= 3 * PyLong_SHIFT) {
                            return (unsigned int) (((((((unsigned int)digits[2]) << PyLong_SHIFT) | (unsigned int)digits[1]) << PyLong_SHIFT) | (unsigned int)digits[0]));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(unsigned int) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(unsigned int, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(unsigned int) >= 4 * PyLong_SHIFT) {
                            return (unsigned int) (((((((((unsigned int)digits[3]) << PyLong_SHIFT) | (unsigned int)digits[2]) << PyLong_SHIFT) | (unsigned int)digits[1]) << PyLong_SHIFT) | (unsigned int)digits[0]));
                        }
                    }
                    break;
            }
#endif
#if CYTHON_COMPILING_IN_CPYTHON
            if (unlikely(Py_SIZE(x) < 0)) {
                goto raise_neg_overflow;
            }
#else
            {
                int result = PyObject_RichCompareBool(x, Py_False, Py_LT);
                if (unlikely(result < 0))
                    return (unsigned int) -1;
                if (unlikely(result == 1))
                    goto raise_neg_overflow;
            }
#endif
            if (sizeof(unsigned int) <= sizeof(unsigned long)) {
                __PYX_VERIFY_RETURN_INT_EXC(unsigned int, unsigned long, PyLong_AsUnsignedLong(x))
#ifdef HAVE_LONG_LONG
            } else if (sizeof(unsigned int) <= sizeof(unsigned PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(unsigned int, unsigned PY_LONG_LONG, PyLong_AsUnsignedLongLong(x))
#endif
            }
        } else {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (unsigned int) 0;
                case -1: __PYX_VERIFY_RETURN_INT(unsigned int, sdigit, (sdigit) (-(sdigit)digits[0]))
                case  1: __PYX_VERIFY_RETURN_INT(unsigned int,  digit, +digits[0])
                case -2:
                    if (8 * sizeof(unsigned int) - 1 > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(unsigned int, long, -(long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(unsigned int) - 1 > 2 * PyLong_SHIFT) {
                            return (unsigned int) (((unsigned int)-1)*(((((unsigned int)digits[1]) << PyLong_SHIFT) | (unsigned int)digits[0])));
                        }
                    }
                    break;
                case 2:
                    if (8 * sizeof(unsigned int) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(unsigned int, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(unsigned int) - 1 > 2 * PyLong_SHIFT) {
                            return (unsigned int) ((((((unsigned int)digits[1]) << PyLong_SHIFT) | (unsigned int)digits[0])));
                        }
                    }
                    break;
                case -3:
                    if (8 * sizeof(unsigned int) - 1 > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(unsigned int, long, -(long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(unsigned int) - 1 > 3 * PyLong_SHIFT) {
                            return (unsigned int) (((unsigned int)-1)*(((((((unsigned int)digits[2]) << PyLong_SHIFT) | (unsigned int)digits[1]) << PyLong_SHIFT) | (unsigned int)digits[0])));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(unsigned int) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(unsigned int, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(unsigned int) - 1 > 3 * PyLong_SHIFT) {
                            return (unsigned int) ((((((((unsigned int)digits[2]) << PyLong_SHIFT) | (unsigned int)digits[1]) << PyLong_SHIFT) | (unsigned int)digits[0])));
                        }
                    }
                    break;
                case -4:
                    if (8 * sizeof(unsigned int) - 1 > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(unsigned int, long, -(long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(unsigned int) - 1 > 4 * PyLong_SHIFT) {
                            return (unsigned int) (((unsigned int)-1)*(((((((((unsigned int)digits[3]) << PyLong_SHIFT) | (unsigned int)digits[2]) << PyLong_SHIFT) | (unsigned int)digits[1]) << PyLong_SHIFT) | (unsigned int)digits[0])));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(unsigned int) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(unsigned int, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(unsigned int) - 1 > 4 * PyLong_SHIFT) {
                            return (unsigned int) ((((((((((unsigned int)digits[3]) << PyLong_SHIFT) | (unsigned int)digits[2]) << PyLong_SHIFT) | (unsigned int)digits[1]) << PyLong_SHIFT) | (unsigned int)digits[0])));
                        }
                    }
                    break;
            }
#endif
            if (sizeof(unsigned int) <= sizeof(long)) {
                __PYX_VERIFY_RETURN_INT_EXC(unsigned int, long, PyLong_AsLong(x))
#ifdef HAVE_LONG_LONG
            } else if (sizeof(unsigned int) <= sizeof(PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(unsigned int, PY_LONG_LONG, PyLong_AsLongLong(x))
#endif
            }
        }
        {
#if CYTHON_COMPILING_IN_PYPY && !defined(_PyLong_AsByteArray)
            PyErr_SetString(PyExc_RuntimeError,
                            "_PyLong_AsByteArray() not available in PyPy, cannot convert large numbers");
#else
            unsigned int val;
            PyObject *v = __Pyx_PyNumber_IntOrLong(x);
 #if PY_MAJOR_VERSION < 3
            if (likely(v) && !PyLong_Check(v)) {
                PyObject *tmp = v;
                v = PyNumber_Long(tmp);
                Py_DECREF(tmp);
            }
 #endif
            if (likely(v)) {
                int one = 1; int is_little = (int)*(unsigned char *)&one;
                unsigned char *bytes = (unsigned char *)&val;
                int ret = _PyLong_AsByteArray((PyLongObject *)v,
                                              bytes, sizeof(val),
                                              is_little, !is_unsigned);
                Py_DECREF(v);
                if (likely(!ret))
                    return val;
            }
#endif
            return (unsigned int) -1;
        }
    } else {
        unsigned int val;
        PyObject *tmp = __Pyx_PyNumber_IntOrLong(x);
        if (!tmp) return (unsigned int) -1;
        val = __Pyx_PyInt_As_unsigned_int(tmp);
        Py_DECREF(tmp);
        return val;
    }
raise_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "value too large to convert to unsigned int");
    return (unsigned int) -1;
raise_neg_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "can't convert negative value to unsigned int");
    return (unsigned int) -1;
}

/* CIntFromPy */
            static CYTHON_INLINE int __Pyx_PyInt_As_int(PyObject *x) {
    const int neg_one = (int) -1, const_zero = (int) 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_MAJOR_VERSION < 3
    if (likely(PyInt_Check(x))) {
        if (sizeof(int) < sizeof(long)) {
            __PYX_VERIFY_RETURN_INT(int, long, PyInt_AS_LONG(x))
        } else {
            long val = PyInt_AS_LONG(x);
            if (is_unsigned && unlikely(val < 0)) {
                goto raise_neg_overflow;
            }
            return (int) val;
        }
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (int) 0;
                case  1: __PYX_VERIFY_RETURN_INT(int, digit, digits[0])
                case 2:
                    if (8 * sizeof(int) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) >= 2 * PyLong_SHIFT) {
                            return (int) (((((int)digits[1]) << PyLong_SHIFT) | (int)digits[0]));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(int) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) >= 3 * PyLong_SHIFT) {
                            return (int) (((((((int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0]));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(int) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) >= 4 * PyLong_SHIFT) {
                            return (int) (((((((((int)digits[3]) << PyLong_SHIFT) | (int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0]));
                        }
                    }
                    break;
            }
#endif
#if CYTHON_COMPILING_IN_CPYTHON
            if (unlikely(Py_SIZE(x) < 0)) {
                goto raise_neg_overflow;
            }
#else
            {
                int result = PyObject_RichCompareBool(x, Py_False, Py_LT);
                if (unlikely(result < 0))
                    return (int) -1;
                if (unlikely(result == 1))
                    goto raise_neg_overflow;
            }
#endif
            if (sizeof(int) <= sizeof(unsigned long)) {
                __PYX_VERIFY_RETURN_INT_EXC(int, unsigned long, PyLong_AsUnsignedLong(x))
#ifdef HAVE_LONG_LONG
            } else if (sizeof(int) <= sizeof(unsigned PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(int, unsigned PY_LONG_LONG, PyLong_AsUnsignedLongLong(x))
#endif
            }
        } else {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (int) 0;
                case -1: __PYX_VERIFY_RETURN_INT(int, sdigit, (sdigit) (-(sdigit)digits[0]))
                case  1: __PYX_VERIFY_RETURN_INT(int,  digit, +digits[0])
                case -2:
                    if (8 * sizeof(int) - 1 > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, long, -(long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 2 * PyLong_SHIFT) {
                            return (int) (((int)-1)*(((((int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
                case 2:
                    if (8 * sizeof(int) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 2 * PyLong_SHIFT) {
                            return (int) ((((((int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
                case -3:
                    if (8 * sizeof(int) - 1 > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, long, -(long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 3 * PyLong_SHIFT) {
                            return (int) (((int)-1)*(((((((int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(int) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 3 * PyLong_SHIFT) {
                            return (int) ((((((((int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
                case -4:
                    if (8 * sizeof(int) - 1 > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, long, -(long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 4 * PyLong_SHIFT) {
                            return (int) (((int)-1)*(((((((((int)digits[3]) << PyLong_SHIFT) | (int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(int) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 4 * PyLong_SHIFT) {
                            return (int) ((((((((((int)digits[3]) << PyLong_SHIFT) | (int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
            }
#endif
            if (sizeof(int) <= sizeof(long)) {
                __PYX_VERIFY_RETURN_INT_EXC(int, long, PyLong_AsLong(x))
#ifdef HAVE_LONG_LONG
            } else if (sizeof(int) <= sizeof(PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(int, PY_LONG_LONG, PyLong_AsLongLong(x))
#endif
            }
        }
        {
#if CYTHON_COMPILING_IN_PYPY && !defined(_PyLong_AsByteArray)
            PyErr_SetString(PyExc_RuntimeError,
                            "_PyLong_AsByteArray() not available in PyPy, cannot convert large numbers");
#else
            int val;
            PyObject *v = __Pyx_PyNumber_IntOrLong(x);
 #if PY_MAJOR_VERSION < 3
            if (likely(v) && !PyLong_Check(v)) {
                PyObject *tmp = v;
                v = PyNumber_Long(tmp);
                Py_DECREF(tmp);
            }
 #endif
            if (likely(v)) {
                int one = 1; int is_little = (int)*(unsigned char *)&one;
                unsigned char *bytes = (unsigned char *)&val;
                int ret = _PyLong_AsByteArray((PyLongObject *)v,
                                              bytes, sizeof(val),
                                              is_little, !is_unsigned);
                Py_DECREF(v);
                if (likely(!ret))
                    return val;
            }
#endif
            return (int) -1;
        }
    } else {
        int val;
        PyObject *tmp = __Pyx_PyNumber_IntOrLong(x);
        if (!tmp) return (int) -1;
        val = __Pyx_PyInt_As_int(tmp);
        Py_DECREF(tmp);
        return val;
    }
raise_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "value too large to convert to int");
    return (int) -1;
raise_neg_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "can't convert negative value to int");
    return (int) -1;
}

/* CIntFromPy */
            static CYTHON_INLINE long __Pyx_PyInt_As_long(PyObject *x) {
    const long neg_one = (long) -1, const_zero = (long) 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_MAJOR_VERSION < 3
    if (likely(PyInt_Check(x))) {
        if (sizeof(long) < sizeof(long)) {
            __PYX_VERIFY_RETURN_INT(long, long, PyInt_AS_LONG(x))
        } else {
            long val = PyInt_AS_LONG(x);
            if (is_unsigned && unlikely(val < 0)) {
                goto raise_neg_overflow;
            }
            return (long) val;
        }
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (long) 0;
                case  1: __PYX_VERIFY_RETURN_INT(long, digit, digits[0])
                case 2:
                    if (8 * sizeof(long) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) >= 2 * PyLong_SHIFT) {
                            return (long) (((((long)digits[1]) << PyLong_SHIFT) | (long)digits[0]));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(long) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) >= 3 * PyLong_SHIFT) {
                            return (long) (((((((long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0]));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(long) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) >= 4 * PyLong_SHIFT) {
                            return (long) (((((((((long)digits[3]) << PyLong_SHIFT) | (long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0]));
                        }
                    }
                    break;
            }
#endif
#if CYTHON_COMPILING_IN_CPYTHON
            if (unlikely(Py_SIZE(x) < 0)) {
                goto raise_neg_overflow;
            }
#else
            {
                int result = PyObject_RichCompareBool(x, Py_False, Py_LT);
                if (unlikely(result < 0))
                    return (long) -1;
                if (unlikely(result == 1))
                    goto raise_neg_overflow;
            }
#endif
            if (sizeof(long) <= sizeof(unsigned long)) {
                __PYX_VERIFY_RETURN_INT_EXC(long, unsigned long, PyLong_AsUnsignedLong(x))
#ifdef HAVE_LONG_LONG
            } else if (sizeof(long) <= sizeof(unsigned PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(long, unsigned PY_LONG_LONG, PyLong_AsUnsignedLongLong(x))
#endif
            }
        } else {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (long) 0;
                case -1: __PYX_VERIFY_RETURN_INT(long, sdigit, (sdigit) (-(sdigit)digits[0]))
                case  1: __PYX_VERIFY_RETURN_INT(long,  digit, +digits[0])
                case -2:
                    if (8 * sizeof(long) - 1 > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, long, -(long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 2 * PyLong_SHIFT) {
                            return (long) (((long)-1)*(((((long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
                case 2:
                    if (8 * sizeof(long) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 2 * PyLong_SHIFT) {
                            return (long) ((((((long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
                case -3:
                    if (8 * sizeof(long) - 1 > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, long, -(long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 3 * PyLong_SHIFT) {
                            return (long) (((long)-1)*(((((((long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(long) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 3 * PyLong_SHIFT) {
                            return (long) ((((((((long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
                case -4:
                    if (8 * sizeof(long) - 1 > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, long, -(long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 4 * PyLong_SHIFT) {
                            return (long) (((long)-1)*(((((((((long)digits[3]) << PyLong_SHIFT) | (long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(long) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 4 * PyLong_SHIFT) {
                            return (long) ((((((((((long)digits[3]) << PyLong_SHIFT) | (long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
            }
#endif
            if (sizeof(long) <= sizeof(long)) {
                __PYX_VERIFY_RETURN_INT_EXC(long, long, PyLong_AsLong(x))
#ifdef HAVE_LONG_LONG
            } else if (sizeof(long) <= sizeof(PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(long, PY_LONG_LONG, PyLong_AsLongLong(x))
#endif
            }
        }
        {
#if CYTHON_COMPILING_IN_PYPY && !defined(_PyLong_AsByteArray)
            PyErr_SetString(PyExc_RuntimeError,
                            "_PyLong_AsByteArray() not available in PyPy, cannot convert large numbers");
#else
            long val;
            PyObject *v = __Pyx_PyNumber_IntOrLong(x);
 #if PY_MAJOR_VERSION < 3
            if (likely(v) && !PyLong_Check(v)) {
                PyObject *tmp = v;
                v = PyNumber_Long(tmp);
                Py_DECREF(tmp);
            }
 #endif
            if (likely(v)) {
                int one = 1; int is_little = (int)*(unsigned char *)&one;
                unsigned char *bytes = (unsigned char *)&val;
                int ret = _PyLong_AsByteArray((PyLongObject *)v,
                                              bytes, sizeof(val),
                                              is_little, !is_unsigned);
                Py_DECREF(v);
                if (likely(!ret))
                    return val;
            }
#endif
            return (long) -1;
        }
    } else {
        long val;
        PyObject *tmp = __Pyx_PyNumber_IntOrLong(x);
        if (!tmp) return (long) -1;
        val = __Pyx_PyInt_As_long(tmp);
        Py_DECREF(tmp);
        return val;
    }
raise_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "value too large to convert to long");
    return (long) -1;
raise_neg_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "can't convert negative value to long");
    return (long) -1;
}

/* CheckBinaryVersion */
            static int __Pyx_check_binary_version(void) {
    char ctversion[4], rtversion[4];
    PyOS_snprintf(ctversion, 4, "%d.%d", PY_MAJOR_VERSION, PY_MINOR_VERSION);
    PyOS_snprintf(rtversion, 4, "%s", Py_GetVersion());
    if (ctversion[0] != rtversion[0] || ctversion[2] != rtversion[2]) {
        char message[200];
        PyOS_snprintf(message, sizeof(message),
                      "compiletime version %s of module '%.100s' "
                      "does not match runtime version %s",
                      ctversion, __Pyx_MODULE_NAME, rtversion);
        return PyErr_WarnEx(NULL, message, 1);
    }
    return 0;
}

/* ModuleImport */
            #ifndef __PYX_HAVE_RT_ImportModule
#define __PYX_HAVE_RT_ImportModule
static PyObject *__Pyx_ImportModule(const char *name) {
    PyObject *py_name = 0;
    PyObject *py_module = 0;
    py_name = __Pyx_PyIdentifier_FromString(name);
    if (!py_name)
        goto bad;
    py_module = PyImport_Import(py_name);
    Py_DECREF(py_name);
    return py_module;
bad:
    Py_XDECREF(py_name);
    return 0;
}
#endif

/* TypeImport */
            #ifndef __PYX_HAVE_RT_ImportType
#define __PYX_HAVE_RT_ImportType
static PyTypeObject *__Pyx_ImportType(const char *module_name, const char *class_name,
    size_t size, int strict)
{
    PyObject *py_module = 0;
    PyObject *result = 0;
    PyObject *py_name = 0;
    char warning[200];
    Py_ssize_t basicsize;
#ifdef Py_LIMITED_API
    PyObject *py_basicsize;
#endif
    py_module = __Pyx_ImportModule(module_name);
    if (!py_module)
        goto bad;
    py_name = __Pyx_PyIdentifier_FromString(class_name);
    if (!py_name)
        goto bad;
    result = PyObject_GetAttr(py_module, py_name);
    Py_DECREF(py_name);
    py_name = 0;
    Py_DECREF(py_module);
    py_module = 0;
    if (!result)
        goto bad;
    if (!PyType_Check(result)) {
        PyErr_Format(PyExc_TypeError,
            "%.200s.%.200s is not a type object",
            module_name, class_name);
        goto bad;
    }
#ifndef Py_LIMITED_API
    basicsize = ((PyTypeObject *)result)->tp_basicsize;
#else
    py_basicsize = PyObject_GetAttrString(result, "__basicsize__");
    if (!py_basicsize)
        goto bad;
    basicsize = PyLong_AsSsize_t(py_basicsize);
    Py_DECREF(py_basicsize);
    py_basicsize = 0;
    if (basicsize == (Py_ssize_t)-1 && PyErr_Occurred())
        goto bad;
#endif
    if (!strict && (size_t)basicsize > size) {
        PyOS_snprintf(warning, sizeof(warning),
            "%s.%s size changed, may indicate binary incompatibility. Expected %zd, got %zd",
            module_name, class_name, basicsize, size);
        if (PyErr_WarnEx(NULL, warning, 0) < 0) goto bad;
    }
    else if ((size_t)basicsize != size) {
        PyErr_Format(PyExc_ValueError,
            "%.200s.%.200s has the wrong size, try recompiling. Expected %zd, got %zd",
            module_name, class_name, basicsize, size);
        goto bad;
    }
    return (PyTypeObject *)result;
bad:
    Py_XDECREF(py_module);
    Py_XDECREF(result);
    return NULL;
}
#endif

/* InitStrings */
            static int __Pyx_InitStrings(__Pyx_StringTabEntry *t) {
    while (t->p) {
        #if PY_MAJOR_VERSION < 3
        if (t->is_unicode) {
            *t->p = PyUnicode_DecodeUTF8(t->s, t->n - 1, NULL);
        } else if (t->intern) {
            *t->p = PyString_InternFromString(t->s);
        } else {
            *t->p = PyString_FromStringAndSize(t->s, t->n - 1);
        }
        #else
        if (t->is_unicode | t->is_str) {
            if (t->intern) {
                *t->p = PyUnicode_InternFromString(t->s);
            } else if (t->encoding) {
                *t->p = PyUnicode_Decode(t->s, t->n - 1, t->encoding, NULL);
            } else {
                *t->p = PyUnicode_FromStringAndSize(t->s, t->n - 1);
            }
        } else {
            *t->p = PyBytes_FromStringAndSize(t->s, t->n - 1);
        }
        #endif
        if (!*t->p)
            return -1;
        ++t;
    }
    return 0;
}

static CYTHON_INLINE PyObject* __Pyx_PyUnicode_FromString(const char* c_str) {
    return __Pyx_PyUnicode_FromStringAndSize(c_str, (Py_ssize_t)strlen(c_str));
}
static CYTHON_INLINE char* __Pyx_PyObject_AsString(PyObject* o) {
    Py_ssize_t ignore;
    return __Pyx_PyObject_AsStringAndSize(o, &ignore);
}
static CYTHON_INLINE char* __Pyx_PyObject_AsStringAndSize(PyObject* o, Py_ssize_t *length) {
#if CYTHON_COMPILING_IN_CPYTHON && (__PYX_DEFAULT_STRING_ENCODING_IS_ASCII || __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT)
    if (
#if PY_MAJOR_VERSION < 3 && __PYX_DEFAULT_STRING_ENCODING_IS_ASCII
            __Pyx_sys_getdefaultencoding_not_ascii &&
#endif
            PyUnicode_Check(o)) {
#if PY_VERSION_HEX < 0x03030000
        char* defenc_c;
        PyObject* defenc = _PyUnicode_AsDefaultEncodedString(o, NULL);
        if (!defenc) return NULL;
        defenc_c = PyBytes_AS_STRING(defenc);
#if __PYX_DEFAULT_STRING_ENCODING_IS_ASCII
        {
            char* end = defenc_c + PyBytes_GET_SIZE(defenc);
            char* c;
            for (c = defenc_c; c < end; c++) {
                if ((unsigned char) (*c) >= 128) {
                    PyUnicode_AsASCIIString(o);
                    return NULL;
                }
            }
        }
#endif
        *length = PyBytes_GET_SIZE(defenc);
        return defenc_c;
#else
        if (__Pyx_PyUnicode_READY(o) == -1) return NULL;
#if __PYX_DEFAULT_STRING_ENCODING_IS_ASCII
        if (PyUnicode_IS_ASCII(o)) {
            *length = PyUnicode_GET_LENGTH(o);
            return PyUnicode_AsUTF8(o);
        } else {
            PyUnicode_AsASCIIString(o);
            return NULL;
        }
#else
        return PyUnicode_AsUTF8AndSize(o, length);
#endif
#endif
    } else
#endif
#if (!CYTHON_COMPILING_IN_PYPY) || (defined(PyByteArray_AS_STRING) && defined(PyByteArray_GET_SIZE))
    if (PyByteArray_Check(o)) {
        *length = PyByteArray_GET_SIZE(o);
        return PyByteArray_AS_STRING(o);
    } else
#endif
    {
        char* result;
        int r = PyBytes_AsStringAndSize(o, &result, length);
        if (unlikely(r < 0)) {
            return NULL;
        } else {
            return result;
        }
    }
}
static CYTHON_INLINE int __Pyx_PyObject_IsTrue(PyObject* x) {
   int is_true = x == Py_True;
   if (is_true | (x == Py_False) | (x == Py_None)) return is_true;
   else return PyObject_IsTrue(x);
}
static CYTHON_INLINE PyObject* __Pyx_PyNumber_IntOrLong(PyObject* x) {
#if CYTHON_USE_TYPE_SLOTS
  PyNumberMethods *m;
#endif
  const char *name = NULL;
  PyObject *res = NULL;
#if PY_MAJOR_VERSION < 3
  if (PyInt_Check(x) || PyLong_Check(x))
#else
  if (PyLong_Check(x))
#endif
    return __Pyx_NewRef(x);
#if CYTHON_USE_TYPE_SLOTS
  m = Py_TYPE(x)->tp_as_number;
  #if PY_MAJOR_VERSION < 3
  if (m && m->nb_int) {
    name = "int";
    res = PyNumber_Int(x);
  }
  else if (m && m->nb_long) {
    name = "long";
    res = PyNumber_Long(x);
  }
  #else
  if (m && m->nb_int) {
    name = "int";
    res = PyNumber_Long(x);
  }
  #endif
#else
  res = PyNumber_Int(x);
#endif
  if (res) {
#if PY_MAJOR_VERSION < 3
    if (!PyInt_Check(res) && !PyLong_Check(res)) {
#else
    if (!PyLong_Check(res)) {
#endif
      PyErr_Format(PyExc_TypeError,
                   "__%.4s__ returned non-%.4s (type %.200s)",
                   name, name, Py_TYPE(res)->tp_name);
      Py_DECREF(res);
      return NULL;
    }
  }
  else if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError,
                    "an integer is required");
  }
  return res;
}
static CYTHON_INLINE Py_ssize_t __Pyx_PyIndex_AsSsize_t(PyObject* b) {
  Py_ssize_t ival;
  PyObject *x;
#if PY_MAJOR_VERSION < 3
  if (likely(PyInt_CheckExact(b))) {
    if (sizeof(Py_ssize_t) >= sizeof(long))
        return PyInt_AS_LONG(b);
    else
        return PyInt_AsSsize_t(x);
  }
#endif
  if (likely(PyLong_CheckExact(b))) {
    #if CYTHON_USE_PYLONG_INTERNALS
    const digit* digits = ((PyLongObject*)b)->ob_digit;
    const Py_ssize_t size = Py_SIZE(b);
    if (likely(__Pyx_sst_abs(size) <= 1)) {
        ival = likely(size) ? digits[0] : 0;
        if (size == -1) ival = -ival;
        return ival;
    } else {
      switch (size) {
         case 2:
           if (8 * sizeof(Py_ssize_t) > 2 * PyLong_SHIFT) {
             return (Py_ssize_t) (((((size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
         case -2:
           if (8 * sizeof(Py_ssize_t) > 2 * PyLong_SHIFT) {
             return -(Py_ssize_t) (((((size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
         case 3:
           if (8 * sizeof(Py_ssize_t) > 3 * PyLong_SHIFT) {
             return (Py_ssize_t) (((((((size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
         case -3:
           if (8 * sizeof(Py_ssize_t) > 3 * PyLong_SHIFT) {
             return -(Py_ssize_t) (((((((size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
         case 4:
           if (8 * sizeof(Py_ssize_t) > 4 * PyLong_SHIFT) {
             return (Py_ssize_t) (((((((((size_t)digits[3]) << PyLong_SHIFT) | (size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
         case -4:
           if (8 * sizeof(Py_ssize_t) > 4 * PyLong_SHIFT) {
             return -(Py_ssize_t) (((((((((size_t)digits[3]) << PyLong_SHIFT) | (size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
      }
    }
    #endif
    return PyLong_AsSsize_t(b);
  }
  x = PyNumber_Index(b);
  if (!x) return -1;
  ival = PyInt_AsSsize_t(x);
  Py_DECREF(x);
  return ival;
}
static CYTHON_INLINE PyObject * __Pyx_PyInt_FromSize_t(size_t ival) {
    return PyInt_FromSize_t(ival);
}


#endif /* Py_PYTHON_H */
