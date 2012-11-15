#ifndef STEVENSHEN_MACRO_H
#define STEVENSHEN_MACRO_H

#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(ClassName) \
  ClassName(const ClassName&); \
  void operator = (const ClassName&); 
#endif

#ifndef NO_GLAG
  #include <gflags/gflags.h>
#endif

#ifndef NO_GLOG
  #include <glog/logging.h>
#else
  #ifndef CHECK
  #define CHECK(Phrase) \
    { auto v__ = Phrase; \
      if (!v__) { exit(-1); } \
    }
  #endif
#endif

#endif /* STEVENSHEN_MACRO_H */
