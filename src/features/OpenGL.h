#ifndef SIFTSLAM_OPENGL_H
#define SIFTSLAM_OPENGL_H
#include <GSLAM/core/Point.h>

#ifdef HAS_OPENGL

#if defined(__linux)
    #include <GL/glew.h>
    #include <GL/glut.h>
#else
    #include <GL/glew.h>
//    #include "gui/GL_headers/glext.h"
#endif

inline void glVertex(const pi::Point3d& pt)
{
    glVertex3d(pt.x,pt.y,pt.z);
}

inline void glVertex(const pi::Point3f& pt)
{
    glVertex3f(pt.x,pt.y,pt.z);
}

inline void glColor(const pi::Point3ub& color)
{
    glColor3ub(color.x,color.y,color.z);
}
#endif

#endif
