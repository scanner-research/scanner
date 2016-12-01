##******************************************************************************
##  Copyright(C) 2012-2015 Intel Corporation. All Rights Reserved.
##  
##  The source code, information  and  material ("Material") contained herein is
##  owned  by Intel Corporation or its suppliers or licensors, and title to such
##  Material remains  with Intel Corporation  or its suppliers or licensors. The
##  Material  contains proprietary information  of  Intel or  its  suppliers and
##  licensors. The  Material is protected by worldwide copyright laws and treaty
##  provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
##  modified, published, uploaded, posted, transmitted, distributed or disclosed
##  in any way  without Intel's  prior  express written  permission. No  license
##  under  any patent, copyright  or  other intellectual property rights  in the
##  Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
##  implication, inducement,  estoppel or  otherwise.  Any  license  under  such
##  intellectual  property  rights must  be express  and  approved  by  Intel in
##  writing.
##
##  *Third Party trademarks are the property of their respective owners.
##
##  Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
##  this  notice or  any other notice embedded  in Materials by Intel or Intel's
##  suppliers or licensors in any way.
##
##******************************************************************************
##  Content: Intel(R) Media SDK Samples projects creation and build
##******************************************************************************

if(LINUX)
  set( os_arch "lin" )
elseif(APPLE)
  set( os_arch "darwin" )
endif()
if( __ARCH MATCHES ia32)
  set( os_arch "${os_arch}_ia32" )
else( )
  set( os_arch "${os_arch}_x64" )
endif( )

if(UNIX OR APPLE)
  if(CMAKE_MFX_HOME)
    set( MFX_API_HOME ${CMAKE_MFX_HOME} )
  else()
    set( MFX_API_HOME $ENV{MFX_HOME} )
  endif()
  find_path( MFX_INCLUDE mfxdefs.h PATHS ${MFX_API_HOME} PATH_SUFFIXES include )
  find_library ( MFX_LIBRARY libmfx.a PATHS ${MFX_API_HOME}/lib PATH_SUFFIXES ${os_arch} )

  # required:
  include_directories( ${MFX_API_HOME}/include )
  link_directories( ${MFX_API_HOME}/lib/${os_arch} )

else( )
  set( MFX_INCLUDE NOTFOUND )
  set( MFX_LIBRARY NOTFOUND )

endif( )

if(NOT MFX_INCLUDE MATCHES NOTFOUND)
  set( MFX_FOUND TRUE )
  include_directories( ${MFX_INCLUDE} )
endif( )

if(NOT DEFINED MFX_FOUND)
  message( FATAL_ERROR "Intel(R) Media SDK was not found in ${MFX_API_HOME} (${MFX_INCLUDE}, ${MFX_LIBRARY} required)! Set/check MFX_HOME environment variable!")
else ( )
  message( STATUS "Intel(R) Media SDK ${MFX_INCLUDE}, ${MFX_LIBRARY} was found here ${MFX_API_HOME}")
endif( )

if(NOT MFX_LIBRARY MATCHES NOTFOUND)
  get_filename_component(MFX_LIBRARY_PATH ${MFX_LIBRARY} PATH )
  message( STATUS "Intel(R) Media SDK ${MFX_LIBRARY_PATH} will be used")
  link_directories( ${MFX_LIBRARY_PATH} )
endif( )

if(UNIX)
  set(MFX_LDFLAGS "-Wl,--default-symver" )
endif( )
