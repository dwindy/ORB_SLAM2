# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/137/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/137/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xin/Downloads/ORB_SLAM2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xin/Downloads/ORB_SLAM2/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/mono_kitti.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mono_kitti.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mono_kitti.dir/flags.make

CMakeFiles/mono_kitti.dir/Examples/Monocular/mono_kitti.cc.o: CMakeFiles/mono_kitti.dir/flags.make
CMakeFiles/mono_kitti.dir/Examples/Monocular/mono_kitti.cc.o: ../Examples/Monocular/mono_kitti.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xin/Downloads/ORB_SLAM2/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mono_kitti.dir/Examples/Monocular/mono_kitti.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mono_kitti.dir/Examples/Monocular/mono_kitti.cc.o -c /home/xin/Downloads/ORB_SLAM2/Examples/Monocular/mono_kitti.cc

CMakeFiles/mono_kitti.dir/Examples/Monocular/mono_kitti.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mono_kitti.dir/Examples/Monocular/mono_kitti.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xin/Downloads/ORB_SLAM2/Examples/Monocular/mono_kitti.cc > CMakeFiles/mono_kitti.dir/Examples/Monocular/mono_kitti.cc.i

CMakeFiles/mono_kitti.dir/Examples/Monocular/mono_kitti.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mono_kitti.dir/Examples/Monocular/mono_kitti.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xin/Downloads/ORB_SLAM2/Examples/Monocular/mono_kitti.cc -o CMakeFiles/mono_kitti.dir/Examples/Monocular/mono_kitti.cc.s

# Object files for target mono_kitti
mono_kitti_OBJECTS = \
"CMakeFiles/mono_kitti.dir/Examples/Monocular/mono_kitti.cc.o"

# External object files for target mono_kitti
mono_kitti_EXTERNAL_OBJECTS =

../Examples/Monocular/mono_kitti: CMakeFiles/mono_kitti.dir/Examples/Monocular/mono_kitti.cc.o
../Examples/Monocular/mono_kitti: CMakeFiles/mono_kitti.dir/build.make
../Examples/Monocular/mono_kitti: ../lib/libORB_SLAM2.so
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_shape.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_stitching.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_objdetect.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_superres.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_videostab.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_calib3d.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_features2d.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_flann.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_highgui.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_ml.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_photo.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_video.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_videoio.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_imgproc.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libopencv_core.so.3.2.0
../Examples/Monocular/mono_kitti: /usr/local/lib/libpangolin.so
../Examples/Monocular/mono_kitti: /usr/lib/x86_64-linux-gnu/libOpenGL.so
../Examples/Monocular/mono_kitti: /usr/lib/x86_64-linux-gnu/libGLX.so
../Examples/Monocular/mono_kitti: /usr/lib/x86_64-linux-gnu/libGLU.so
../Examples/Monocular/mono_kitti: /usr/lib/x86_64-linux-gnu/libGLEW.so
../Examples/Monocular/mono_kitti: /usr/lib/x86_64-linux-gnu/libX11.so
../Examples/Monocular/mono_kitti: /usr/lib/x86_64-linux-gnu/libXext.so
../Examples/Monocular/mono_kitti: /usr/lib/x86_64-linux-gnu/libOpenGL.so
../Examples/Monocular/mono_kitti: /usr/lib/x86_64-linux-gnu/libGLX.so
../Examples/Monocular/mono_kitti: /usr/lib/x86_64-linux-gnu/libGLU.so
../Examples/Monocular/mono_kitti: /usr/lib/x86_64-linux-gnu/libGLEW.so
../Examples/Monocular/mono_kitti: /usr/lib/x86_64-linux-gnu/libX11.so
../Examples/Monocular/mono_kitti: /usr/lib/x86_64-linux-gnu/libXext.so
../Examples/Monocular/mono_kitti: ../Thirdparty/DBoW2/lib/libDBoW2.so
../Examples/Monocular/mono_kitti: ../Thirdparty/g2o/lib/libg2o.so
../Examples/Monocular/mono_kitti: CMakeFiles/mono_kitti.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xin/Downloads/ORB_SLAM2/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../Examples/Monocular/mono_kitti"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mono_kitti.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mono_kitti.dir/build: ../Examples/Monocular/mono_kitti

.PHONY : CMakeFiles/mono_kitti.dir/build

CMakeFiles/mono_kitti.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mono_kitti.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mono_kitti.dir/clean

CMakeFiles/mono_kitti.dir/depend:
	cd /home/xin/Downloads/ORB_SLAM2/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xin/Downloads/ORB_SLAM2 /home/xin/Downloads/ORB_SLAM2 /home/xin/Downloads/ORB_SLAM2/cmake-build-debug /home/xin/Downloads/ORB_SLAM2/cmake-build-debug /home/xin/Downloads/ORB_SLAM2/cmake-build-debug/CMakeFiles/mono_kitti.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mono_kitti.dir/depend

