# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jvergare/TGVD-Proyecto-Interseccion

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jvergare/TGVD-Proyecto-Interseccion/build

# Include any dependencies generated for this target.
include CMakeFiles/octree_module.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/octree_module.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/octree_module.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/octree_module.dir/flags.make

CMakeFiles/octree_module.dir/octree.cpp.o: CMakeFiles/octree_module.dir/flags.make
CMakeFiles/octree_module.dir/octree.cpp.o: ../octree.cpp
CMakeFiles/octree_module.dir/octree.cpp.o: CMakeFiles/octree_module.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jvergare/TGVD-Proyecto-Interseccion/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/octree_module.dir/octree.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/octree_module.dir/octree.cpp.o -MF CMakeFiles/octree_module.dir/octree.cpp.o.d -o CMakeFiles/octree_module.dir/octree.cpp.o -c /home/jvergare/TGVD-Proyecto-Interseccion/octree.cpp

CMakeFiles/octree_module.dir/octree.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/octree_module.dir/octree.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jvergare/TGVD-Proyecto-Interseccion/octree.cpp > CMakeFiles/octree_module.dir/octree.cpp.i

CMakeFiles/octree_module.dir/octree.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/octree_module.dir/octree.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jvergare/TGVD-Proyecto-Interseccion/octree.cpp -o CMakeFiles/octree_module.dir/octree.cpp.s

# Object files for target octree_module
octree_module_OBJECTS = \
"CMakeFiles/octree_module.dir/octree.cpp.o"

# External object files for target octree_module
octree_module_EXTERNAL_OBJECTS =

octree_module.cpython-39-x86_64-linux-gnu.so: CMakeFiles/octree_module.dir/octree.cpp.o
octree_module.cpython-39-x86_64-linux-gnu.so: CMakeFiles/octree_module.dir/build.make
octree_module.cpython-39-x86_64-linux-gnu.so: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
octree_module.cpython-39-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpthread.a
octree_module.cpython-39-x86_64-linux-gnu.so: CMakeFiles/octree_module.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jvergare/TGVD-Proyecto-Interseccion/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module octree_module.cpython-39-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/octree_module.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /home/jvergare/TGVD-Proyecto-Interseccion/build/octree_module.cpython-39-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/octree_module.dir/build: octree_module.cpython-39-x86_64-linux-gnu.so
.PHONY : CMakeFiles/octree_module.dir/build

CMakeFiles/octree_module.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/octree_module.dir/cmake_clean.cmake
.PHONY : CMakeFiles/octree_module.dir/clean

CMakeFiles/octree_module.dir/depend:
	cd /home/jvergare/TGVD-Proyecto-Interseccion/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jvergare/TGVD-Proyecto-Interseccion /home/jvergare/TGVD-Proyecto-Interseccion /home/jvergare/TGVD-Proyecto-Interseccion/build /home/jvergare/TGVD-Proyecto-Interseccion/build /home/jvergare/TGVD-Proyecto-Interseccion/build/CMakeFiles/octree_module.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/octree_module.dir/depend

