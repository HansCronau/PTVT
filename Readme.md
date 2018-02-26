Path Traced Virtual Textures
============================

This project is a slimmed down version of the Optix 4.0.2 SDK.
It contains a virtual texture demo built on top of the optixPathTracer sample.


Requirements
------------

- 64-bit system
- CUDA Toolkit 6.5, 7.0, or 7.5
- OptiX SDK 4.0.2
- DevIL SDK 1.8.0 (http://openil.sourceforge.net/download.php)

Advised:

- Windows 7 or later
- Visual Studio 2013 (not later!)

For the requirements of compiling with OptiX see Development Environment Requirements:

- https://docs.nvidia.com/gameworks/content/gameworkslibrary/optix/optix_release_notes.htm


Installation
------------

1. Make sure to have downloaded and installed the CUDA Toolkit, the DevIL SDK (just save it somewhere), and the OptiX SDK.
2. To generate the project with CMake, this folder needs to be placed next to the "SDK" folder in your OptiX installation folder.
   Typically the path will resemble C:\ProgramData\NVIDIA Corporation\OptiX SDK <version number>\.
3. After moving to this folder carefully(!) follow the instructions in "INSTALL-<PLATFORM>.txt" for instructions for your specific platform.
4. After following these steps, check the CMakeGUI to see if all options starting with "DEVIL_" are correct paths to your DevIL SDK.
   Match all "DEVIL_" paths in CMake to where you saved the DevIL SDK and repeat the CMake steps of pressing Configure twice and Generate once.
5. By now, CMake will have added the DevIL include directory to your project, and linked with the DevIL libraries.
   For Windows systems I have also configured CMake to automatically move all .dlls (DevIL) located in folder "install" to the right folders.
   You don't need to do that manually! :D
   However, if you are not on Windows, this will not help you and for this step you'll be on your own. :(
6. To build and run the project in Visual Studio, dont forget to right click the project "optixPathTracer" in the Solution Explorer and select "Set as StartUp Project".


Important
---------

If you wish to *modify* this project, especially if you wish to add files, be sure to learn how to operate CMake.
That helped me a lot.


~ Hans Cronau
