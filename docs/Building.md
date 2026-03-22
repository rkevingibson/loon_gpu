# Building

Loon GPU uses CMake for it's build system. If you're familiar with running cmake, building should just work - all dependencies are downloaded via CMake's `FetchContent` module. There are a number of CMake presets defined in the repo which can be used but they're not necessary.

## On Windows
While I generally use Clang for day-to-day development of the library, MSVC has been tested and works.
If you want to use Visual Studio, you should be able to clone the repo, open the folder in Visual Studio, and hit build. 

## On MacOS


## On Linux

Linux hasn't been tested, but may work with Clang- there's very little platform-specific code. Currently examples will not work on linux as I don't have a platform layer implemented yet. 