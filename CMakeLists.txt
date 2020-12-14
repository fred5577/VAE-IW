cmake_minimum_required(VERSION 2.8.12)
project(bprost)

#FIND_PACKAGE(pybind11 REQUIRED)

add_subdirectory(pybind11)
pybind11_add_module(bprost srcC/main.cpp srcC/screen.cpp srcC/screen.h)