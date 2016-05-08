# Copyright 2015 Samsung Austin Semiconductor, LLC.

# Description: Makefile for building a cbp 2016 submission.
#
# The following commands and Makefile worked for CENTOS v7:
# ================================
# yum install gcc-c++
# yum install boost
# yum install boost-devel.x86_64
#
# The following commands worked for Ubuntu:
#==========================================
#  sudo apt get update
#  sudo apt-get install libboost-all-dev
#
#  installed it in /usr/include/boost
#  lib at /usr/lib/x86_64-linux-gnu/
#
#  locate boost | grep lib
#
#  sudo apt-get install g++
#    Installed 4.8


# Include path to Boost area on local machine
#
#worked with both CENTOS and Ubuntu: export BOOST    := /usr/
export BOOST    := /usr/
#export BOOST    := /sarc/spa/tools/conda/envs/spa-1

ifndef BOOST
$(error "You must define BOOST")
endif

export CBP_BASE := $(CURDIR)/..
CXX := g++

OBJDIR := $(CBP_BASE)/bin
SRC    := $(CBP_BASE)/sim:base
VPATH  := $(SRC)

LDLIBS   += -lboost_iostreams

LDFLAGS += -L$(BOOST)/lib -Wl,-rpath $(BOOST)/lib

INCLUDE_DIRS :=-I$(CBP_BASE) -I./base/ -I/usr/include -I/user/include/boost/ -I/usr/include/boost/iostreams/ -I/usr/include/boost/iostreams/device/ -I/usr/include/python2.7 -lpython2.7 


CPPFLAGS := -g -O3 -Wall -std=c++11 \
		-Wextra -Winline -Winit-self -Wno-sequence-point -Wno-unused-function -Wno-inline -W -Wcast-qual -Wpointer-arith -Woverloaded-virtual\
		-fPIC \

PROGRAMS := predictor

objects = predictor.o main.o

all:
	$(MAKE) -d BUDGET=cbp32KB PREDICTOR=gshare predictor
	$(MAKE) -d BUDGET=cbp32KB PREDICTOR=perceptron predictor
#       $(MAKE) -d BUDGET=cbp32KB PREDICTOR=bias_free_neural predictor
#	$(MAKE) -d BUDGET=cbp32KB PREDICTOR=global_local_tage predictor
#	$(MAKE) -d BUDGET=cbp32KB PREDICTOR=hpattern predictor
#	$(MAKE) -d BUDGET=cbp32KB PREDICTOR=poTageColt predictor
#	$(MAKE) -d BUDGET=cbp32KB PREDICTOR=poTageSc predictor
#	$(MAKE) -d BUDGET=cbp32KB PREDICTOR=t_bag predictor
#	$(MAKE) -d BUDGET=cbp32KB PREDICTOR=tage_sc_l predictor
#	$(MAKE) -d BUDGET=cbp32KB PREDICTOR=temporal_stream predictor
#	$(MAKE) -d BUDGET=cbp32KB PREDICTOR=wormhole predictor


predictor: CPPFLAGS += $(INCLUDE_DIRS) -I./predictors/$(PREDICTOR)/$(BUDGET) 
predictor: VPATH += ./predictors/$(PREDICTOR)/$(BUDGET)
predictor: predictors/$(PREDICTOR)/$(BUDGET)/predictor.o base/main.o
	$(CXX) $(CPPFLAGS) $(LDFLAGS)  -L./predictors/$(PREDICTOR)/$(BUDGET) -L./base/ -o ../bin/$(BUDGET)_$(PREDICTOR)_new $^ $(LDLIBS) 




dbg: clean
	$(MAKE) DBG_BUILD=1 all

clean:
	rm -f $(PROGRAMS) $(objects)
