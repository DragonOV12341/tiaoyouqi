hipcc conv2d.cpp -o conv2d -w -lrocblas -lMIOpen -lhiprtc -std=c++17 -fPIC -lstdc++ -I${DTK_HOME}/include -L${DTK_HOME}/lib64 