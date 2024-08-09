rm -rf build
mkdir build 
cd build 
cmake .. -DCMAKE_INSTALL_PREFIX="./PD_WM_GNN" -DPYTHON_EXECUTABLE=$(which python)
make -j 10
make install