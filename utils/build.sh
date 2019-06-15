PYTHON_DIR=$HOME/anaconda3
cmake \
-DPYTHON_LIBRARY=$PYTHON_DIR/lib/libpython3.6m.so \
-DPYTHON_INCLUDE_DIR=$PYTHON_DIR/include/python3.6m \
-DPYTHON_INCLUDE_DIR2=$PYTHON_DIR/include \
-DBoost_INCLUDE_DIR=/usr/local/include \
-DBoost_NumPy_INCLUDE_DIR=/usr/local/include \
-DBoost_NumPy_LIBRARY_DIR=/usr/local/lib \
-DOpenCV_DIR=$PYTHON_DIR/include/opencv \
-Wno-dev \
. && make
