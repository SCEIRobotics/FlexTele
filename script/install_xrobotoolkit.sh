project_dir=$(pwd)

uv pip install pybind11
uv pip install setuptools

pybind11_cmake_dir=$(python -m pybind11 --cmakedir)
export pybind11_DIR=${pybind11_cmake_dir}

# Build xrobotoolkit-pc-service
cd ${project_dir}/3rdparty/xrobotoolkit-pc-service
cd RoboticsService/PXREARobotSDK
bash build.sh
pc_service_dir=${project_dir}/3rdparty/xrobotoolkit-pc-service/RoboticsService

# Build xrobotoolkit-pc-service-pybind
cd ${project_dir}/3rdparty/xrobotoolkit-pc-service-pybind
mkdir -p lib
mkdir -p include
cp ${pc_service_dir}/PXREARobotSDK/PXREARobotSDK.h include/
cp -r ${pc_service_dir}/PXREARobotSDK/nlohmann include/nlohmann/
cp ${pc_service_dir}/PXREARobotSDK/build/libPXREARobotSDK.so lib/

uv pip uninstall xrobotoolkit_sdk 
python setup.py install