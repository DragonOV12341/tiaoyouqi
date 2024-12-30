model_name="MyFirstApp_build"

num=0
# 判断传入参数的个数
if [ $# -eq 1 ]; then
    num=$1
fi

# data
cd ${APP_SOURCE_PATH}/data
python3 ../script/gen_data_bin.py $num

# model
cd ${APP_SOURCE_PATH}/model
python gen_conv.py $num

if [ -d ${APP_SOURCE_PATH}/build/intermediates/host ];then
	rm -rf ${APP_SOURCE_PATH}/build/intermediates/host
fi

mkdir -p ${APP_SOURCE_PATH}/build/intermediates/host
cd ${APP_SOURCE_PATH}/build/intermediates/host

cmake ../../../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE

make