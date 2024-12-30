model_name="MyFirstApp_run"

num=0
# 判断传入参数的个数
if [ $# -eq 1 ]; then
    num=$1
fi

cd ${APP_SOURCE_PATH}/out

./main $num
