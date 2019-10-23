# bash

#export UID=$(id -u)
export GID=$(id -g)
docker build --build-arg USER=$USER \
    --build-arg UID=$UID \
    --build-arg GID=$GID \
    --build-arg PW=$USER \
    -t epic_kitchens:py2-caffe2-cuda9 \
    .
