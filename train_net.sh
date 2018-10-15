GLOG_logtostderr=0 GLOG_log_dir=./Log/ ../../build/tools/caffe train --solver=solver.prototxt \
    #-weights ./mobilenet.caffemodel \
    2>&1| tee ./caffe.log$@

