## 在rec_log文件夹下面存放日志
- offline_train.log:存放离线训练的训练日志，记录每一轮训练的时间和总时间
- online_train.log:存放online的训练日志，包括每一次的训练时间以及操作数据库的时间等
- workFlow.log: 记录Django服务启动以来的日志，包括online和offline训练，网络请求，数据库操作等。用于记录整个工作流程