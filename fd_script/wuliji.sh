my_host=`hostname -i`
row_id=`awk "/$my_host/{print NR}" /root/paddlejob/workspace/hostfile`


PADDLE_TRAINER_ID=$(($row_id-1))
PADDLE_TRAINERS_NUM=`wc -l /root/paddlejob/workspace/hostfile`
