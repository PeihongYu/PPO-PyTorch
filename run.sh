RUNS=6
declare -a model_dir=("exp2" "exp2" "exp2" "exp2" "vishnu")
model_id=(800 1200 1300 1700 2250)

for ((i=1;i<${RUNS};i++));
do
  echo $i
  echo ${model_dir[$i]}
  echo ${model_id[$i]}
  python test_continuous.py --model_dir ${model_dir[$i]} --model_id ${model_id[$i]}
  # python test_continuous.py --model_dir ${i} --model_id ${j}
done

#RUNS=2
#for ((i=1;i<${RUNS};i++));
#do
#  echo $((i*100))
#  python test_continuous.py --model_id $((i*2250))
#  # python test_continuous.py --model_dir ${i} --model_id ${j}
#  echo $i
#done