#! /bin/sh

echo -n "Enter the largest number of parallel connections: (1 - max): "
read arg
echo "You entered: $arg"
arg_looping=$arg

while [ $arg_looping -gt 0 ]
do
    let iter_count=$arg-$arg_looping
    iter_count=$(( $iter_count + 1 ))
    echo "Connecting to public server with parallel count: $iter_count" > $iter_count.txt
    ./iperf3 -c bouygues.testdebit.info -p 5204 -P $iter_count -t 5 >> $iter_count.txt
    (( arg_looping-- ))
done

