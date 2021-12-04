POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

    case $key in
        -j|--jupyter-notebook)
			port="$2"
          	echo "starting jupyter-note-book on port: $port"
		  	jupyter notebook --no-browser --ip 127.0.0.1 —port $port
          	shift
          	;;
        -t|--tensorboard-logdir-port)
			logdir="$2"
			port="$3"
          	echo "starting tensorboard on port: $port with logdir: $logdir"
          	tensorboard --logdir $logdir --port $port
          	shift
          	;;

        *)    # unknown option
         	POSITIONAL+=("$1") # save it in an array for later
          	shift # past argument
          	;;
    esac
done
echo "starting sparkUI on port: 4040"
echo "COMPLETE"