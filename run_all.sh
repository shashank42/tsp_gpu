./columbus data/mona-lisa100K -trip= data/mona-lisa100K_trip_june1st  -global_search= 10 -local_search= 30 -temp= 15000 -decay= .9985 -maxiter= 16000 |& tee data/mona_output.txt

./columbus data/earring200K -trip= data/earring200K_trip_june1st -global_search= 10 -local_search= 30 -temp= 15000 -decay= .9985 -maxiter= 16000 |& tee data/earring_output.txt

./columbus data/vangoh120k -trip= data/vangoh120k_trip_june1st -global_search= 10 -local_search= 30 -temp= 15000 -decay= .9985 -maxiter= 16000 |& tee data/vangoh_output.txt

./columbus data/venus140K -trip= data/venus140K_trip_june1st -global_search= 10 -local_search= 30 -temp= 15000 -decay= .9985 -maxiter= 16000 |& tee data/venus_output.txt

./columbus data/lrb744710 -trip= data/lrb744710_trip_june1st -global_search= 10 -local_search= 30 -temp= 15000 -decay= .9985 -maxiter= 16000 |& tee data/lrb_output.txt
