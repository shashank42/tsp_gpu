//for swap
    float myrandf = curand_uniform(&states[tid]);
    myrandf *= ((float)(N[0] - 4) - 1.0+0.9999999999999999);
    myrandf += 1.0;
    int city_one_swap = (int)truncf(myrandf);

    //because we need only positive part, normal dist is not ideal.
    int sample_space = (int)floor(3+exp(- 0.01 / T[0]) * (float)N[0]);
    // space have to be lesser than N-2, there will be error when
    // city_one=1 and city_two=N-1, that's also an edge condition
    // anyway that's not a problem because I saw your sample space
    // parameter starts from a small percentage, that's fine
	int min_city_two = city_one_swap + 3;
	//this is the key change, we fix city_two larger than city_one
	//so we won;t sample cities that near city one!
    int max_city_two = (city_one_swap +3+ sample_space < N[0])?
        city_one_swap +3+ sample_space:
            (N[0] - 1);
    myrandf = curand_uniform(&states[tid]);
    myrandf *= ((float)max_city_two - (float)min_city_two + 0.999999999999999);
    myrandf += min_city_two;
    int city_two_swap = (int)truncf(myrandf);

    //by this sampling method there's strictly no increase
    //because all loss change calculation is correct now!
