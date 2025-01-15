#SP Estimate
SP Estimate is a state & parameter estimation library implemented in rust, with the hope of
expanding the tooling available in the rust ecosystem and grow the rust robotics community.

## Parameters
Parameters are values that do not evolve over time. Eventual goal is that all methods should work
for scalars and vectors.

### Running Mean
First and most basic parameter estimation, each measurement is given the same weight and there is
an infinite horizon. Implemented in an online fashion using [Welford's algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm).
