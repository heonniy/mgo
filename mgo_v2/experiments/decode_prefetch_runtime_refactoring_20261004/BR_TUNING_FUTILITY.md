# Stop only BR tuning extensions that provably cannot pass

The owner requested autonomous jitter handling and efficient use of repetitions.
The two/three-repeat stable gates and the five/seven-repeat 2% relative mean-CI
threshold remain unchanged. Paired BR/LA timing rules remain unchanged.

For BR-only tuning, after at least three samples, stop as **ineligible** if no
possible set of positive future observations can pass the existing precision
threshold at either permitted endpoint (five or seven samples). This rejects a
provably futile cell; it never makes a noisy cell eligible or discards a sample.
All existing completed seven-repeat data remain preserved and reported.

For m known positive times, S=sum(x), Q=sum(x*x), and final n with r=n-m
remaining observations, the minimum possible relative t-CI halfwidth is:

    t95[n-1] * sqrt((m*Q - S*S) / ((n-1)*(S*S + r*Q)))

The optimum gives every future time the value Q/S. Derivation: minimize
(Q+r*y*y)/(S+r*y)^2; its stationary minimum is y=Q/S. Unequal future values can
only increase the sum of squares for a fixed future sum. Compare both E2E and
TPOT; each permitted endpoint must be impossible for at least one metric.
This grants unrealistically perfect future repeats, so exceeding 2% is a
conservative proof that the current cell cannot become eligible within the cap.

The active B128 worker retains the originally loaded 2/3/5/7 rule. The next
fresh worker (B256 and subsequent BR confirmation) uses this early rejection.
Eligibility thresholds are identical across batches. No runtime code or
measurement interval changes, and no active timing sample is interrupted.
