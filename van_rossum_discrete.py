def van_rossum_distance2(exp_spike_train, est_spike_train, tau):

    ################################################################################
    # Calculates the Van Rossum distance between two spike trains
    ################################################################################
    # --- Computes Van Rossum distance analytical terms
    sum_u_u = sum_exp_pairwise(exp_spike_train, exp_spike_train, tau)
    sum_v_v = sum_exp_pairwise(est_spike_train, est_spike_train, tau)
    sum_u_v = sum_exp_pairwise(exp_spike_train, est_spike_train, tau)
    # --- Squared distance
    distance = sum_u_u + sum_v_v - 2 * sum_u_v

    # --- Applies normalization
    distance = jnp.sqrt(jnp.maximum(0.0, distance))

    return distance
