

/**
 * Computes L2 distance between two points of dimension nDim.
 * 
 * @param x Pointer to the beginning of the array representing the first
 *     data point.
 * @param y Poitner to the beginning of the array representing the second
 *     data point.
 * @param nDim The dimensionality (number of elements) in each data point
 *     (must be the same for x and y).
 */
double dist(double *x, double *y, int nDim) {
  double accum = 0.0;
  for (int i = 0; i < nDim; i++) {
    accum += pow((x[i] - y[i]), 2);
  }
  return sqrt(accum);
}

/**
 * Assigns each data point to its "closest" cluster centroid.
 */
export void computeAssignments_ispc(WorkerArgs *const args) {
  double *minDist = new double[args->M];

  int m_start = args->M/MAX_THREADS*args->threadId;
  int m_stop = args->M/MAX_THREADS*args->threadId + args->M/MAX_THREADS;
  
  // Initialize arrays
  // for (int m =0; m < args->M; m++) {
  foreach (int m =m_start...m_stop) {
    minDist[m] = 1e30;
    args->clusterAssignments[m] = -1;
  }

  // Assign datapoints to closest centroids
  for (int k = args->start; k < args->end; k++) {
    // for (int m = 0; m < args->M; m++) {
    foreach (int m =m_start ...m_stop) {
      double d = dist(&args->data[m * args->N],
                      &args->clusterCentroids[k * args->N], args->N);
      if (d < minDist[m]) {
        minDist[m] = d;
        args->clusterAssignments[m] = k;
      }
    }
  }

  free(minDist);
}
