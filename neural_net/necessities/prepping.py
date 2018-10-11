import numpy as np

def reBatchSeries(data, size=20):
    # in:          [timestep] [samples] [state]
    # out: [batch] [timestep] [samples] [state]
    # Some calcs
    n_series = np.shape(data)[0] # Length of series
    n_samples = np.shape(data)[1] # Number of samples
    n_batches = int(np.ceil(n_samples/size)) # Number of batches
    # Room for batches
    batches = [[[] for _ in range(n_series)] for batch_n in range(n_batches)]
    # Adding data
    for series_n in range(n_series):        # Iterating through timesteps
        for sample_n in range(n_samples):   # Iterating through samples
            batches[sample_n % n_batches][series_n].append(data[series_n][sample_n])
    # Returning
    return [np.array(batch) for batch in batches]

def batchifySeries(data):
    # in:  [samples] [timestep] [state]
    # out: [timestep] [samples] [state]
    # Calcs
    n_series = len(data[0]) # Length of series
    n_samples = len(data) # Number of samples
    # Room for batches
    batch = [[] for _ in range(n_series)]
    # Adding data
    for series_n in range(n_series):
        for sample_n in range(n_samples):
            batch[series_n].append(data[sample_n][series_n])
    # Returning
    return np.array(batch)

def img2series(data):
    return np.rot90(data, k=-1)
