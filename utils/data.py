import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

class NumpyDataset(Dataset):
    def __init__(self, X: np.array, y: np.array, sentinel_number: int, band_order: list[str], requested_indices: list[str], return_distros=False, full_distrib=None):


        self.data = torch.from_numpy(X)
        self.labels = torch.from_numpy(y)


        assert self.data.shape[0] == self.labels.shape[0]

        if full_distrib:
            self.full_distrib = torch.from_numpy(full_distrib)
            assert self.full_distrib.shape[0] == self.labels.shape[0]

        self.band_order = band_order
        self.requested_indices = requested_indices

        self.return_distros = return_distros
        
        if sentinel_number is 1:
            self.sliced_data = compute_sentinel1_indices(self.data, requested_indices, band_order)
        else:
            self.sliced_data = compute_sentinel2_indices(self.data, requested_indices, band_order)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        
        if not self.return_distros:
            return self.data[idx], self.sliced_data[idx], self.labels[idx]
        
        return self.data[idx], self.sliced_data[idx], self.full_distrib[idx]
 

def load_pickled_ds(fname: str):
    with open(fname, 'rb') as f:
        x = pickle.load(f)
    return x


def compute_sentinel2_indices(data, requested_indices, channel_order):
    """
    Compute vegetation indices and selected Sentinel-2 bands from batched data.
    Parameters:
    - data (numpy array): Input array of shape (Batch, Channels, Width, Height), where
                          Channels are ordered according to Sentinel-2 bands.
    - requested_indices (set): Set of strings specifying desired outputs (e.g., {"NDVI", "R", "G", "EVI"}).
    - channel_order (list): List of strings defining the order of channels in the data.
                            Example: ["B1", "B02", "B03", "B04", ..., "B12"].
    Returns:
    - result (numpy array): Output array of shape (Batch, len(requested_indices), Width, Height),
                            where each channel corresponds to a computed index or band.
    """
    # Map Sentinel-2 bands to their indices in the input array
    channel_map = {band: i for i, band in enumerate(channel_order)}
    # Add shorthand mappings for common RGB bands
    shorthand_map = {
        "B": "B02",  # Blue
        "G": "B03",  # Green
        "R": "B04",  # Red
        "NIR": "B08",  # Near Infrared
        "SWIR1": "B11",  # Shortwave Infrared 1
        "SWIR2": "B12"   # Shortwave Infrared 2
    }
    # Resolve all shorthand indices to full band names
    resolved_indices = {shorthand_map.get(idx, idx) for idx in requested_indices}
    # Prepare output array
    batch_size, _, width, height = data.shape
    result = np.zeros((batch_size, len(requested_indices), width, height))
    # Compute requested indices
    for i, index in enumerate(requested_indices):
        resolved_index = shorthand_map.get(index, index)
        if resolved_index in channel_map:
            # Directly copy specified bands (e.g., "R", "G", etc.)
            result[:, i, :, :] = data[:, channel_map[resolved_index], :, :]
        elif resolved_index == "NDVI":
            # NDVI = (NIR - R) / (NIR + R)
            nir = data[:, channel_map["B08"], :, :]  # NIR (Band 8)
            red = data[:, channel_map["B04"], :, :]  # Red (Band 4)
            result[:, i, :, :] = (nir - red) / (nir + red + 1e-8)
        elif resolved_index == "EVI":
            # EVI = 2.5 * (NIR - R) / (NIR + 6*R - 7.5*B + 1)
            nir = data[:, channel_map["B08"], :, :]
            red = data[:, channel_map["B04"], :, :]
            blue = data[:, channel_map["B02"], :, :]
            result[:, i, :, :] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-6)
        elif resolved_index == "GNDVI":
            # GNDVI = (NIR - G) / (NIR + G)
            nir = data[:, channel_map["B08"], :, :]
            green = data[:, channel_map["B03"], :, :]
            result[:, i, :, :] = (nir - green) / (nir + green + 1e-6)
        elif resolved_index == "SAVI":
            # SAVI = (NIR - R) * (1 + L) / (NIR + R + L), L=0.5
            nir = data[:, channel_map["B08"], :, :]
            red = data[:, channel_map["B04"], :, :]
            result[:, i, :, :] = ((nir - red) * 1.5) / (nir + red + 0.5 + 1e-6)
        elif resolved_index == "ARVI":
            # ARVI = (NIR - (2*R - B)) / (NIR + (2*R - B))
            nir = data[:, channel_map["B08"], :, :]
            red = data[:, channel_map["B04"], :, :]
            blue = data[:, channel_map["B02"], :, :]
            result[:, i, :, :] = (nir - (2 * red - blue)) / (nir + (2 * red - blue) + 1e-8)
        elif resolved_index == "MSAVI":
            # MSAVI = (2 * NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - R))) / 2
            nir = data[:, channel_map["B08"], :, :]
            red = data[:, channel_map["B04"], :, :]
            numerator = 2 * nir + 1
            denominator = np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red)) + 1e-8
            result[:, i, :, :] = (numerator - denominator) / 2
        elif resolved_index == "NDWI":
            # NDWI = (G - NIR) / (G + NIR)
            green = data[:, channel_map["B03"], :, :]
            nir = data[:, channel_map["B08"], :, :]
            result[:, i, :, :] = (green - nir) / (green + nir + 1e-8)
        else:
            raise ValueError(f"Unknown index or band: {index}")
    return result


def compute_sentinel1_indices(data, requested_indices, channel_order):
    """
    Compute vegetation indices from Sentinel-1 data.
    Parameters:
    - data (numpy array): Input array of shape (NumSamples, Bands, W, H).
                          Channels are ordered as in `channel_order`.
    - requested_indices (list): List of requested indices (e.g., ["RVI", "NDPI"]).
    - channel_order (list): List defining the order of bands in `data` (e.g., ["VV", "VH", "VV-VH"]).
    Returns:
    - result (numpy array): Output array of shape (NumSamples, len(requested_indices), W, H),
                            where each channel corresponds to a computed index.
    """
    # Map channel names to indices
    channel_map = {band: i for i, band in enumerate(channel_order)}
    # Ensure required bands exist
    if "VV" not in channel_map or "VH" not in channel_map:
        raise ValueError("Input data must include 'VV' and 'VH' bands.")
    # Extract channels from the data
    vv = data[:, channel_map["VV"], :, :]  # (NumSamples, W, H)
    vh = data[:, channel_map["VH"], :, :]  # (NumSamples, W, H)
    vv_vh = data[:, channel_map["VV-VH"], :, :] if "VV-VH" in channel_map else vv / (vh + 1e-6)
    # Prepare output array
    num_samples, _, width, height = data.shape
    result = np.zeros((num_samples, len(requested_indices), width, height))
    # Compute requested indices
    for i, index in enumerate(requested_indices):
        if index == "VV":
            result[:, i, :, :] = vv  # Return VV band
        elif index == "VH":
            result[:, i, :, :] = vh  # Return VH band
        elif index == "RVI":
            result[:, i, :, :] = (4 * vh) / (vv + vh + 1e-6)
        elif index == "VV-VH":
            result[:, i, :, :] = vv_vh  # Use precomputed channel
        elif index == "NDPI":
            result[:, i, :, :] = (vv - vh) / (vv + vh + 1e-6)
        elif index == "DPSVI":
            result[:, i, :, :] = np.sqrt(abs(vh * vv)) # here is a squezy moment bcs it's wrong calculations
        elif index == "CSI":
            result[:, i, :, :] = vv / (vh + 1e-6)
        elif index == "VSI":
            result[:, i, :, :] = vh**2 - vv**2
        else:
            raise ValueError(f"Unknown index: {index}")
    return result
