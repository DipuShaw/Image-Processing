import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, exposure, img_as_ubyte, color
from tqdm import tqdm
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import threshold_multiotsu
from skimage.segmentation import watershed
from skimage.morphology import medial_axis
import scipy.ndimage as spim
import warnings
import pandas as pd
from skimage.measure import regionprops_table, label
from scipy.ndimage import generate_binary_structure
from scipy.stats import gaussian_kde

# Function to load TIFF images from a folder
def load_tiff_images_from_folder(pathName):
    images = []
    for filename in os.listdir(pathName):
        full_path = os.path.join(pathName, filename)
        if os.path.isfile(full_path) and (filename.endswith('.tiff') or filename.endswith('.tif')):
            try:
                image = io.imread(full_path)
                if image is not None:
                    images.append(image)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return images

# Function to save images to a folder
def save_images(image_list, folder, subfolder, ext='.tif'):
    save_img_path = os.path.join(folder, subfolder)
    os.makedirs(save_img_path, exist_ok=True)

    for i, image in tqdm(enumerate(image_list)):
        filename = f'image_{i:03d}{ext}'  # Generating a unique filename
        try:
            io.imsave(os.path.join(save_img_path, filename), image)
        except Exception as e:
            print(f"Error saving image {filename}: {e}")

# Function to normalize image intensities
def normalize_images(images):
    normalized_images = []
    for image in tqdm(images):
        p2, p98 = np.percentile(image, (2, 98))
        normalized_image = exposure.rescale_intensity(image, in_range=(p2, p98))
        normalized_images.append(normalized_image)
    return normalized_images

# Function to denoise images
def denoise_images(images):
    denoised_images = []
    for image in tqdm(images):
        if image.ndim == 3:
            image_gray = color.rgb2gray(image)  # Convert to grayscale
            sigma_est = np.mean(estimate_sigma(image_gray))
            nlm_denoise_img = denoise_nl_means(image_gray, h=1.15 * sigma_est, patch_size=5, patch_distance=6)
        else:
            sigma_est = np.mean(estimate_sigma(image))
            nlm_denoise_img = denoise_nl_means(image, h=1.15 * sigma_est, patch_size=5, patch_distance=6)
        denoised_images.append(img_as_ubyte(nlm_denoise_img))
    return denoised_images

# Function to perform multi-otsu segmentation
def multi_threshold_segmentation(images, segments):
    segmented_images = []
    thresholds_list = []
    for image in tqdm(images):
        if image.ndim == 3:
            image = color.rgb2gray(image)  # Convert to grayscale
        unique_values = len(np.unique(img_as_ubyte(image)))
        if unique_values < segments:
            print(f"Warning: Not enough unique values in image for {segments} segments. Using 2 segments instead.")
            segments = 2
        thresholds = threshold_multiotsu(img_as_ubyte(image), classes=segments)
        regions = np.digitize(img_as_ubyte(image), bins=thresholds)
        segmented_images.append(regions)
        thresholds_list.append(thresholds)
    return segmented_images, thresholds_list

# Function to perform watershed segmentation
def perform_watershed(dt, sigma=0.4, iters=1000):
    seg_regions = []
    reduced_peaks = []

    for i in tqdm(range(len(dt))):
        dt1 = spim.gaussian_filter(input=dt[i], sigma=sigma)
        peaks = find_peaks(dt1)
        peaks = trim_saddle_points(peaks=peaks, dt=dt1, max_iters=iters)
        peaks = trim_nearby_peaks(peaks=peaks, dt=dt1)
        peaks, _ = spim.label(peaks)

        regions = watershed(image=-dt[i], markers=peaks, mask=dt[i] > 0)
        regions = regions.astype(int)  # Convert to integer array
        seg_regions.append(regions)
        reduced_peaks.append(peaks)

    return seg_regions, reduced_peaks

# Function to calculate porosity
def porosity_calc(segmented_img, index):
    region_of_interest = []
    phi = np.zeros(len(segmented_img))
    for i in tqdm(range(len(segmented_img))):
        image_to_binarize = segmented_img[i]
        region = image_to_binarize == index
        region_of_interest.append(region)
        phi[i] = np.sum(region) / np.size(region)

    fraction = np.mean(phi)
    return region_of_interest, fraction

# Function to perform distance transform
def distance_transform(bool_img):
    dt = []
    skeleton = []
    for i in tqdm(range(len(bool_img))):
        skel, distance = medial_axis(bool_img[i], return_distance=True)
        dt.append(distance)
        skeleton.append(skel)
    return dt, skeleton

# Function to find peaks in the distance transform
from skimage.morphology import ball

def find_peaks(dt, r_max=4):
    im = dt > 0
    if im.ndim != im.squeeze().ndim:
        warnings.warn('Input image contains a singleton axis:' + str(im.shape) +
                      ' Reduce dimensionality with np.squeeze(im) to avoid' +
                      ' unexpected behavior.')
    if im.ndim == 2:
        footprint = generate_binary_structure(2, 1)
    elif im.ndim == 3:
        footprint = generate_binary_structure(3, 1)
    else:
        raise Exception("only 2-d and 3-d images are supported")
    mx = spim.maximum_filter(dt + 2 * (~im), footprint=footprint)
    peaks = (dt == mx) * im
    return peaks

def trim_saddle_points(peaks, dt, max_iters=10):
    # Placeholder implementation of trim_saddle_points function
    return peaks  # Returning the same peaks as placeholder

def trim_nearby_peaks(peaks, dt):
    # Placeholder implementation of trim_nearby_peaks function
    return peaks  # Returning the same peaks as placeholder

def randomize_colors(regions):
    # Placeholder implementation of randomize_colors function
    return regions  # Returning the same regions as placeholder

def preprocess_intensity_image(image_path):
    intensity_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if intensity_image is None:
        raise ValueError("Error: Unable to load intensity image")

    if len(intensity_image.shape) != 2:
        raise ValueError("Error: Intensity image is not 2-D after loading")

    return intensity_image

def preprocess_images_watershed(regions, bool_img):
    if len(regions.shape) != 2:
        raise ValueError("Regions image must be 2-D")
    if len(bool_img.shape) != 2:
        raise ValueError("Intensity image must be 2-D")

    return regions, bool_img

def property_extract(regions, bool_img, scales):
    Appended_df = []

    for i in tqdm(range(len(regions))):
        regions_array = np.array(regions[i])
        bool_img_array = np.array(bool_img[i])

        regions_binary, bool_img_binary = preprocess_images_watershed(regions_array, bool_img_array)

        labeled_image = label(regions_binary)

        if len(labeled_image.shape) != len(bool_img_binary.shape):
            labeled_image = np.squeeze(labeled_image)
            bool_img_binary = np.squeeze(bool_img_binary)

        if labeled_image.shape != bool_img_binary.shape:
            raise ValueError("Label and intensity image shapes must match.")

        properties = property_extract_helper(labeled_image, bool_img_binary)

        df = pd.DataFrame(properties)
        df['area'] = df['area'] * scales[i] ** 2
        df['equivalent_diameter_2d'] = df['equivalent_diameter'] * scales[i]  # Rename to clarify 2D diameter
        df.insert(0, 'Image', i + 1)
        Appended_df.append(df)

    Appended_df = pd.concat(Appended_df)

    return Appended_df

def property_extract_helper(labeled_image, bool_img_binary):
    properties = regionprops_table(labeled_image, bool_img_binary, properties=['label', 'area', 'equivalent_diameter'])
    return properties

# Function for plotting distribution
def distribution(data, plot=('hist', 'kde'), avg=('amean', 'gmean', 'median', 'mode'), binsize='auto',
                 bandwidth='silverman', **fig_kw):
    fig, ax = plt.subplots(**fig_kw)

    if 'hist' in plot:
        if isinstance(binsize, (int, float)):
            binsize = binsize

        y_values, bins = np.histogram(data, bins=binsize, range=(data.min(), data.max()), density=True)

        print('=======================================')
        print('Number of classes = ', len(bins) - 1)
        print('binsize = ', round(bins[1] - bins[0], 2))
        print('=======================================')
        left_edges = np.delete(bins, -1)
        h = bins[1] - bins[0]
        ax.bar(left_edges, y_values, width=h, color='xkcd:azure', edgecolor='#d9d9d9', align='edge')

    if 'kde' in plot:
        kde = gaussian_kde(data, bw_method=bandwidth)
        bandwidth = round(kde.covariance_factor() * data.std(ddof=1), 2)
        x_values = np.linspace(data.min(), data.max(), num=1000)
        y_values = kde(x_values)

        print('=======================================')
        print('KDE bandwidth = ', round(bandwidth, 2))
        print('=======================================')

        ax.plot(x_values, y_values, color='#2F4858')
        if 'hist' not in plot:
            ax.fill_between(x_values, y_values, color='#80419d', alpha=0.65)

    if 'amean' in avg:
        amean = np.mean(data)
        ax.vlines(amean, 0, np.max(y_values), linestyle='solid', color='#2F4858', label='arith. mean', linewidth=2.5)
        print('=======================================')
        print('amean = ', round(amean, 2))
        print('=======================================')

    if 'gmean' in avg:
        positive_data = data[data > 0]
        if len(positive_data) > 0:
            gmean = np.exp(np.mean(np.log(positive_data)))
            ax.vlines(gmean, 0, np.max(y_values), linestyle='solid', color='#fec44f', label='geo. mean')
            print('=======================================')
            print('gmean = ', round(gmean, 2))
            print('=======================================')
        else:
            print('=======================================')
            print('gmean cannot be calculated due to non-positive values')
            print('=======================================')

    if 'median' in avg:
        median = np.median(data)
        ax.vlines(median, 0, np.max(y_values), linestyle='dashed', color='#2F4858', label='median', linewidth=2.5)
        print('=======================================')
        print('median = ', round(median, 2))
        print('=======================================')

    if 'mode' in avg and 'kde' in plot:
        mode = x_values[np.argmax(y_values)]
        ax.vlines(mode, 0, np.max(y_values), linestyle='dotted', color='#2F4858', label='mode', linewidth=2.5)
        print('=======================================')
        print('mode = ', round(mode, 2))
        print('=======================================')

    ax.set_ylabel('density', color='#252525')
    ax.set_xlabel(r'apparent diameter ($\mu m$)', color='#252525')
    ax.set_xscale('linear')
    ax.legend(loc='best', fontsize=16)

    fig.tight_layout()

    return fig, ax

# Saltykov function and related functions (provided by you)
def Saltykov(diameters, areas, numbins=10, calc_vol=None, text_file=None, return_data=False, left_edge=0):
    if isinstance(numbins, int) is False:
        raise ValueError('Numbins must be a positive integer')
    if numbins <= 0:
        raise ValueError('Numbins must be higher than zero')
    if isinstance(left_edge, (int, float)):
        if left_edge < 0:
            raise ValueError("left_edge must be a positive scalar or 'min'")

    # Compute the histogram
    if left_edge == 'min':
        freq, bin_edges = np.histogram(diameters, bins=numbins, range=(diameters.min(), diameters.max()), density=True)
    else:
        freq, bin_edges = np.histogram(diameters, bins=numbins, range=(left_edge, diameters.max()), density=True)

    h = bin_edges[1] - bin_edges[0]
    binsize = bin_edges[1] - bin_edges[0]

    # Create an array with the left edges of the bins and other with the midpoints
    left_edges = np.delete(bin_edges, -1)
    mid_points = left_edges + binsize / 2

    # Estimate the cumulative areas of each pore or grain size interval
    cumulativeAreas = np.zeros(len(mid_points))
    for index, values in enumerate(mid_points):
        mask = np.logical_and(diameters >= values, diameters < (values + h))
        area_sum = np.sum(areas[mask])
        cumulativeAreas[index] = round(area_sum, 1)

    # Normalize the y-axis values to percentage of the total area
    totalArea = sum(cumulativeAreas)
    cumulativeAreasNorm = [(x / float(totalArea)) * 100 for x in cumulativeAreas]

    # Unfold the population of apparent diameters using the Saltykov method
    freq3D = unfold_population(np.array(cumulativeAreasNorm), bin_edges, binsize, mid_points)

    # Calculate the volume-weighted cumulative frequency distribution
    x_vol = binsize * (4 / 3.) * np.pi * (mid_points ** 3)
    freq_vol = x_vol * freq3D

    # Normalize the y-axis values to percentage of the total area
    totalVol = sum(freq_vol)
    cumulativeVolNorm = [(x / float(totalVol)) * 100 for x in freq_vol]

    cdf = np.cumsum(freq_vol)
    cdf_norm = 100 * (cdf / cdf[-1])

    # Estimate the volume of a particular grain size fraction (if proceed)
    if calc_vol is not None:
        x, y = mid_points, cdf_norm
        index = np.argmax(mid_points > calc_vol)
        angle = np.arctan((y[index] - y[index - 1]) / (x[index] - x[index - 1]))
        volume = y[index - 1] + np.tan(angle) * (calc_vol - x[index - 1])
        if volume < 100.0:
            print('=======================================')
            print('volume fraction (up to', calc_vol, 'microns) =', round(volume, 2), '%')
            print('=======================================')
        else:
            print('=======================================')
            print('volume fraction (up to', calc_vol, 'microns) =', 100, '%')
            print('=======================================')

    # Create a text file (if apply) with the midpoints, class frequencies, and cumulative volumes
    if text_file is not None:
        if isinstance(text_file, str) is False:
            print('text_file must be None or string type')
        df = pd.DataFrame({'mid_points': np.around(mid_points, 3), 'freqs': np.around(freq3D, 4), 'freqs2one': np.around(freq3D * binsize, 3), 'cum_vol': np.around(cdf_norm, 2)})
        if text_file.endswith('.txt'):
            df.to_csv(text_file, sep='\t')
        elif text_file.endswith('.csv'):
            df.to_csv(text_file, sep=';')
        else:
            raise ValueError('text file must be specified as .csv or .txt')
        print('=======================================')
        print('The file {} was created'.format(text_file))
        print('=======================================')

    # Return data or figure (if apply)
    if return_data is True:
        return left_edges, cumulativeVolNorm, h, mid_points, cdf_norm
    elif return_data is False:
        print('=======================================')
        print('bin size = {:0.2f}'.format(binsize))
        print('=======================================')
        return Saltykov_plot(left_edges, cumulativeVolNorm, binsize, mid_points, cdf_norm)
    else:
        raise TypeError('return_data must be set as True or False')

def unfold_population(freq, bin_edges, binsize, mid_points, normalize=True):
    d_values = np.copy(bin_edges)
    midpoints = np.copy(mid_points)
    i = len(midpoints) - 1

    while i > 0:
        j = i
        D = d_values[-1]
        Pi = wicksell_solution(D, d_values[i], d_values[i + 1])

        if freq[i] > 0:
            while j > 0:
                D = midpoints[-1]
                Pj = wicksell_solution(D, d_values[j - 1], d_values[j])
                P_norm = (Pj * freq[i]) / Pi
                np.put(freq, j - 1, freq[j - 1] - P_norm)  # Replace specified elements of an array
                j -= 1

            i -= 1
            d_values = np.delete(d_values, -1)
            midpoints = np.delete(midpoints, -1)

        # If the value of the current class is zero or negative move to the next class
        else:
            i -= 1
            d_values = np.delete(d_values, -1)
            midpoints = np.delete(midpoints, -1)

    if normalize is True:
        freq = np.clip(freq, a_min=0.0, a_max=None)  # Replacing negative values with zero
        freq_norm = freq / sum(freq)  # Normalize to one
        freq_norm = freq_norm / binsize  # Normalize such that the integral over the range is one
        return freq_norm
    else:
        return freq

def wicksell_solution(D, d1, d2):
    R, r1, r2 = D / 2, d1 / 2, d2 / 2
    return 1 / R * (np.sqrt(R**2 - r1**2) - np.sqrt(R**2 - r2**2))

def Saltykov_plot(left_edges, freq3D, binsize, mid_points, cdf_norm):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # Frequency vs grain size plot
    ax1.bar(left_edges, freq3D, width=binsize, color='xkcd:azure', edgecolor='#d9d9d9', align='edge')
    ax1.set_ylabel('density', fontsize=18)
    ax1.set_xlabel(r'log(diameter) ($\mu m$)', fontsize=18)

    # Volume-weighted cumulative frequency curve
    ax2.set_ylim([-2, 105])
    ax2.plot(mid_points, cdf_norm, 'o-', color='#ed4256', label='volume weighted CFD', linewidth=2)
    ax2.set_ylabel('cumulative volume (%)', color='#252525')
    ax2.set_xlabel(r'log(diameter) ($\mu m$)', color='#252525')

    fig.tight_layout()
    return fig, (ax1, ax2)

# Define volume_weighted function
def volume_weighted(diameters, volumes, binsize=20, **fig_kw):
    """Generates an area-weighted histogram and returns different
    area-weighted statistics.
    Parameters
    ----------
    diameters : array_like
        the size of the pores or grains
    volumes : array_like
        the volumes of the pores or grains
    binsize : string or positive scalar, optional
        If 'auto', it defines the plug-in method to calculate the bin size.
        When integer or float, it directly specifies the bin size.
        Default: the 'auto' method.
        | Available plug-in methods:
        | 'auto' (fd if sample_size > 1000 or Sturges otherwise)
        | 'doane' (Doane's rule)
        | 'fd' (Freedman-Diaconis rule)
        | 'rice' (Rice's rule)
        | 'scott' (Scott rule)
        | 'sqrt' (square-root rule)
        | 'sturges' (Sturge's rule)
    **fig_kw :
        additional keyword arguments to control the size (figsize) and
        resolution (dpi) of the plot. Default figsize is (6.4, 4.8).
        Default resolution is 100 dpi.
    Examples
    --------
    >>> volume_weighted(data['diameters'], data['Volume'])
    >>> volume_weighted(data['diameters'], data['Volume'], binsize='doane', dpi=300)
    """

    # Ensure volumes is a numpy array
    volumes = np.array(volumes)

    # Estimate weighted mean
    volume_total = np.sum(volumes)
    weighted_areas = volumes / volume_total
    weighted_mean = np.sum(diameters * weighted_areas)

    histogram, bin_edges = np.histogram(diameters, bins=binsize, range=(0.0, diameters.max()))
    h = bin_edges[1] - bin_edges[0]

    # Estimate the cumulative areas of each pore or grain size interval
    cumulativeVolumes = np.zeros(len(bin_edges))
    for index, values in enumerate(bin_edges[:-1]):
        mask = np.logical_and(diameters >= values, diameters < (values + h))
        volume_sum = np.sum(volumes[mask])
        cumulativeVolumes[index] = round(volume_sum, 1)

    # Get the index of the modal interval
    getIndex = np.argmax(cumulativeVolumes)

    print('=======================================')
    print('DESCRIPTIVE STATISTICS')
    print('Volume-weighted mean pore or grain size = {:0.2f} microns'.format(weighted_mean))
    print('=======================================')
    print('HISTOGRAM FEATURES')
    print('The modal interval is {left:0.2f} - {right:0.2f} microns'.format(left=bin_edges[getIndex],
                                                                            right=bin_edges[getIndex] + h))
    print('The number of classes are {}'.format(len(histogram)))
    if isinstance(binsize, str):
        print('The bin size is {bin:0.2f} according to the {rule} rule'.format(bin=h, rule=binsize))
    print('=======================================')

    # Normalize the y-axis values to percentage of the total area
    totalVolume = sum(cumulativeVolumes)
    cumulativeVolumesNorm = [(x / float(totalVolume)) * 100 for x in cumulativeVolumes]
    maxValue = max(cumulativeVolumesNorm)

    return bin_edges, cumulativeVolumesNorm, h


# Update the main section to extract actual data from processed images
if __name__ == "__main__":
    folder_path = "E:\\BTCM\\pics"
    save_path = "E:\\BTCM\\processed_images"

    # Load images
    print("Loading images...")
    images = load_tiff_images_from_folder(folder_path)
    print(f"Loaded {len(images)} images.")

    # Normalize images
    print("Normalizing images...")
    normalized_images = normalize_images(images)

    # Denoise images
    print("Denoising images...")
    denoised_images = denoise_images(normalized_images)

    # Perform multi-threshold segmentation
    print("Performing multi-threshold segmentation...")
    segments = 4  # Example number of segments
    segmented_images, thresholds = multi_threshold_segmentation(denoised_images, segments)

    # Debug: Check segmentation results
    for i, seg in enumerate(segmented_images):
        print(f"Image {i+1}: Unique values in segmented image - {np.unique(seg)}")

    # Calculate porosity
    print("Calculating porosity...")
    region_of_interest, porosity = porosity_calc(segmented_images, index=1)
    print(f"Porosity: {porosity}")

    # Check if porosity is zero
    if porosity == 0:
        print("Warning: Porosity is calculated as 0. Verify the segmentation and index used for porosity calculation.")
        # Visualize the segmented image and region of interest for debugging
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(segmented_images[0], cmap='gray')
        ax[0].set_title('Segmented Image')
        ax[1].imshow(region_of_interest[0], cmap='gray')
        ax[1].set_title('Region of Interest')
        plt.show()

    # Save processed images
    print("Saving processed images...")
    save_images(segmented_images, save_path, "segmented")

    print("Processing complete.")

    # Perform distance transform on the region of interest
    dt_images, skeletons = distance_transform(region_of_interest)

    # Perform watershed segmentation
    watershed_images, _ = perform_watershed(dt_images)

    # Display original, denoised, segmented, distance transform, and watershed images
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))

    axes[0].imshow(images[0], cmap='gray')
    axes[0].set_title('Original Image')

    axes[1].imshow(denoised_images[0], cmap='gray')
    axes[1].set_title('Denoised Image')

    axes[2].imshow(segmented_images[0], cmap='gray')
    axes[2].set_title('Segmented Image')

    axes[3].imshow(dt_images[0], cmap='gray')
    axes[3].set_title('Distance Transform')

    axes[4].imshow(watershed_images[0], cmap='gray')
    axes[4].set_title('Watershed Image')

    plt.tight_layout()
    plt.savefig('processing_steps.png')  # Save the figure
    plt.show()  # Display the figure

    # Extract properties from segmented regions
    scales = [1] * len(segmented_images)  # Assuming a scale of 1 for simplicity
    properties_df = property_extract(segmented_images, denoised_images, scales)

    # Extract diameters and areas for Saltykov method
    diameters_2d = properties_df['equivalent_diameter_2d'].values
    areas = properties_df['area'].values

    # Apply Saltykov method to estimate 3D diameters
    left_edges, cumulativeVolNorm, h, diameters_3d, cdf_norm = Saltykov(diameters_2d, areas, return_data=True)

    # Plot distribution with 3D diameters
    print("Plotting 3D diameter distribution...")
    fig, ax = distribution(diameters_3d)
    plt.savefig('diameter_distribution_plot.png')  # Save the figure
    plt.show()  # Display the figure

    # Perform cumulative volume distribution analysis using Saltykov results
    print("Performing cumulative volume distribution analysis...")
    bin_edges, cumulativeVolumesNorm, h = volume_weighted(diameters_3d, cumulativeVolNorm, binsize='auto', figsize=(10, 6), dpi=150)

    print("Analysis complete.")

    # Ensure the volume-weighted histogram is also displayed
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.bar(bin_edges[:-1], cumulativeVolumesNorm[:len(bin_edges)-1], width=h, color='xkcd:azure', edgecolor='#d9d9d9', align='edge')
    ax.set_ylabel('Cumulative Volume (%)', fontsize=14)
    ax.set_xlabel('Diameter (microns)', fontsize=14)
    plt.tight_layout()
    plt.savefig('volume_weighted_histogram.png')  # Save the figure
    plt.show()  # Display the figure

    # Plot log of 3D diameters vs cumulative volume fraction
    print("Plotting log of 3D diameters vs cumulative volume fraction...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(np.log10(diameters_3d), cdf_norm, 'o-', color='#ed4256', label='CDF')
    ax.set_ylabel('Cumulative Volume Fraction (%)', fontsize=14)
    ax.set_xlabel('Log of Diameter (microns)', fontsize=14)
    plt.tight_layout()
    plt.savefig('log_diameter_vs_cumulative_volume.png')  # Save the figure
    plt.show()  # Display the figure
