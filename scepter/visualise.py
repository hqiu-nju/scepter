"""
TBD

Author: boris.sorokin <mralin@protonmail.com>
Date: 16-04-2025
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution_curves(total_Prx_recovered, label="Total Prx", unit_str="dB(W)", plot_type="ccdf"):
    """
    Plot fancy distribution curves (CDF and/or CCDF) of the recovered total received power.
    
    Depending on the input parameter 'plot_type', this function will produce:
      - "cdf": a single-panel plot of the cumulative distribution function,
      - "ccdf": a single-panel plot of the complementary cumulative distribution function (with a log-scaled y-axis),
      - "both": a two-panel figure showing both curves side by side.
    
    Parameters
    ----------
    total_Prx_recovered : astropy.units.Quantity or ndarray
        The recovered total received power. If an astropy Quantity, its .value attribute is used.
    label : str, optional
        Label for the x-axis (e.g., "Total Received Power"). Default is "Total Prx".
    unit_str : str, optional
        A string representing the unit (e.g., "dB(W)"). Default is "dB(W)".
    plot_type : str, optional
        Determines which plots to display. Acceptable values are:
           "cdf"   - Plot only the CDF.
           "ccdf"  - Plot only the CCDF.
           "both"  - Plot both CDF and CCDF side-by-side (default).
    
    Returns
    -------
    None
        Displays the generated plot(s).
    """
    # Process input: if it's an astropy Quantity, use its .value.
    try:
        data = total_Prx_recovered.value
    except AttributeError:
        data = np.array(total_Prx_recovered)
        
    # Flatten data into one dimension.
    data = data.flatten()
    
    # Sort the data.
    data_sorted = np.sort(data)
    N = len(data_sorted)
    
    # Compute cumulative probabilities for CDF.
    cdf = np.linspace(1./N, 1.0, N)
    # Compute CCDF (complementary CDF), with a slight adjustment to avoid zeros.
    ccdf = 1 - cdf + (1.0 / N)
    
    # Set a modern seaborn style.
    sns.set(style="whitegrid", context="talk")
    
    # Validate plot_type.
    if plot_type not in ("cdf", "ccdf", "both"):
        raise ValueError("plot_type must be one of 'cdf', 'ccdf', or 'both'.")
    
    if plot_type == "both":
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot CDF.
        axes[0].plot(data_sorted, cdf, marker='o', linestyle='-', color='royalblue',
                     markersize=4, markerfacecolor='white', linewidth=2)
        axes[0].set_title(f"CDF of {label}", fontsize=16, fontweight='bold')
        axes[0].set_xlabel(f"{label} ({unit_str})", fontsize=14)
        axes[0].set_ylabel("Cumulative Probability", fontsize=14)
        axes[0].tick_params(axis='both', which='major', labelsize=12)
        
        # Plot CCDF.
        axes[1].plot(data_sorted, ccdf, marker='o', linestyle='-', color='darkorange',
                     markersize=4, markerfacecolor='white', linewidth=2)
        axes[1].set_title(f"CCDF of {label}", fontsize=16, fontweight='bold')
        axes[1].set_xlabel(f"{label} ({unit_str})", fontsize=14)
        axes[1].set_ylabel("Complementary CDF", fontsize=14)
        axes[1].set_yscale("log")
        axes[1].tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
    
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        if plot_type == "cdf":
            ax.plot(data_sorted, cdf, marker='o', linestyle='-', color='royalblue',
                    markersize=4, markerfacecolor='white', linewidth=2)
            ax.set_title(f"CDF of {label}", fontsize=16, fontweight='bold')
            ax.set_ylabel("Cumulative Probability", fontsize=14)
        elif plot_type == "ccdf":
            ax.plot(data_sorted, ccdf*100, marker='o', linestyle='-', color='darkorange',
                    markersize=4, markerfacecolor='white', linewidth=2)
            ax.set_title(f"CCDF of {label}", fontsize=16, fontweight='bold')
            ax.set_ylabel("Complementary CDF, %", fontsize=14)
            ax.set_yscale("log")
        ax.set_xlabel(f"{label} ({unit_str})", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        
    plt.show()