import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import to_rgba
import numpy as np
import os

# Fig. 9
def fig9_plot_grouped_error_bars(csv_path, labelmap, model_order=None, figsize=(20, 6), output_path=None, title=None, no_legend=False):
    """
    Generate grouped bar graph showing performance and power estimation errors.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing the data
    labelmap : dict
        Dictionary mapping model names to display labels
        Example: {'bertmodel': 'BERT', 'gpt2model': 'GPT-2'}
    model_order : list, optional
        List of model names defining the order of models on x-axis
        Example: ['bertmodel', 'gpt2model', 'optmodel', 'qwen2model', 'resnet101', 'mobilevitmodel', 'vitmodel']
        If None, models appear in the order they appear in the data
    figsize : tuple, optional
        Figure size (width, height). Default: (20, 6)
    output_path : str, optional
        If provided, saves the figure to this path. Can be .png or .pdf
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    
    Example:
    --------
    >>> labelmap = {'bertmodel': 'BERT', 'gpt2model': 'GPT-2'}
    >>> model_order = ['bertmodel', 'gpt2model']
    >>> fig, ax = plot_grouped_error_bars('data.csv', labelmap, model_order)
    >>> plt.savefig('output.png', dpi=300, bbox_inches='tight')
    >>> plt.show()
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Calculate power estimation error
    # gee_power = gee_energy / gee_time * 1000 (converting to W from mJ/ms)
    df['gee_power'] = df['gee_energy'] / df['gee_time'] * 1000
    df['true_power'] = df['measured_energy'] / df['measured_time'] * 1000
    df['gee_power_abs_pct_err'] = np.abs(df['true_power'] - df['gee_power']) / df['true_power'] * 100
    
    # Fill NaN values with appropriate defaults
    df['seq'] = df['seq'].fillna(0)
    df['mode'] = df['mode'].fillna('')
    df['config'] = df['config'].fillna('')
    
    # Create unique identifiers for each configuration
    df['unique_id'] = (df['model'] + '_' + 
                       df['config'] + '_' + 
                       df['prec'] + '_' + 
                       df['mode'] + '_' + 
                       df['batch'].astype(str) + '_' + 
                       df['seq'].astype(str))
    
    # Create x-tick labels (only batch and seq, no model name)
    def create_label(row):
        batch_str = str(int(row['batch']))
        if row['seq'] > 1:
            seq_str = str(int(row['seq'])) if int(row['seq']) < 1024 else ('{}k'.format(int(int(row['seq']/1024))))
            return f"{batch_str}\n{seq_str}"
        else:
            return f"{labelmap[row['model']]}"
    
    df['x_label'] = df.apply(create_label, axis=1)
    
    # Filter out any rows with invalid unique_ids
    df = df[df['unique_id'].notna() & (df['unique_id'] != '') & (df['unique_id'] != 'nan')]
    
    # Sort the dataframe
    # 1. First by model_order if provided
    if model_order is not None:
        # Create a model order mapping
        model_order_map = {model: idx for idx, model in enumerate(model_order)}
        # Add a sort column based on model_order, use large number for models not in order
        df['model_sort'] = df['model'].map(lambda x: model_order_map.get(x, 999))
    else:
        # Use the order models appear in the data
        unique_models = df['model'].unique()
        model_order_map = {model: idx for idx, model in enumerate(unique_models)}
        df['model_sort'] = df['model'].map(model_order_map)
    
    # 2. Then by batch (ascending)
    # 3. Then by seq (ascending)
    df = df.sort_values(['model_sort', 'batch', 'seq'], ascending=[True, True, True])
    
    # Get unique configurations in sorted order
    unique_configs = df['unique_id'].unique()
    n_configs = len(unique_configs)
    
    # Track model boundaries for vertical lines
    model_boundaries = []
    model_groups = {}  # Track start and end indices for each model
    prev_model = None
    current_model_start = 0
    
    # Prepare data for plotting
    limicro_errors = []
    neusight_errors = []
    gee_time_errors = []
    gee_power_errors = []
    x_labels = []
    
    for idx, config in enumerate(unique_configs):
        config_data = df[df['unique_id'] == config].iloc[0]
        limicro_errors.append(config_data['limicro_time_abs_pct_err'])
        neusight_errors.append(config_data['neusight_time_abs_pct_err'])
        gee_time_errors.append(config_data['gee_time_abs_pct_err'])
        gee_power_errors.append(config_data['gee_power_abs_pct_err'])
        # limicro_errors.append(config_data['limicro_time']/config_data['measured_time'])
        # neusight_errors.append(config_data['neusight_time']/config_data['measured_time'])
        # gee_time_errors.append(config_data['gee_time']/config_data['measured_time'])
        # gee_power_errors.append(config_data['gee_power']/config_data['true_power'])
        x_labels.append(config_data['x_label'])
        
        # Track model changes for vertical lines and grouping
        current_model = config_data['model']
        if prev_model is not None and current_model != prev_model:
            # Add boundary at the position between prev and current
            model_boundaries.append(idx - 0.5)
            # Store the previous model's range
            model_groups[prev_model] = (current_model_start, idx - 1)
            current_model_start = idx
        prev_model = current_model
    
    # Don't forget the last model group
    if prev_model is not None:
        model_groups[prev_model] = (current_model_start, len(unique_configs) - 1)
    
    # Calculate averages
    limicro_avg_err = df['limicro_time_abs_pct_err'].mean()
    neusight_avg_err = df['neusight_time_abs_pct_err'].mean()
    gee_time_avg_err = df['gee_time_abs_pct_err'].mean()
    gee_power_avg_err = df['gee_power_abs_pct_err'].mean()
    print("Li MAPE: {:.2f} %".format(limicro_avg_err))
    print("NeuSight MAPE: {:.2f} %".format(neusight_avg_err))
    print("GEE MAPE: {:.2f} %".format(gee_time_avg_err))
    print("GEE POWER MAPE: {:.2f} %".format(gee_power_avg_err))
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    # ax.axhline(y=1.0, color='black', linestyle='-')
    # Set up bar positions
    x = np.arange(n_configs + 1)  # +1 for average
    width = 0.2
    
    # Define colors and hatches
    colors = ['#0173b2', '#de8f05', '#029e73', '#cc78bc', 
          '#ece133', '#56b4e9', '#ca9161', '#949494']
    color_limicro = "#0173b2"      # Dark gray
    color_neusight = "#de8f05"     # Light gray
    color_gee_time = '#029e73'     # Red (as specified)
    color_gee_power = "#cc78bc"    # Blue (clearly differentiated from red)
    
    hatch_limicro = '//'
    hatch_neusight = 'xx'
    hatch_gee_time = 'oo'
    hatch_gee_power = '**'
    
    # Plot bars for each configuration
    bars1 = ax.bar(x[:-1] - 1.5*width, limicro_errors, width, 
                   label='Latency, Li et al.', color=color_limicro, 
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x[:-1] - 0.5*width, neusight_errors, width, 
                   label='Latency, NeuSight', color=color_neusight,
                   edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x[:-1] + 0.5*width, gee_time_errors, width, 
                   label='Latency, Ours', color=color_gee_time,
                   edgecolor='black', linewidth=0.5)
    bars4 = ax.bar(x[:-1] + 1.5*width, gee_power_errors, width, 
                   label='Power, Ours', color=color_gee_power, 
                   edgecolor='black', linewidth=0.5)
    
    # Plot average bars
    bars1_avg = ax.bar(x[-1] - 1.5*width, limicro_avg_err, width, 
                       color=color_limicro,
                       edgecolor='black', linewidth=0.5)
    bars2_avg = ax.bar(x[-1] - 0.5*width, neusight_avg_err, width, 
                       color=color_neusight,
                       edgecolor='black', linewidth=0.5)
    bars3_avg = ax.bar(x[-1] + 0.5*width, gee_time_avg_err, width, 
                       color=color_gee_time,
                       edgecolor='black', linewidth=0.5)
    bars4_avg = ax.bar(x[-1] + 1.5*width, gee_power_avg_err, width, 
                       color=color_gee_power,
                       edgecolor='black', linewidth=0.5)
    
    # Add vertical dotted line before average
    ax.axvline(x=n_configs - 0.5, color='black', linestyle=':', linewidth=1.5)
    
    # Add vertical dotted lines between different models
    for boundary in model_boundaries:
        ax.axvline(x=boundary, color='black', linestyle=':', linewidth=1.0, alpha=0.8)
    
    # Set x-axis labels (only batch,seq)
    all_labels = x_labels + ['Avg']
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=11, rotation=0, ha='center')
    
    # Add model names centered below each model group
    for model, (start_idx, end_idx) in model_groups.items():
        if (model == 'vitmodel') or (model == 'mobilevitmodel') or (model == 'resnet101'):
            continue
        center_pos = (start_idx + end_idx) / 2
        model_label = labelmap.get(model, model)
        ax.text(center_pos, -0.4, model_label, 
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=11, fontweight='normal')
    
    # Set labels
    ax.set_ylabel('Abs. Pct. Error (%)', fontsize=13)
    ax.set_xlabel('', fontsize=13)  # Remove default x-label since we have model names below
    
    ax.set_xlim(-0.5, len(all_labels)-0.5)

    y_max = 35
    ax.set_ylim(0, y_max)


    # Add text annotations for values exceeding y_max
    # For regular configurations
    for idx in range(len(limicro_errors)):
        x_pos = idx
        
        # Check each bar type
        if limicro_errors[idx] > y_max:
            ax.text(x_pos - 1.5*width, y_max - 2, str(int(limicro_errors[idx])), 
                   ha='center', va='top', fontsize=7, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
        
        if neusight_errors[idx] > y_max:
            ax.text(x_pos - 0.5*width, y_max - 2, str(int(neusight_errors[idx])), 
                   ha='center', va='top', fontsize=7, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
        
        if gee_time_errors[idx] > y_max:
            ax.text(x_pos + 0.5*width, y_max - 2, str(int(gee_time_errors[idx])), 
                   ha='center', va='top', fontsize=7, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
        
        if gee_power_errors[idx] > y_max:
            ax.text(x_pos + 1.5*width, y_max - 2, str(int(gee_power_errors[idx])), 
                   ha='center', va='top', fontsize=7, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
    
    
    # Add legend
    if not no_legend:
        ax.legend(loc='upper center', fontsize=13, frameon=True, ncols=4, bbox_to_anchor=(0.5, 1.4))
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.8, linestyle='-', linewidth=1)
    ax.set_axisbelow(True)

    if title is not None:
        plt.title(title, fontsize=14, loc='left', fontweight='bold')
    
    # Adjust layout to make room for model names below
    # plt.tight_layout()
    
    # Make extra space at the bottom for model names
    plt.subplots_adjust(bottom=0.15)
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path+'.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_path+'.png', dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")

    plt.show()
    
    return fig, ax


# Fig. 11
def fig11_plot_dvfs(df, configurations, model_name_map):
    # Calculate power metrics
    df['measured_power'] = df['measured_energy'] / df['measured_time'] * 1000.
    df['gee_power'] = df['gee_energy'] / df['gee_time'] * 1000.

    # Calculate subplot layout
    n_configs = len(configurations)
    ncols = 2
    nrows = int(np.ceil(n_configs / ncols))

    # Create figure with subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, 2*nrows))
    axes = axes.flatten()  # Flatten to make indexing easier

    # Plot each configuration
    for idx, config_tuple in enumerate(configurations):
        model, config, batch, seq, mode = config_tuple
        ax = axes[idx]
        
        # Filter data for this combination
        mask = (
            (df['model'] == model) & 
            (df['batch'] == batch) & 
            (df['seq'] == seq) & 
            (df['mode'] == mode) & 
            (df['config'] == config)
        )
        subset = df[mask].copy()
        
        # Get measured_freq (assuming it's the same for all rows in this subset)
        if len(subset) > 0:
            measured_freq = subset['measured_freq'].iloc[0]
            
            # Filter points where abs(measured_freq - freq) > 45
            subset = subset[np.abs(subset['measured_freq'] - subset['freq']) <= 90].copy()
        
        # Sort by frequency for better visualization
        subset = subset.sort_values('freq')
        
        if len(subset) > 0:
            # Calculate MAPE
            mape = np.mean(np.abs((subset['gee_power'] - subset['measured_power']) / subset['measured_power'])) * 100
            
            ax.set_xlim(509, 1411)
            ax.set_ylim(50, 260)

            # Plot scatter points
            ax.scatter(subset['freq'], subset['measured_power'], 
                    marker='o', color='#0173B2', s=100, label='Truth', zorder=3)
            ax.scatter(subset['freq'], subset['gee_power'], 
                    marker='^', color='#DE8F05', s=100, label='Estimated', zorder=3)
            
            # Find the largest frequency that was plotted
            max_freq = subset['freq'].max()
            
            # Draw vertical line at max frequency
            ax.axvline(x=max_freq, color='gray', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
            
            # Shade the area to the right of the max frequency
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.axvspan(max_freq, xlim[1], color='gray', alpha=0.2, zorder=1)
            ax.set_xlim(xlim)  # Restore original xlim
            ax.set_ylim(ylim)  # Restore original ylim
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Labels and title

            if idx % 2 == 0:
                ax.set_ylabel('Power (W)', fontsize=12)

            if idx // 2 == (nrows-1):    
                ax.set_xlabel('Frequency (MHz)', fontsize=12)

            model_name = model_name_map[model]
            ax.set_title(f'{model_name}, batch={batch}, seq={seq}', fontsize=12, fontweight='bold')
            
            # Legend
            # ax.legend(fontsize=9, loc='upper left')
            
            # Add MAPE annotation
            ax.text(0.05, 0.9, f'MAPE: {mape:.1f}%', 
                    transform=ax.transAxes, 
                    fontsize=11,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'No data after filtering', 
                    transform=ax.transAxes, 
                    fontsize=12,
                    horizontalalignment='center',
                    verticalalignment='center')
            ax.set_title(f'{model}, batch={batch}, seq={seq}', fontsize=12, fontweight='bold')

        if idx == 0:
            ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(1.1, 1.55), ncols=2)

    plt.subplots_adjust(hspace=0.4)
    
    # Hide any unused subplots
    for idx in range(n_configs, len(axes)):
        axes[idx].axis('off')

    plt.show()

# Fig. 10
def get_mape(parent_path, estimation_methods, estimation_nametags):
    time_mape = {}
    energy_mape = {}
    power_mape = {}

    for i, method in enumerate(estimation_methods):
        df = pd.read_csv(os.path.join(parent_path, method, 'estimation_result.csv'))
        df['power'] = df['energy'] / df['time'] * 1000.
        
        t_m = np.abs(df['time_percent_error']).mean()
        e_m = np.abs(df['energy_percent_error']).mean()

        df['power_estimate'] = df['energy_estimate'] / df['time_estimate'] * 1000
        df['power_percent_error'] = (df['power_estimate'] - df['power']) / df['power'] * 100.
        
        p_m = np.abs(df['power_percent_error']).mean()

        time_mape[estimation_nametags[i]] = t_m
        energy_mape[estimation_nametags[i]] = e_m
        power_mape[estimation_nametags[i]] = p_m

    return time_mape, energy_mape, power_mape

def create_time_bar_plot(ax, input_dict, methods, colors, omit_y=False, title='GPU'):
    """Modified to accept ax as parameter instead of creating fig internally"""
    # Extract tick names and methods
    tick_names = list(input_dict.keys())
    
    # Prepare data for plotting
    time_data = []
    
    for tick_name in tick_names:
        time_result, energy_result, power_result = input_dict[tick_name]
        
        # Extract values for each method (use 0 if method not present)
        time_values = [time_result.get(method, 0) for method in methods]
        time_data.append(time_values)
    
    # Convert to numpy arrays for easier manipulation
    time_data = np.array(time_data)
    
    # Bar width and positions
    n_methods = len(methods)
    n_ticks = len(tick_names)
    bar_width = 0.8 / n_methods
    
    # Create positions for each group of bars
    x_positions = np.arange(n_ticks)

    # Plot time results
    for i, method in enumerate(methods):
        offset = (i - (n_methods - 1) / 2) * bar_width
        bars = ax.bar(x_positions + offset, np.minimum(time_data[:, i], 60), bar_width, 
                      label=method, color=colors[i], edgecolor='black', linewidth=1)
        
        # Add numbers inside the bar for values exceeding 60
        for j, value in enumerate(time_data[:, i]):
            if value > 60:
                ax.text(x_positions[j] + offset, 64, f'{value:.0f}', 
                       ha='center', va='center', fontsize=12, fontweight='normal', color='black')
    
    # Configure subplot
    ax.set_xlabel('')
    if not omit_y:
        ax.set_ylabel('MAPE (%)', fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_names, fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 60)
    ax.set_title(title, loc='left', fontweight='bold', fontsize=14)
    
    return ax

def fig10_plot(parent_folder='../experiments_single/results', seed=1):

    estimation_methods = ['limicro', \
                      'neusight', \
                      'gee_v3_refFalse_dvfsFalse']

    estimation_nametags = ['Li et al.', 'NeuSight', 'Ours']

    a100_gemm_bf16 = os.path.join(parent_folder, 'yz8_bf16bf16_tc_artifact', 'seed{}'.format(seed))
    a100_gemm_fp32 = os.path.join(parent_folder, 'yz8_fp32_cuda_artifact', 'seed{}'.format(seed))
    a100_softmax_bf16 = os.path.join(parent_folder, 'yz8_bf16_softmax_artifact', 'seed{}'.format(seed))
    a100_layernorm_bf16 = os.path.join(parent_folder, 'yz8_bf16_layernorm_artifact', 'seed{}'.format(seed))
    a100_conv_bf16 = os.path.join(parent_folder, 'yz8_conv2d_bf16bf16_tc_artifact', 'seed{}'.format(seed))
    a100_elementwise = os.path.join(parent_folder, 'yz8_elementwise_artifact', 'seed{}'.format(seed))

    a100_results = {}
    a100_results['TC'] = get_mape(a100_gemm_bf16, estimation_methods, estimation_nametags)
    a100_results['CUDA'] = get_mape(a100_gemm_fp32, estimation_methods, estimation_nametags)
    a100_results['Cnv'] = get_mape(a100_conv_bf16, estimation_methods, estimation_nametags)
    a100_results['Soft'] = get_mape(a100_softmax_bf16, estimation_methods, estimation_nametags)
    a100_results['LN'] = get_mape(a100_layernorm_bf16, estimation_methods, estimation_nametags)
    a100_results['EW'] = get_mape(a100_elementwise, estimation_methods, estimation_nametags)

    a10_gemm_bf16 = os.path.join(parent_folder, 'a10_bf16bf16_tc_artifact', 'seed{}'.format(seed))
    a10_gemm_fp32 = os.path.join(parent_folder, 'a10_fp32_cuda_artifact', 'seed{}'.format(seed))
    a10_softmax_bf16 = os.path.join(parent_folder, 'a10_bf16_softmax_artifact', 'seed{}'.format(seed))
    a10_layernorm_bf16 = os.path.join(parent_folder, 'a10_bf16_layernorm_artifact', 'seed{}'.format(seed))
    a10_conv_bf16 = os.path.join(parent_folder, 'a10_conv2d_bf16bf16_tc_artifact', 'seed{}'.format(seed))
    a10_elementwise = os.path.join(parent_folder, 'a10_elementwise_artifact', 'seed{}'.format(seed))

    a10_results = {}
    a10_results['TC'] = get_mape(a10_gemm_bf16, estimation_methods, estimation_nametags)
    a10_results['CUDA'] = get_mape(a10_gemm_fp32, estimation_methods, estimation_nametags)
    a10_results['Cnv'] = get_mape(a10_conv_bf16, estimation_methods, estimation_nametags)
    a10_results['Soft'] = get_mape(a10_softmax_bf16, estimation_methods, estimation_nametags)
    a10_results['LN'] = get_mape(a10_layernorm_bf16, estimation_methods, estimation_nametags)
    a10_results['EW'] = get_mape(a10_elementwise, estimation_methods, estimation_nametags)

    colors = ['#0173b2', '#de8f05', '#029e73']

    figsize = (7.5, 2)  # Width doubled for 2 subplots

    # Create main figure with subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    time_nametags = ["Li et al.", 'NeuSight', 'Ours']

    # Call the function twice with different axes
    create_time_bar_plot(axes[0], a100_results, time_nametags, colors, title='A100-PCIE')
    create_time_bar_plot(axes[1], a10_results, time_nametags,colors,  omit_y=True, title='A10')  # Use different data for second plot

    # Add a single legend at the top of the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=13)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

# Fig. 7
def fig7_plot(parent_folder='../experiments_single/results', seed=1):
    a100_gemm_bf16 = os.path.join(parent_folder, 'yz8_bf16bf16_tc_artifact', 'seed{}'.format(seed))
    a100_softmax_bf16 = os.path.join(parent_folder, 'yz8_bf16_softmax_artifact', 'seed{}'.format(seed))
    a100_flashattn_bf16 = os.path.join(parent_folder, 'yz8_flashattention_artifact', 'seed{}'.format(seed))

    estimation_method = 'gee_v3_refFalse_dvfsFalse'

    data_configs = [
        {
            'parent_path': a100_gemm_bf16,
            'estimation_method': estimation_method,
            'filter': 'regular',  # or 'all'
            'workload': 'gemm',
            'title': 'GEMM'
        },
        {
            'parent_path': a100_softmax_bf16,
            'estimation_method': estimation_method,
            'filter': 'regular',
            'workload': 'softmax',
            'title': 'Softmax'
        },
        {
            'parent_path': a100_flashattn_bf16,
            'estimation_method': estimation_method,
            'filter': 'regular',
            'workload': 'flashattention',
            'title': 'FlashAttn'
        }
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    for idx, (ax, config) in enumerate(zip(axes, data_configs)):
        parent_path = config['parent_path']
        estimation_method = config['estimation_method']
        filter_type = config.get('filter', 'all')
        workload = config.get('workload', 'gemm')
        title = config.get('title', '')
        
        df = pd.read_csv(os.path.join(parent_path, estimation_method, 'estimation_result.csv'))
        
        if filter_type == 'regular':
            if workload == 'gemm':
                df = df.loc[(df['dimM'] >= 128) & (df['dimN'] >= 128) & (df['dimK'] >= 128)]
            elif (workload == 'softmax') or (workload == 'layernorm'):
                df = df.loc[(df['batch'] >= 128) & (df['dim'] >= 128) & (df['dim'] <= 2 ** 20)]
            elif workload == 'elementwise':
                df = df.loc[(df['dim'] < 2 ** 20) & (df['dim'] > 2 ** 10)]
        
        df['power'] = df['energy'] / df['time'] * 1000.
        df['power_estimate'] = df['energy_estimate'] / df['time_estimate'] * 1000
        
        plt.sca(ax)
        ax.scatter(df['power'], df['power_estimate'], marker='o', color='darkblue', alpha=0.5, s=70)
        ax.set_xlabel('Ground Truth (W)', fontsize=18)
        
        # Only add y-label to the first subplot
        if idx == 0:
            ax.set_ylabel('Estimated (W)', fontsize=18)
        
        ax.grid(which='both', alpha=0.5)
        
        min_power = df['power'].min()
        max_power = df['power'].max()
        ax.set_xlim(min_power - 1, max_power + 1)
        ax.set_ylim(min_power - 1, max_power + 1)
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(title, fontsize=18)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((df['power'] - df['power_estimate']) / df['power'])) * 100
        
        # Add MAPE annotation
        ax.annotate(f'MAPE\n{mape:.1f}%', 
                    xy=(0.65, 0.1), 
                    xycoords='axes fraction',
                    fontsize=16,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
        
        # Add diagonal line
        ax.plot([min_power - 1, max_power + 1], [min_power - 1, max_power + 1], 
                linestyle='--', color='red', linewidth=2)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()