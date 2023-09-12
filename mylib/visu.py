import numpy as np
import pylab
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import animation
from mylib.parameters import Params
import matplotlib

# set plt style with dark background
#plt.style.use('dark_background')
# add grid to plot with opacity 0.3
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.ioff()
matplotlib.rcParams['interactive'] = False

def view1D_time_state(time: int, flow_history: np.ndarray, params: Params, xlim=None, title=None):
    
    NT = params.get('NT')
    NX = params.get('NX')
    DT = params.get('DT')

    index = np.abs(NT - time).argmin()
    # if xlim is not None:
    #     index1 = np.abs(NT - xlim[0]).argmin()
    #     index2 = np.abs(NT - xlim[1]).argmin()
    # plt.xlim()
    fig, ax = plt.subplots()
    ax.plot(NX, flow_history[:, index])
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    if title is None:
        title = f"Time : {NT[index]:.4f}"
    ax.set_title(title)
    return fig

def view1D_pos_state(pos: int, flow_history: np.ndarray, params: Params):
    NT = params.get('NT')
    NX = params.get('NX')
    
    index = np.abs(NX - pos).argmin()
    fig, ax = plt.subplots()
    ax.plot(NT, flow_history[index, :])
    ax.set_xlabel("Time")
    ax.set_ylabel("Velocity")
    ax.set_title(f"Position : {NX[index]:.2f}")
    return fig

def view2D(flow, params: Params, title=""):

    L = params.get('L')
    duree = params.get('T')

    def space_formatter(x, pos):
        return '{:.2f}'.format(x * L / flow.shape[0])
    
    def time_formatter(x, pos):
        return '{:.2f}'.format(x * duree / flow.shape[1])

    fig, axes = pylab.subplots(1, 1, figsize=(16, 5))
    im = axes.imshow(flow, origin='upper', cmap='inferno')
    axes.yaxis.set_major_formatter(space_formatter)
    axes.xaxis.set_major_formatter(time_formatter)
    pylab.colorbar(im) ; pylab.xlabel('time (s)'); pylab.ylabel('position x'); pylab.title(title)

def compute_spectre(flow, params: Params, dim="time", fixe_point='all'):
    """
    Compute the Fast Fourier Transform of a signal flow
    :param flow: signal
    :param params: parameters of the simulation, type Params
    :param dim: dimension of fixe_point (can be 'time' or 'space')
    :param fixe_point: fixed point where the wavenumber is observed (can be 'all' or int)
    :return: freq, fft, freq, PSD, index
    """

    DT = params.get('DT')
    NT = params.get('NT')
    NX = params.get('NX')
    index = 0
    u = flow.copy()

    if dim == "time":
        axis = 1
    elif dim == "space":
        axis = 0

    if fixe_point == 'all':
        u = np.mean(u, axis=axis).reshape(-1)
    else:
        if dim == "time":
            index = np.abs(NT - fixe_point).argmin()
            u = u[:, index].reshape(-1)
        elif dim == "space":
            index = np.abs(NX - fixe_point).argmin()
            u = u[index, :].reshape(-1)
            
    N = u.size
    freq = np.fft.fftfreq(N, d=DT)
    fft = np.fft.fft(u)
    PSD = np.abs(fft)**2
    return np.abs(freq[:N//2]), np.abs(fft[:N//2]), freq, PSD, index

def show_spectre(params, data: tuple, xlim=None, ylim=None, dim="time", klim=(10,1000), ky=10e7, title=None):
    """
    freq, fft, freq_psd, PSD, index = data
    """
    freq, fft, freq_psd, PSD, index = data


    NT = params.get('NT')
    NX = params.get('NX')

    fig, axes = plt.subplots(ncols=2, figsize=(15,4))
    if title is not None:
        fig.suptitle(title)
    else:
        if dim == 'time':
            fig.suptitle(f"Spectre of the signal at time t = {NT[index]:.2f}")
        elif dim == 'space':
            fig.suptitle(f"Spectre of the signal at position x = {NX[index]:.2f}")
    axes[0].set_title("fft")
    axes[1].set_title("psd")
    axes[0].set_xlabel("k")
    axes[1].set_xlabel("k")
    axes[0].set_ylabel("fft")
    axes[1].set_ylabel("psd")
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[0].set_xscale('log')
    #axes[1].set_xscale('log')
    if xlim is not None:
        axes[0].set_xlim(left=0, right=xlim)
        axes[1].set_xlim(left=-xlim, right=xlim)
    if ylim is not None:    
        axes[0].set_ylim(bottom=ylim)
        axes[1].set_ylim(bottom=ylim)
    axes[0].plot(freq, np.abs(fft))
    axes[1].plot(freq_psd, PSD)

    # keep only the kmin < k < kmax
    axes[0].plot(freq[int(klim[0]):int(klim[1])], int(ky) * freq[int(klim[0]):int(klim[1])]**(-5/3))

    if dim == 'time':
        axes[0].set_title(f"FFT")
    elif dim == 'space':
        axes[0].set_title(f"PSD")
    plt.show()

    return fig, axes

def spectre(flow, params: Params, dim="time", fixe_point='all', xlim=None, ylim=None, klim=(1e1, 1e3), ky=1e3):
    data = compute_spectre(flow, params, dim, fixe_point)
    fig, axes = show_spectre(params, data, xlim, ylim, dim, klim, ky)
    return fig, axes

def animate_flow(flow_history_, params: Params, name='out', xlim=None, fps=200):
    """
    Animate using matplotlib animation
    """
    NX = params.get('NX')
    DT = params.get('DT')
    L = params.get('L')

    if xlim is None:
        xlim = (0, L)

    fig = plt.figure()
    #ax = fig.add_subplot(ylim=(-np.max(flow_history_), np.max(flow_history_)), xlim=xlim)
    ax = plt.axes(ylim=(-np.max(flow_history_)*1.1, np.max(flow_history_*1.1)), xlim=xlim)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    line, = ax.plot([], [], lw=2)

    # animation function.  This is called sequentially
    def animate(i):
        line.set_data(NX, flow_history_[:, i].reshape(-1))
        return line

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=tqdm(range(flow_history_.shape[1]), initial=0), interval=fps, blit=False)
    anim.save(name + '.gif', fps=int(1/DT));

def animate_spectre(flow_history_, params, name='out_spectre', xlim_fft=(None, None), xlim_psd=(None, None),  ylim_fft=(None, None), ylim_psd=(None, None), fps=60):
    DT = params.get('DT')
    N = params.get('N')

    spectre_fft_history = []
    spectre_psd_history = []
    for t in params.get('NT'):
        freq, fft, freq_, PSD, index = compute_spectre(flow_history_, params, dim='time', fixe_point=t, show=False)
        spectre_fft_history.append(fft.reshape(-1,1))
        spectre_psd_history.append(PSD.reshape(-1,1))

    spectre_fft_history_ = np.concatenate(spectre_fft_history, axis=1)
    spectre_psd_history_ = np.concatenate(spectre_psd_history, axis=1)

    fig = plt.figure()
    fig, axes = plt.subplots(ncols=2, figsize=(15,4))
    axes[0].set_ylim(ylim_fft[0], ylim_fft[1])
    axes[1].set_ylim(ylim_psd[0], ylim_psd[1])
    axes[0].set_xlim(xlim_fft[0], xlim_fft[1])
    axes[1].set_xlim(xlim_psd[0], xlim_psd[1])
    axes[0].set_title("fft")
    axes[1].set_title("psd")
    axes[0].set_xlabel("k")
    axes[1].set_xlabel("k")
    axes[0].set_ylabel("E(k)")
    axes[1].set_ylabel("psd")
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[0].set_xscale('log')
 
    line, = axes[0].plot([], [], lw=2)
    line2, = axes[1].plot([], [], lw=2)

    print(type(line), type(line2))

    # animation function.  This is called sequentially
    def animate(i):
        fig.suptitle(f"t={i*DT:.2f}")
        line.set_data(freq[:N//2], np.abs(spectre_fft_history_[:N//2, i]))
        line2.set_data(freq_, spectre_psd_history_[:, i])
        return line, line2
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=tqdm(range(spectre_fft_history_.shape[1]), initial=0), interval=fps, blit=False)
    anim.save(name + '.gif', fps=int(1/DT));

def merge_figure(fig_list, legend=None, axe_index=0, figsize=(12,5), xlim=None, ylim=None, title=None):
    
    #Create new figure
    new_fig = plt.figure(figsize=figsize)
    new_axes = new_fig.add_subplot(1,1,1)

    i = 0
    for fig in fig_list:
    
        axes = fig.get_axes()

        ax = axes[axe_index]

        x, y = ax.get_lines()[0].get_data()

        if xlim is not None:
            new_axes.set_xlim(xlim)
        
        if ylim is not None:
            new_axes.set_ylim(ylim)

        # Set same scale than first figure
        new_axes.set_xscale(ax.get_xscale())
        new_axes.set_yscale(ax.get_yscale())

        # Set same labels than first figure
        new_axes.set_xlabel(ax.get_xlabel())
        new_axes.set_ylabel(ax.get_ylabel())
        
        # Set same title than first figure
        if title is None:
            title = ax.get_title()
        
        new_axes.set_title(title)

        # Add data of first figure
        new_axes.plot(x, y, label=legend[i])
        i += 1

    # Set legend
    new_fig.legend()
    # Show the figure
    plt.show()
    return new_fig

def plot_k2(klim=(10,1000), ky=100):
    freq = np.array([    0.,       195.3125,   390.625,    585.9375,   781.25,     976.5625,  1171.875,   1367.1875,  1562.5  ,   1757.8125,  1953.125,   2148.4375,  2343.75 ,   2539.0625,  2734.375,   2929.6875,  3125.   ,   3320.3125,  3515.625,   3710.9375,  3906.25 ,   4101.5625,  4296.875,   4492.1875,  4687.5  ,   4882.8125,  5078.125,   5273.4375,  5468.75 ,   5664.0625,  5859.375,   6054.6875,  6250.   ,   6445.3125,  6640.625,   6835.9375,  7031.25 ,   7226.5625,  7421.875,   7617.1875,  7812.5  ,   8007.8125,  8203.125,   8398.4375,  8593.75 ,   8789.0625,  8984.375,   9179.6875,  9375.   ,   9570.3125,  9765.625,   9960.9375, 10156.25 ,  10351.5625, 10546.875,  10742.1875, 10937.5  ,  11132.8125, 11328.125,  11523.4375, 11718.75 ,  11914.0625, 12109.375,  12304.6875, 12500.   ,  12695.3125, 12890.625,  13085.9375, 13281.25 ,  13476.5625, 13671.875,  13867.1875, 14062.5  ,  14257.8125, 14453.125,  14648.4375, 14843.75 ,  15039.0625, 15234.375,  15429.6875, 15625.   ,  15820.3125, 16015.625,  16210.9375, 16406.25 ,  16601.5625, 16796.875,  16992.1875, 17187.5  ,  17382.8125, 17578.125,  17773.4375, 17968.75 ,  18164.0625, 18359.375,  18554.6875, 18750.   ,  18945.3125, 19140.625,  19335.9375, 19531.25 ,  19726.5625, 19921.875,  20117.1875, 20312.5  ,  20507.8125, 20703.125,  20898.4375, 21093.75 ,  21289.0625, 21484.375,  21679.6875, 21875.   ,  22070.3125, 22265.625,  22460.9375, 22656.25 ,  22851.5625, 23046.875,  23242.1875, 23437.5  ,  23632.8125, 23828.125,  24023.4375, 24218.75 ,  24414.0625, 24609.375,  24804.6875, 25000.   ,  25195.3125, 25390.625,  25585.9375, 25781.25 ,  25976.5625, 26171.875,  26367.1875, 26562.5  ,  26757.8125, 26953.125,  27148.4375, 27343.75 ,  27539.0625, 27734.375,  27929.6875, 28125.   ,  28320.3125, 28515.625,  28710.9375, 28906.25 ,  29101.5625, 29296.875,  29492.1875, 29687.5  ,  29882.8125, 30078.125,  30273.4375, 30468.75 ,  30664.0625, 30859.375,  31054.6875, 31250.   ,  31445.3125, 31640.625,  31835.9375, 32031.25 ,  32226.5625, 32421.875,  32617.1875, 32812.5  ,  33007.8125, 33203.125,  33398.4375, 33593.75 ,  33789.0625, 33984.375,  34179.6875, 34375.   ,  34570.3125, 34765.625,  34960.9375, 35156.25 ,  35351.5625, 35546.875,  35742.1875, 35937.5  ,  36132.8125, 36328.125,  36523.4375, 36718.75 ,  36914.0625, 37109.375,  37304.6875, 37500.   ,  37695.3125, 37890.625,  38085.9375, 38281.25 ,  38476.5625, 38671.875,  38867.1875, 39062.5  ,  39257.8125, 39453.125,  39648.4375, 39843.75 ,  40039.0625, 40234.375,  40429.6875, 40625.   ,  40820.3125, 41015.625,  41210.9375, 41406.25 ,  41601.5625, 41796.875,  41992.1875, 42187.5  ,  42382.8125, 42578.125,  42773.4375, 42968.75 ,  43164.0625, 43359.375,  43554.6875, 43750.   ,  43945.3125, 44140.625,  44335.9375, 44531.25 ,  44726.5625, 44921.875,  45117.1875, 45312.5  ,  45507.8125, 45703.125,  45898.4375, 46093.75 ,  46289.0625, 46484.375,  46679.6875, 46875.   ,  47070.3125, 47265.625,  47460.9375, 47656.25 ,  47851.5625, 48046.875,  48242.1875, 48437.5  ,  48632.8125, 48828.125,  49023.4375, 49218.75 ,  49414.0625, 49609.375,  49804.6875])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(freq[int(klim[0]):int(klim[1])], int(ky) * freq[int(klim[0]):int(klim[1])]**(-5/3))
    ax.loglog()
    return fig, ax
