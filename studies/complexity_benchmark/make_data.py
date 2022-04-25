from timeit import default_timer as timer

import numpy as np
import pandas as pd

import neurokit2 as nk


# Utility function
def time_function(
    x,
    fun=nk.fractal_petrosian,
    index="FD_Petrosian",
    name="nk_fractal_petrosian",
    **kwargs,
):
    t0 = timer()
    rez, _ = fun(x, **kwargs)
    return pd.DataFrame(
        {
            "Duration": [timer() - t0],
            "Result": [rez],
            "Index": [index],
            "Method": [name],
        }
    )


# Parameters
# df = pd.read_csv("data_Signals.csv")

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "Random-Walk"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=10, dimension=3), show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "lorenz_10_2.5_28"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=10, dimension=3), show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "lorenz_20_2_30"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=10, dimension=3), show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "oscillatory"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=10, dimension=3), show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)

# ================
# Generate Signal
# ================
def run_benchmark(noise_intensity=0.01):
    # Initialize data storage
    data_signal = []
    data_complexity = []

    print("Noise intensity: {}".format(noise_intensity))
    for duration in [0.5, 1, 2]:
        for method in ["Random-Walk", "lorenz_10_2.5_28", "lorenz_20_2_30", "oscillatory"]:
            if method == "Random-Walk":
                delay = 10
                k = 30
                signal = nk.complexity_simulate(
                    duration=duration,
                    sampling_rate=1000,
                    method="random",
                )
            elif method == "lorenz_10_2.5_28":
                delay = 4
                k = 30
                signal = nk.complexity_simulate(
                    duration=duration * 2,
                    sampling_rate=500,
                    method="lorenz",
                    sigma=10.0,
                    beta=2.5,
                    rho=28.0,
                )
            elif method == "lorenz_20_2_30":
                delay = 15
                k = 30
                signal = nk.complexity_simulate(
                    duration=duration * 2,
                    sampling_rate=500,
                    method="lorenz",
                    sigma=20.0,
                    beta=2,
                    rho=30.0,
                )

            elif method == "oscillatory":
                delay = 10
                k = 30
                signal = nk.signal_simulate(
                    duration=duration,
                    sampling_rate=1000,
                    frequency=[5, 11, 18, 24, 42, 60, 63],
                )

            # Standardize
            signal = nk.standardize(signal)

            # Add Noise
            for noise in np.linspace(-2, 2, 5):
                noise_ = nk.signal_noise(duration=duration, sampling_rate=1000, beta=noise)
                signal_ = nk.standardize(signal + (nk.standardize(noise_) * noise_intensity))

                # Save the signal to visualize the type of signals fed into the benchmarking
                if duration == 1:

                    data_signal.append(
                        pd.DataFrame(
                            {
                                "Signal": signal_,
                                "Length": len(signal_),
                                "Duration": range(1, len(signal_) + 1),
                                "Noise": noise,
                                "Noise_Intensity": noise_intensity,
                                "Method": method,
                            }
                        )
                    )

            # ================
            # Complexity
            # ================

            rez = pd.DataFrame(
                {
                    "Duration": [np.nan, np.nan, np.nan],
                    "Result": [np.nanstd(signal_), noise_intensity, len(signal_)],
                    "Index": ["SD", "Noise", "Length"],
                    "Method": ["np.nanstd", "noise", "len"],
                }
            )

            # Fractals
            # ----------
            for x in ["A", "B", "C", "D", "r", 3, 10, 100, 1000]:
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.fractal_petrosian,
                            index=f"PFD ({x})",
                            name="nk_fractal_petrosian",
                            method=x,
                        ),
                    ]
                )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_sevcik,
                        index="SFD",
                        name="nk_fractal_sevcik",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_katz,
                        index="KFD",
                        name="nk_fractal_katz",
                    ),
                ]
            )

            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_sda,
                        index="SDAFD",
                        name="nk_fractal_sda",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_nld,
                        index="NLDFD",
                        name="nk_fractal_nld",
                        corrected=False,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_psdslope,
                        index="PSDFD (Voss1998)",
                        name="nk_fractal_psdslope",
                        method="voss1988",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_psdslope,
                        index="PSDFD (Hasselman2013)",
                        name="nk_fractal_psdslope",
                        method="hasselman2013",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_higuchi,
                        index="HFD",
                        name="nk_fractal_higuchi",
                        k_max=k,
                    ),
                ]
            )

            # Entropy
            # ----------
            for x in ["A", "B", "C", "D", "r", 3, 10, 100, 1000]:
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_shannon,
                            method=x,
                            index=f"ShanEn ({x})",
                            name="nk_entropy_shannon",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_cumulative_residual,
                            method=x,
                            index=f"CREn ({x})",
                            name="nk_entropy_cumulative_residual",
                        ),
                    ]
                )

            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_svd,
                        index="SVDEn",
                        name="nk_entropy_svd",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_kolmogorov,
                        index="K2En",
                        name="nk_entropy_kolmogorov",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_attention,
                        index="AttEn",
                        name="nk_entropy_attention",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_approximate,
                        index="ApEn",
                        name="entropy_approximate",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_approximate,
                        index="cApEn",
                        name="entropy_approximate",
                        delay=delay,
                        dimension=3,
                        corrected=True,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_differential,
                        index="DiffEn",
                        name="nk_entropy_differential",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_distribution,
                        index="DistrEn",
                        name="nk_entropy_distribution",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_spectral,
                        index="SPEn",
                        name="entropy_spectral",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_sample,
                        index="SampEn",
                        name="nk_entropy_sample",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_fuzzy,
                        index="FuzzyEn",
                        name="nk_entropy_fuzzy",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_fuzzy,
                        index="FuzzyApEn",
                        name="nk_entropy_fuzzy",
                        delay=delay,
                        dimension=3,
                        approximate=True,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_fuzzy,
                        index="FuzzycApEn",
                        name="nk_entropy_fuzzy",
                        delay=delay,
                        dimension=3,
                        approximate=True,
                        corrected=True,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_range,
                        index="RangeEn (A)",
                        name="entropy_range",
                        delay=delay,
                        dimension=3,
                        approximate=True,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_range,
                        index="RangeEn (Ac)",
                        name="entropy_range",
                        delay=delay,
                        dimension=3,
                        approximate=True,
                        corrected=True,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_range,
                        index="RangeEn (B)",
                        name="entropy_range",
                        delay=delay,
                        dimension=3,
                        approximate=False,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_permutation,
                        index="PEn",
                        name="nk_entropy_permutation",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_permutation,
                        index="WPEn",
                        name="nk_entropy_permutation",
                        delay=delay,
                        dimension=3,
                        weighted=True,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_permutation,
                        index="CPEn",
                        name="nk_entropy_permutation",
                        delay=delay,
                        dimension=3,
                        conditional=True,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_permutation,
                        index="CWPEn",
                        name="nk_entropy_permutation",
                        delay=delay,
                        dimension=3,
                        weighted=True,
                        conditional=True,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_permutation,
                        index="CRPEn",
                        name="nk_entropy_permutation",
                        delay=delay,
                        dimension=3,
                        conditional=True,
                        algorithm=nk.entropy_renyi,
                        alpha=2,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_bubble,
                        index="BubbEn",
                        name="nk_entropy_bubble",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_cosinesimilarity,
                        index="CoSiEn",
                        name="nk_entropy_cosinesimilarity",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="MSCoSiEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                        method="MSCoSiEn",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="MSEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                        method="MSEn",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="CMSEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                        method="CMSEn",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="RCMSEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                        method="RCMSEn",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="MMSEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                        method="MMSEn",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="IMSEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                        method="IMSEn",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="MSApEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                        method="MSApEn",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="MSPEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                        method="MSPEn",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="CMSPEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                        method="CMSPEn",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="MMSPEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                        method="MMSPEn",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="IMSPEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                        method="IMSPEn",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="MSWPEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                        method="MSWPEn",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="CMSWPEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                        method="CMSWPEn",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="MMSWPEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                        method="MMSWPEn",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="IMSWPEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                        method="IMSWPEn",
                    ),
                ]
            )
            # rez = pd.concat(
            #     [
            #         rez,
            #         time_function(
            #             signal_,
            #             nk.entropy_multiscale,
            #             index="FuzzyMSEn",
            #             name="nk_entropy_multiscale",
            #             delay=delay,
            #             dimension=3,
            #             method="MSEn",
            #             fuzzy=True,
            #         ),
            #     ]
            # )
            # rez = pd.concat(
            #     [
            #         rez,
            #         time_function(
            #             signal_,
            #             nk.entropy_multiscale,
            #             index="FuzzyCMSEn",
            #             name="nk_entropy_multiscale",
            #             delay=delay,
            #             dimension=3,
            #             method="CMSEn",
            #             fuzzy=True,
            #         ),
            #     ]
            # )
            # rez = pd.concat(
            #     [
            #         rez,
            #         time_function(
            #             signal_,
            #             nk.entropy_multiscale,
            #             index="FuzzyRCMSEn",
            #             name="nk_entropy_multiscale",
            #             delay=delay,
            #             dimension=3,
            #             method="RCMSEn",
            #             fuzzy=True,
            #         ),
            #     ]
            # )
            # rez = pd.concat(
            #     [
            #         rez,
            #         time_function(
            #             signal_,
            #             nk.entropy_multiscale,
            #             index="FuzzyMMSEn",
            #             name="nk_entropy_multiscale",
            #             delay=delay,
            #             dimension=3,
            #             method="MMSEn",
            #             fuzzy=True,
            #         ),
            #     ]
            # )
            # rez = pd.concat(
            #     [
            #         rez,
            #         time_function(
            #             signal_,
            #             nk.entropy_multiscale,
            #             index="FuzzyIMSEn",
            #             name="nk_entropy_multiscale",
            #             delay=delay,
            #             dimension=3,
            #             method="IMSEn",
            #             fuzzy=True,
            #         ),
            #     ]
            # )

            # Other
            # ----------
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.complexity_hjorth,
                        index="Hjorth",
                        name="nk_complexity_hjorth",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fisher_information,
                        index="FI",
                        name="nk_fisher_information",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.complexity_rr,
                        index="RR",
                        name="nk_complexity_rr",
                    ),
                ]
            )

            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.complexity_hurst,
                        index="H (corrected)",
                        name="nk_complexity_hurst",
                        corrected=True,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.complexity_hurst,
                        index="H (uncorrected)",
                        name="nk_complexity_hurst",
                        corrected=False,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_correlation,
                        index="CD",
                        name="nk_fractal_correlation",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            # rez = pd.concat(
            #     [
            #         rez,
            #         time_function(
            #             signal_,
            #             nk.complexity_lempelziv,
            #             index="LZC",
            #             name="nk_complexity_lempelziv",
            #             delay=delay,
            #             dimension=3,
            #         ),
            #     ]
            # )
            # rez = pd.concat(
            #     [
            #         rez,
            #         time_function(
            #             signal_,
            #             nk.complexity_lempelziv,
            #             index="PLZC",
            #             name="nk_complexity_lempelziv",
            #             delay=delay,
            #             dimension=3,
            #             permutation=True,
            #         ),
            #     ]
            # )

            # Add info
            rez["Length"] = len(signal_)
            rez["Noise_Type"] = noise
            rez["Noise_Intensity"] = noise_intensity
            rez["Signal"] = method

            data_complexity.append(rez)
    return pd.concat(data_signal), pd.concat(data_complexity)


out = nk.parallel_run(
    run_benchmark,
    [{"noise_intensity": i} for i in np.linspace(0.01, 3, 16)],
    n_jobs=8,
    verbose=5,
)

pd.concat([out[i][0] for i in range(len(out))]).to_csv("data_Signals.csv", index=False)
pd.concat([out[i][1] for i in range(len(out))]).to_csv("data_Complexity.csv", index=False)


print("FINISHED.")