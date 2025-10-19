# Wah-Coder

Yes. Use a **vocoder-style analysis → control → resonant-bank synthesis**. You extract a speech-like spectral envelope, then drive a wah/formant filterbank on the instrument.

# Architecture

1. **Analysis bank (modulator)**

   * Split the modulator (m[n]) (mic or prerecorded voice) into (K) bands with linear- or mel-spaced BPFs (A_k(z)).
   * Envelope per band:
     [
     e_k[n] = \text{LP}\left(|A_k(z),m[n]|\right)
     ]
     with rectifier + one-pole LP or attack/release detector.

2. **Mapping**

   * Gain-only control: (g_k[n] = e_k[n]^\gamma) (γ sets articulation).
   * Optional **center-frequency warping**: let each resonant band’s (f_{c,k}(n)) follow (e_k[n]) or a lower-dimensional projection (e.g., map F1/F2 estimates to two moving bands; keep others static).

3. **Synthesis bank (carrier)**

   * Run the instrument (x[n]) through (K) **resonant filters** (S_k(z; f_{c,k}(n), Q_k)) in parallel.
   * Apply band gains:
     [
     y[n] = \sum_{k=1}^{K} g_k[n]; S_k(z;,f_{c,k}(n),Q_k),x[n]
     ]
   * Optional dry mix and limiter.

4. **Control sources**

   * **Pure vocoder**: modulator = speech. Instrument = carrier.
   * **Hybrid talking-wah**: replace analysis bank with an **LPC** or **cepstral envelope** of a voice, or drive a few key formants (F1, F2, F3) while other bands are LFO/envelope-controlled.
   * **No mic**: use prerecorded vowel trajectories or HMM/RNN formant curves to set (f_{c,k}(t), g_k(t)).

# Practical choices

* Bands (K): 8–16 for intelligibility; 24+ for fidelity.
* Centers: vowel formants (≈ F1 300–900 Hz, F2 800–2500 Hz, F3 2–3.5 kHz), plus “support” bands between them.
* Filters: state-variable or RBJ biquads. Update coeffs at audio rate only if needed; otherwise every N samples.
* Envelopes: attack 3–10 ms, release 30–200 ms; log-domain smoothing improves stability.
* Latency: IIR banks are near-zero latency; FFT vocoder variant adds window delay.

# Math details

* Analysis:
  [
  e_k[n] = a_k,e_k[n-1] + (1-a_k),|u_k[n]|,;; u_k[n]=(A_k*m)[n]
  ]
  with separate (a_{\text{att}}, a_{\text{rel}}) if using A/R logic.
* Synthesis biquad (per band):
  [
  y_k[n] = b_{0,k}x_k[n] + b_{1,k}x_k[n-1] + b_{2,k}x_k[n-2] - a_{1,k}y_k[n-1] - a_{2,k}y_k[n-2]
  ]
  Coefficients from (f_{c,k}(n), Q_k); gain multiply by (g_k[n]).

# Minimal NumPy/Numba skeleton

```python
import numpy as np
from numba import njit

@njit(cache=True)
def one_pole(a, y_prev, x):
    # y[n] = a*y[n-1] + (1-a)*x[n]
    y = np.empty_like(x)
    yp = y_prev
    b = 1.0 - a
    for i in range(x.size):
        yp = a * yp + b * x[i]
        y[i] = yp
    return y, yp

@njit(cache=True)
def biquad_process(b0,b1,b2,a1,a2, z1z2, x):
    y = np.empty_like(x)
    z1, z2 = z1z2
    for i in range(x.size):
        s = x[i]
        o = b0*s + z1
        z1 = b1*s - a1*o + z2
        z2 = b2*s - a2*o
        y[i] = o
    z1z2[0], z1z2[1] = z1, z2
    return y

# Assume precomputed analysis filters A_k and synthesis filters S_k coeffs per band
# and simple gain-only mapping g_k[n] from envelopes.
@njit(cache=True)
def block_process(x, m, anal_coeffs, synth_coeffs, env_a, z_env, z_bq_anal, z_bq_synth, gamma):
    """
    x: carrier block (samples,)
    m: modulator block (samples,)
    anal_coeffs: (K, 5) -> b0,b1,b2,a1,a2 for analysis filters
    synth_coeffs: (K, 5) -> b0,b1,b2,a1,a2 for synthesis filters (fixed or updated per block)
    env_a: scalar smoothing coeff a (or per-band vector)
    z_env: (K,) envelope states
    z_bq_anal: (K,2) analysis biquad states
    z_bq_synth: (K,2) synthesis biquad states
    """
    K = anal_coeffs.shape[0]
    y_sum = np.zeros_like(x)
    for k in range(K):
        b0,b1,b2,a1,a2 = anal_coeffs[k]
        # analysis band + envelope
        ak = biquad_process(b0,b1,b2,a1,a2, z_bq_anal[k], m)
        env_rect = np.abs(ak)
        # single-pole smoothing
        env, z_env[k] = one_pole(env_a, z_env[k], env_rect)
        g = np.power(env, gamma)  # articulate

        # synthesis path
        b0,b1,b2,a1,a2 = synth_coeffs[k]
        sk = biquad_process(b0,b1,b2,a1,a2, z_bq_synth[k], x)
        y_sum += g * sk
    return y_sum
```

# Variants

* **Formant-tracked mapping**: estimate F1/F2 from the modulator (LPC or cepstrum). Drive only 2–3 resonant bands’ (f_c(t)) and (g_k(t)). Rest remain static.
* **FFT vocoder hybrid**: use STFT to get a smooth spectral envelope, then fit (g_k) per band and drive IIR resonant bank for low-latency time-domain synthesis.
* **Expressive controls**: add pedal/LFO to bias (f_{c,k}) or compress/expand envelopes: (g_k \leftarrow (g_k^\gamma) \cdot \beta + (1-\beta)) for mix.

# Key safeguards

* Normalize output power per block to avoid level jumps.
* Limit (Q_k) and coefficient update rate to keep filters stable.
* Gate very low envelopes to reduce breath/noise bleed.

This hybrid gives **vocoder intelligibility** with **wah-like immediacy and low latency**, producing clear “talking” guitar or synth.


---

# LPC

Yes. Use **LPC analysis → control mapping → resonant-bank synthesis**. The speech drives a multi-band wah (“wah-coder”).

# Signal path

1. **LPC analysis of modulator (m[n])**

   * Frame length 20–30 ms, hop 5–10 ms, pre-emphasis (x'[n]=x[n]-\alpha x[n-1]) with (\alpha\approx0.97).
   * Compute autocorrelation (r[\ell]), solve Levinson–Durbin for LPC order (p) (8–16 for 16 kHz; 12–20 for 44.1 kHz).
   * Get prediction polynomial (A(z)=1+\sum_{i=1}^{p} a_i z^{-i}).
   * Optional: convert to LSFs for stable interpolation between frames.

2. **Envelope and formants**

   * Spectral envelope (E(\omega)=\sigma^2/|A(e^{j\omega})|^2) (σ² = residual power).
   * Formant estimates: find peaks of (E(\omega)) or roots of (A(z)) near the unit circle → (F_k, B_k) (center and bandwidth).

3. **Mapping to a wah/formant bank on the carrier (x[n])**
   Two options:

   * **Gain-only vocoder mapping:** fixed band centers (f_{c,k}); per-frame gains (g_k) from envelope samples (E(\omega_k)). Low latency and stable.
   * **Resonant tracking:** set each synthesis band’s center to tracked formants (F_k) and Q from (B_k). More “talking” articulation.

4. **Synthesis**
   [
   y[n] = \sum_{k=1}^{K} g_k[n]; S_k!\left(z; f_{c,k}[n], Q_k[n]\right) , x[n]
   ]
   with parallel RBJ/SVF filters. Smooth all parameters between frames.

# Math details

* **Levinson–Durbin** for LPC:
  [
  a^{(i)}*i = \frac{r[i]-\sum*{k=1}^{i-1} a^{(i-1)}*k r[i-k]}{E*{i-1}},\quad
  a^{(i)}_k = a^{(i-1)}_k - a^{(i)}*i a^{(i-1)}*{i-k},\quad
  E_i=(1-(a^{(i)}*i)^2)E*{i-1}
  ]
* **Formant from LPC roots:** roots of (A(z)) with angles (\theta) near unit circle → (F=\theta,f_s/(2\pi)); bandwidth (B=-\frac{f_s}{\pi}\ln r) with radius (r).

# Practical settings

* Sample rate: 44.1 kHz guitar, modulator at same rate or downsampled to 16 kHz for LPC.
* Order (p): 14–18 @ 22–44 kHz.
* Frames: 1024 samples @ 44.1 kHz (~23 ms), hop 256–512.
* Bands (K): 12–16 mel/erb spaced for gain-only; 3–5 bands if tracking F1–F3(+F4).
* Smoothing: 5–15 ms attack, 30–150 ms release on (g_k). Interpolate LSFs linearly per sample.

# Numba skeleton

```python
import numpy as np
from numba import njit

@njit(cache=True)
def levinson_durbin(r, p):
    a = np.zeros(p+1)
    e = r[0]
    if e <= 1e-12:
        return a, e
    k = 0.0
    for i in range(1, p+1):
        acc = 0.0
        for j in range(1, i):
            acc += a[j]*r[i-j]
        k = (r[i] - acc) / e
        a_new = a.copy()
        a_new[i] = k
        for j in range(1, i):
            a_new[j] = a[j] - k*a[i-j]
        a = a_new
        e *= (1.0 - k*k)
    return a, e  # a[0]=0 here; prepend 1.0 when forming A(z)

@njit(cache=True)
def lpc_autocorr(x, p):
    N = x.size
    r = np.zeros(p+1)
    for k in range(p+1):
        s = 0.0
        for n in range(k, N):
            s += x[n]*x[n-k]
        r[k] = s
    return r

@njit(cache=True)
def env_from_lpc(a, sigma2, fft_bins, fs):
    # Evaluate E(ω)=σ²/|A(e^{jw})|² at bin centers
    K = fft_bins.size
    env = np.empty(K)
    for i in range(K):
        w = 2*np.pi*fft_bins[i]/fs
        # Horner on unit circle
        re = 1.0
        im = 0.0
        cw = np.cos(w); sw = np.sin(w)
        zr = 1.0; zi = 0.0
        for k in range(1, a.size):
            # z^{-k}
            # accumulate A(e^{-jw}) = 1 + sum a[k] e^{-jwk}
            zr_new = zr*cw + zi*sw
            zi = zi*cw - zr*sw
            zr = zr_new
            re += a[k]*zr
            im += a[k]*zi
        mag2 = re*re + im*im
        env[i] = sigma2 / max(mag2, 1e-12)
    return env

@njit(cache=True)
def one_pole(a, y_prev, x):
    y = np.empty_like(x)
    yp = y_prev
    b = 1.0 - a
    for i in range(x.size):
        yp = a*yp + b*x[i]
        y[i] = yp
    return y, yp
```

# Block flow (gain-only mapping)

1. Precompute (K) synthesis filters (S_k) centers on mel/erb grid 200 Hz–4 kHz with suitable (Q_k) (e.g., 8–12).
2. For each speech frame:

   * Pre-emphasize and window; compute (r), then LPC (a[1..p]), residual power (\sigma^2).
   * Sample the LPC envelope at band centers → raw gains (g_k).
   * Log-compress and normalize: (g_k \leftarrow \exp(\gamma \log(g_k+\epsilon))), then RMS-normalize across bands.
   * Interpolate (g_k) over the next hop at audio rate; apply attack/release smoothing.
3. Run carrier through the parallel resonant bank and multiply each band by smoothed (g_k[n]). Sum and limit.

# Variant: formant-tracked

* Extract F1–F3 via LPC root finding. Map them to 3 synthesis bands’ centers (f_{c,k}(n)). Keep support bands with gain-only control for clarity.
* Interpolate LSFs or formant frequencies to prevent coefficient jumps.

# Latency and stability

* Latency ≈ analysis window/2 (+ hop scheduling). IIR synthesis is near zero-latency.
* Use LSF interpolation or per-frame coefficient cross-fade to avoid pops.
* Clamp (Q) and frequency ranges to guitar-friendly bands.

# Outcome

This “wah-coder” yields intelligible vowel articulation like a vocoder, with the **tone** of a resonant multi-wah. It works live if you keep short frames, modest smoothing, and stable filter updates.

# The Peter Frampton Equations

Yes — that’s correct. The device in question is called a Talk Box. It routes the instrument’s sound (often guitar) through a tube into the performer’s mouth, which they shape with their lips/tongue while a microphone picks up the resulting “talking instrument” sound. ([Wikipedia][1])

![Image](https://i.ytimg.com/vi/KcHR-tI0Pj4/hqdefault.jpg)

![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Weezer_-_2022154163929_2022-06-03_Rock_am_Ring_-_Sven_-_1D_X_MK_II_-_1281_-_B70I5835.jpg/250px-Weezer_-_2022154163929_2022-06-03_Rock_am_Ring_-_Sven_-_1D_X_MK_II_-_1281_-_B70I5835.jpg)

![Image](https://i0.wp.com/www.vintageguitar.com/wp-content/uploads/HEIL_01.jpg)

### Key historical facts

* The talk box was used in various forms since the early/mid-20th century (e.g., Alvino Rey, Pete Drake) for “talking guitar” effects. ([Wikipedia][1])
* The manufacturer/engineer Bob Heil built a high-powered version in the early 1970s; it was used by Joe Walsh and became popular. ([Loudwire][2])
* Peter Frampton is strongly associated with the effect — his 1976 live album Frampton Comes Alive! features it prominently (songs like “Show Me the Way”, “Do You Feel Like We Do”). ([Wikipedia][3])
* Frampton credits Pete Drake with introducing him to the talk box and got a Heil model for Christmas in 1974. ([Forbes][4])

So yes: the “guitar talk box” you referenced is real and was made famous in rock partly through Peter Frampton.

[1]: https://en.wikipedia.org/wiki/Talk_box?utm_source=chatgpt.com "Talk box"
[2]: https://loudwire.com/bob-heil-talk-box-dead/?utm_source=chatgpt.com "Talk Box Inventor Bob Heil Dead at 83 - Loudwire"
[3]: https://en.wikipedia.org/wiki/Peter_Frampton?utm_source=chatgpt.com "Peter Frampton"
[4]: https://www.forbes.com/sites/pamwindsor/2022/05/03/rock-legend-peter-frampton-recalls-steel-guitarist-pete-drake-introducing-him-to-the-talk-box/?utm_source=chatgpt.com "Peter Frampton Recalls Steel Guitarist Pete Drake Introducing Him ..."
