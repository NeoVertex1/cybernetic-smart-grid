# cybernetic-smart-grid
Simulated a cybernetic smart grid where imbalance is redistributed—not erased—via harmonic stabilization and damping. Total system energy is conserved. Control equation derived and visualized.

![plot](images/smartgrid_combined_summary_larger_fonts.png)

---

### Harmonic Power Flow Controller

```
power_flow(t) = [(k * (target1 - I1(t)) - k * (target2 - I2(t))) / 2] * exp(-t / τ) * H
```

Where:  
- I1(t), I2(t): imbalance at Node 1 and Node 2 at time t  
- target1, target2: dynamic power targets  
- k: proportional gain  
- τ: coherence damping time  
- H: harmonic stabilization constant (≈ 0.9999206)

---

### Dynamic Target Calculation

```
target = (I1 + I2 + ... + In) / N_adjustable
```

Applies only over adjustable (non-fixed) nodes.

---

### Imbalance Update Rule

```
I_i(t+1) = I_i(t) + ΔP_i(t)
```

Where ΔP_i(t) is the power flow (positive = gain, negative = loss) for node i.

---

### Total System Imbalance (Invariant)

```
|I1| + |I2| + |I3| + |I4| = 2600 W
```

This value remains constant throughout the simulation. No power is created or destroyed — it is only redistributed.

---

